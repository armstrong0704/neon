//! Implementation of append-only file data structure
//! used to keep in-memory layers spilled on disk.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use camino::Utf8PathBuf;
use pageserver_api::shard::TenantShardId;
use tokio_epoll_uring::{BoundedBuf, Slice};
use tokio_util::sync::CancellationToken;
use utils::id::TimelineId;
use utils::sync::gate::GateGuard;

use crate::assert_u64_eq_usize::{U64IsUsize, UsizeIsU64};
use crate::config::PageServerConf;
use crate::context::RequestContext;
use crate::page_cache;
use crate::tenant::storage_layer::inmemory_layer::GlobalResourceUnits;
use crate::tenant::storage_layer::inmemory_layer::vectored_dio_read::File;
use crate::virtual_file::owned_buffers_io::io_buf_aligned::IoBufAlignedMut;
use crate::virtual_file::owned_buffers_io::io_buf_ext::FullSlice;
use crate::virtual_file::owned_buffers_io::slice::SliceMutExt;
use crate::virtual_file::{self, IoBufferMut, TempVirtualFile, VirtualFile};

use arc_swap::ArcSwap;

pub struct EphemeralFile {
    _tenant_shard_id: TenantShardId,
    _timeline_id: TimelineId,
    page_cache_file_id: page_cache::FileId,

    /// The underlying file.
    /// Concurrent reads via pread are fine, but writes must be serialized.
    file: TempVirtualFileCoOwned,

    /// 1. The Mutable Tail.
    /// New writes go here first. Protected by RwLock, but we only hold it
    /// for memory copies. never hold during IO.
    mutable_state: tokio::sync::RwLock<MutableState>,

    /// 2. The Frozen Buffer.
    /// When tail is full, we move it here to flush.
    /// Readers grab this lock-free via ArcSwap.
    frozen_buffer: ArcSwap<Option<FrozenState>>,

    /// 3. Writer Lock.
    /// Ensure single-writer semantics. This protects the frozen buffer from
    /// being clobbered if multiple writers race. Readers dont touch this.
    writer_lock: tokio::sync::Mutex<()>,

    /// Logical size (Memory + Disk).
    /// Updates immediately so readers know data is avail.
    bytes_written: AtomicU64,

    /// Bytes safely fsync'd to disk.
    /// Readers check this boundary to decide when to read from file vs buffer.
    disk_committed_offset: AtomicU64,

    resource_units: std::sync::Mutex<GlobalResourceUnits>,
}

struct MutableState {
    buffer: IoBufferMut,
    start_offset: u64,
}

/// Buffer state during flush.
/// Store offset explicitly to avoid race with disk_committed_offset updates.
struct FrozenState {
    offset: u64,
    buffer: IoBufferMut,
}

#[derive(Debug, Clone)]
struct TempVirtualFileCoOwned {
    inner: Arc<TempVirtualFile>,
}

const TAIL_SZ: usize = 64 * 1024;

impl EphemeralFile {
    pub async fn create(
        conf: &PageServerConf,
        tenant_shard_id: TenantShardId,
        timeline_id: TimelineId,
        gate: &utils::sync::gate::Gate,
        _cancel: &CancellationToken,
        ctx: &RequestContext,
    ) -> anyhow::Result<EphemeralFile> {
        static NEXT_TEMP_DISAMBIGUATOR: AtomicU64 = AtomicU64::new(1);
        let filename_disambiguator =
            NEXT_TEMP_DISAMBIGUATOR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let filename = conf
            .timeline_path(&tenant_shard_id, &timeline_id)
            .join(Utf8PathBuf::from(format!(
                "ephemeral-{filename_disambiguator}"
            )));

        let file = TempVirtualFileCoOwned::new(
            VirtualFile::open_with_options_v2(
                &filename,
                virtual_file::OpenOptions::new()
                    .create_new(true)
                    .read(true)
                    .write(true),
                ctx,
            )
            .await?,
            gate.enter()?,
        );

        let page_cache_file_id = page_cache::next_file_id();

        Ok(EphemeralFile {
            _tenant_shard_id: tenant_shard_id,
            _timeline_id: timeline_id,
            page_cache_file_id,
            file: file.clone(),

            mutable_state: tokio::sync::RwLock::new(MutableState {
                buffer: IoBufferMut::with_capacity(TAIL_SZ),
                start_offset: 0,
            }),

            frozen_buffer: ArcSwap::from_pointee(None),
            writer_lock: tokio::sync::Mutex::new(()),

            bytes_written: AtomicU64::new(0),
            disk_committed_offset: AtomicU64::new(0),
            resource_units: std::sync::Mutex::new(GlobalResourceUnits::new()),
        })
    }
}

impl TempVirtualFileCoOwned {
    fn new(file: VirtualFile, gate_guard: GateGuard) -> Self {
        Self {
            inner: Arc::new(TempVirtualFile::new(file, gate_guard)),
        }
    }
}

impl std::ops::Deref for TempVirtualFileCoOwned {
    type Target = VirtualFile;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum EphemeralFileWriteError {
    #[error("cancelled")]
    Cancelled,
}

impl EphemeralFile {
    pub(crate) fn len(&self) -> u64 {
        // Logical size. Readers see this and might try to read data that
        // is still in mutable/frozen buffers.
        self.bytes_written.load(Ordering::Acquire)
    }

    pub(crate) fn page_cache_file_id(&self) -> page_cache::FileId {
        self.page_cache_file_id
    }

    pub(crate) async fn load_to_io_buf(
        &self,
        ctx: &RequestContext,
    ) -> Result<IoBufferMut, io::Error> {
        let size = self.len().into_usize();
        let buf = IoBufferMut::with_capacity(size);
        let (slice, nread) = self.read_exact_at_eof_ok(0, buf.slice_full(), ctx).await?;
        assert_eq!(nread, size);
        let buf = slice.into_inner();
        assert_eq!(buf.len(), nread);
        assert_eq!(buf.capacity(), size, "we shouldn't be reallocating");
        Ok(buf)
    }

    pub(crate) async fn write_raw(
        &self,
        srcbuf: &[u8],
        ctx: &RequestContext,
    ) -> Result<u64, EphemeralFileWriteError> {
        // serialize writers. this mutex protects the append order and frozen_buffer.
        // readers do NOT touch this lock so they dont stall on slow disk io.
        let _write_guard = self.writer_lock.lock().await;

        let mut src_remaining = srcbuf;
        let mut first_write_pos = None;

        while !src_remaining.is_empty() {
            // 1. Copy to memory (fast)
            let mut guard = self.mutable_state.write().await;

            if first_write_pos.is_none() {
                first_write_pos = Some(guard.start_offset + guard.buffer.len() as u64);
            }

            let capacity = guard.buffer.capacity();
            let current_len = guard.buffer.len();
            let available = capacity - current_len;

            if available > 0 {
                let to_copy = std::cmp::min(src_remaining.len(), available);
                guard.buffer.extend_from_slice(&src_remaining[..to_copy]);
                src_remaining = &src_remaining[to_copy..];

                // bump logical size so readers see it
                self.bytes_written
                    .fetch_add(to_copy as u64, Ordering::Relaxed);
            }

            // 2. Flush if full (slow)
            if guard.buffer.len() == guard.buffer.capacity() {
                let buf_to_flush =
                    std::mem::replace(&mut guard.buffer, IoBufferMut::with_capacity(TAIL_SZ));
                let flush_offset = guard.start_offset;

                guard.start_offset += buf_to_flush.len() as u64;

                // publish to readers immediately
                let frozen_state = FrozenState {
                    offset: flush_offset,
                    buffer: buf_to_flush,
                };
                self.frozen_buffer.store(Arc::new(Some(frozen_state)));

                // drop the rwlock so readers can proceed while we flush
                drop(guard);

                // NOTE: we still hold _write_guard here, so nobody can overwrite frozen_buffer

                // reload buffer ref for IO
                let frozen_arc = self.frozen_buffer.load();
                let state = frozen_arc.as_ref().as_ref().expect(
                    "Buffer disappeared during write, this shd be impossible with writer_lock",
                );
                let data_ref = &state.buffer;

                // copy to scratch buffer for aligned IO
                let mut scratch_buf = IoBufferMut::with_capacity(data_ref.len());
                scratch_buf.extend_from_slice(data_ref);
                let frozen_scratch = scratch_buf.freeze();

                let (_, res) = self
                    .file
                    .write_all_at(
                        FullSlice::must_new(frozen_scratch.slice_full()),
                        flush_offset,
                        ctx,
                    )
                    .await;

                res.map_err(|_| EphemeralFileWriteError::Cancelled)?;

                // update physical offset so readers can use the file
                self.disk_committed_offset
                    .fetch_add(data_ref.len() as u64, Ordering::Release);

                self.frozen_buffer.store(Arc::new(None));
            }
        }

        let mut resource_units = self.resource_units.lock().unwrap();
        resource_units.maybe_publish_size(self.bytes_written.load(Ordering::Relaxed));

        Ok(first_write_pos.unwrap_or(0))
    }

    pub(crate) fn tick(&self) -> Option<u64> {
        let mut resource_units = self.resource_units.lock().unwrap();
        let len = self.bytes_written.load(Ordering::Relaxed);
        resource_units.publish_size(len)
    }
}

impl super::storage_layer::inmemory_layer::vectored_dio_read::File for EphemeralFile {
    async fn read_exact_at_eof_ok<B: IoBufAlignedMut + Send>(
        &self,
        start: u64,
        mut dst: tokio_epoll_uring::Slice<B>,
        ctx: &RequestContext,
    ) -> std::io::Result<(tokio_epoll_uring::Slice<B>, usize)> {
        dst.as_mut_rust_slice_full_zeroed();

        // 1. Snapshot frozen state
        let frozen_guard = self.frozen_buffer.load();
        let maybe_flushed = frozen_guard.as_ref();

        // 2. Lock mutable tail
        let guard = self.mutable_state.read().await;
        let mutable = &guard.buffer;
        let mutable_start = guard.start_offset;

        // 3. Check physical disk limits
        let disk_boundary = self.disk_committed_offset.load(Ordering::Acquire);

        let dst_cap = dst.bytes_total().into_u64();
        let end = start.saturating_add(dst_cap);

        let mutable_end = mutable_start + mutable.len() as u64;
        let dst_slice = dst.as_mut_rust_slice_full_zeroed();

        // Read priority: Mutable -> Frozen -> Disk

        if start < mutable_end && end > mutable_start {
            let overlap_start = std::cmp::max(start, mutable_start);
            let overlap_end = std::cmp::min(end, mutable_end);
            if overlap_end > overlap_start {
                let src_off = (overlap_start - mutable_start) as usize;
                let dst_off = (overlap_start - start) as usize;
                let len = (overlap_end - overlap_start) as usize;
                dst_slice[dst_off..dst_off + len].copy_from_slice(&mutable[src_off..src_off + len]);
            }
        }

        if let Some(state) = maybe_flushed {
            let frozen_start = state.offset;
            let frozen_len = state.buffer.len() as u64;
            let frozen_end = frozen_start + frozen_len;

            if start < frozen_end && end > frozen_start {
                let overlap_start = std::cmp::max(start, frozen_start);
                let overlap_end = std::cmp::min(end, frozen_end);
                if overlap_end > overlap_start {
                    let src_off = (overlap_start - frozen_start) as usize;
                    let dst_off = (overlap_start - start) as usize;
                    let len = (overlap_end - overlap_start) as usize;
                    dst_slice[dst_off..dst_off + len]
                        .copy_from_slice(&state.buffer[src_off..src_off + len]);
                }
            }
        }

        // drop locks before potentially slow IO
        drop(guard);
        drop(frozen_guard);

        if start < disk_boundary {
            let read_end = std::cmp::min(end, disk_boundary);
            let len = (read_end - start) as usize;
            if len > 0 {
                let bounds = dst.bounds();
                let sub_slice = dst.slice(0..len);
                let read_slice = self.file.read_exact_at(sub_slice, start, ctx).await?;
                dst = Slice::from_buf_bounds(Slice::into_inner(read_slice), bounds);
            }
        }

        Ok((dst, (end - start).into_usize()))
    }
}

pub fn is_ephemeral_file(filename: &str) -> bool {
    if let Some(rest) = filename.strip_prefix("ephemeral-") {
        rest.parse::<u32>().is_ok()
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DownloadBehavior;
    use crate::task_mgr::TaskKind;
    use rand::Rng;
    use std::fs;
    use std::str::FromStr;
    use std::time::{Duration, Instant};

    fn harness(
        test_name: &str,
    ) -> (
        &'static PageServerConf,
        TenantShardId,
        TimelineId,
        RequestContext,
    ) {
        let repo_dir = PageServerConf::test_repo_dir(test_name);
        let _ = fs::remove_dir_all(&repo_dir);
        let conf = PageServerConf::dummy_conf(repo_dir);
        let conf: &'static PageServerConf = Box::leak(Box::new(conf));

        let tenant_shard_id = TenantShardId::from_str("11000000000000000000000000000000").unwrap();
        let timeline_id = TimelineId::from_str("22000000000000000000000000000000").unwrap();
        fs::create_dir_all(conf.timeline_path(&tenant_shard_id, &timeline_id)).unwrap();

        let ctx =
            RequestContext::new(TaskKind::UnitTest, DownloadBehavior::Error).with_scope_unit_test();

        (conf, tenant_shard_id, timeline_id, ctx)
    }

    #[tokio::test]
    async fn ephemeral_file_holds_gate_open() {
        const FOREVER: std::time::Duration = std::time::Duration::from_secs(5);

        let (conf, tenant_id, timeline_id, ctx) = harness("ephemeral_file_holds_gate_open");

        let gate = utils::sync::gate::Gate::default();
        let cancel = CancellationToken::new();

        let file = EphemeralFile::create(conf, tenant_id, timeline_id, &gate, &cancel, &ctx)
            .await
            .unwrap();

        let mut closing = tokio::task::spawn(async move {
            gate.close().await;
        });

        tokio::time::pause();
        tokio::time::timeout(FOREVER, &mut closing)
            .await
            .expect_err("closing cannot complete before dropping");

        drop(file);

        tokio::time::timeout(FOREVER, &mut closing)
            .await
            .expect("closing completes right away")
            .expect("closing does not panic");
    }

    #[tokio::test]
    async fn test_ephemeral_file_basics() {
        let (conf, tenant_id, timeline_id, ctx) = harness("test_ephemeral_file_basics");

        let gate = utils::sync::gate::Gate::default();
        let cancel = CancellationToken::new();

        let file = EphemeralFile::create(conf, tenant_id, timeline_id, &gate, &cancel, &ctx)
            .await
            .unwrap();

        let data = b"hello world";
        let off = file.write_raw(data, &ctx).await.unwrap();
        assert_eq!(off, 0);

        let buf = IoBufferMut::with_capacity(data.len());
        let (slice, _) = file
            .read_exact_at_eof_ok(0, buf.slice_full(), &ctx)
            .await
            .unwrap();
        assert_eq!(&slice.into_inner()[..], data);
    }

    #[tokio::test]
    async fn test_flushes_do_happen() {
        let (conf, tenant_id, timeline_id, ctx) = harness("test_flushes_do_happen");

        let gate = utils::sync::gate::Gate::default();
        let cancel = CancellationToken::new();
        let file = EphemeralFile::create(conf, tenant_id, timeline_id, &gate, &cancel, &ctx)
            .await
            .unwrap();

        let cap = TAIL_SZ;
        let write_nbytes = cap * 2 + cap / 2;

        let content: Vec<u8> = rand::rng()
            .sample_iter(rand::distr::StandardUniform)
            .take(write_nbytes)
            .collect();

        let _ = file.write_raw(&content, &ctx).await.unwrap();

        // Verify data integrity
        let load_io_buf_res = file.load_to_io_buf(&ctx).await.unwrap();
        assert_eq!(&load_io_buf_res[..], &content[..]);

        // Verify disk size (should be 2 full chunks)
        let md = file.file.path().metadata().unwrap();
        assert_eq!(md.len(), (2 * cap) as u64);
    }

    #[tokio::test]
    async fn test_read_split_across_boundaries() {
        let (conf, tenant_id, timeline_id, ctx) = harness("test_read_split_across_boundaries");
        let gate = utils::sync::gate::Gate::default();
        let cancel = CancellationToken::new();

        let file = EphemeralFile::create(conf, tenant_id, timeline_id, &gate, &cancel, &ctx)
            .await
            .unwrap();

        let chunk_size = 64 * 1024;

        let chunk1 = vec![b'A'; chunk_size];
        let chunk2 = vec![b'B'; chunk_size];
        let chunk3 = vec![b'C'; 100];

        file.write_raw(&chunk1, &ctx).await.unwrap();
        file.write_raw(&chunk2, &ctx).await.unwrap();
        file.write_raw(&chunk3, &ctx).await.unwrap();

        // Check boundary (64KB - 10)
        let start = (chunk_size - 10) as u64;
        let len = 20;
        let buf = IoBufferMut::with_capacity(len);
        let (slice, _) = file
            .read_exact_at_eof_ok(start, buf.slice_full(), &ctx)
            .await
            .unwrap();
        let bytes = slice.into_inner();

        assert_eq!(&bytes[0..10], &vec![b'A'; 10][..]);
        assert_eq!(&bytes[10..20], &vec![b'B'; 10][..]);
    }

    #[tokio::test]
    async fn test_multi_writer_race_condition() {
        // Ensures that `writer_lock` is working.
        // Without it, interleaved writes clobber frozen_buffer.

        let (conf, tenant_id, timeline_id, _ctx) = harness("test_multi_writer_race_condition");
        let gate = utils::sync::gate::Gate::default();
        let cancel = CancellationToken::new();

        let ctx =
            RequestContext::new(TaskKind::UnitTest, DownloadBehavior::Error).with_scope_unit_test();

        let file = Arc::new(
            EphemeralFile::create(conf, tenant_id, timeline_id, &gate, &cancel, &ctx)
                .await
                .unwrap(),
        );

        let num_writers = 2;
        let chunk_size = 1024;
        let mut handles = vec![];

        for w_id in 0..num_writers {
            let f = file.clone();
            handles.push(tokio::spawn(async move {
                let c = RequestContext::new(TaskKind::UnitTest, DownloadBehavior::Error)
                    .with_scope_unit_test();
                let chunk = vec![w_id as u8; chunk_size];
                for _ in 0..100 {
                    f.write_raw(&chunk, &c).await.unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // Check length consistency
        let expected_len = (num_writers * 100 * chunk_size) as u64;
        assert_eq!(
            file.len(),
            expected_len,
            "File length mismatch! Data lost to writer race"
        );
    }

    #[tokio::test]
    async fn test_concurrent_flush_no_stall() {
        let (conf, tenant_id, timeline_id, ctx) = harness("test_concurrent_flush_no_stall");
        let gate = utils::sync::gate::Gate::default();
        let cancel = CancellationToken::new();

        let file = Arc::new(
            EphemeralFile::create(conf, tenant_id, timeline_id, &gate, &cancel, &ctx)
                .await
                .unwrap(),
        );

        let writer_file = file.clone();
        let writer_handle = tokio::spawn(async move {
            let writer_ctx = RequestContext::new(TaskKind::UnitTest, DownloadBehavior::Error)
                .with_scope_unit_test();
            let chunk = vec![0u8; TAIL_SZ];
            for _ in 0..50 {
                writer_file.write_raw(&chunk, &writer_ctx).await.unwrap();
                tokio::task::yield_now().await;
            }
        });

        let mut reader_handles = vec![];
        for i in 0..5 {
            let reader_file = file.clone();
            reader_handles.push(tokio::spawn(async move {
                let reader_ctx = RequestContext::new(TaskKind::UnitTest, DownloadBehavior::Error)
                    .with_scope_unit_test();
                let start = Instant::now();
                let mut reads = 0;
                while start.elapsed() < Duration::from_secs(1) {
                    let buf = IoBufferMut::with_capacity(1024);
                    if reader_file
                        .read_exact_at_eof_ok(0, buf.slice_full(), &reader_ctx)
                        .await
                        .is_ok()
                    {
                        reads += 1;
                    }
                }
                println!("Reader {} completed {} reads", i, reads);
                reads
            }));
        }

        writer_handle.await.unwrap();
        let mut total_reads = 0;
        for h in reader_handles {
            total_reads += h.await.unwrap();
        }

        assert!(total_reads > 100, "Readers were blocked by the flush lock!");
    }
}
