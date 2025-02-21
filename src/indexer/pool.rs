use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};
use tokio::runtime;

use super::index_writer::SingletonIndexWriterOptions;

pub static TOKIO_INDEXING_WORKER_POOL: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
pub static SEGMENT_UPDATER_POOL: OnceLock<ThreadPool> = OnceLock::new();
pub static TOKIO_MERGER_WORKER_POOL: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
pub static TOKIO_DOCSTORE_COMPRESS_WORKER_POOL: OnceLock<runtime::Runtime> = OnceLock::new();

// The initialization of the pool will be executed exactly once.
pub(crate) fn init_pool(singleton_options: SingletonIndexWriterOptions) {
    let _ = TOKIO_INDEXING_WORKER_POOL.get_or_init(|| {
        runtime::Builder::new_multi_thread()
            .worker_threads(singleton_options.singleton_tokio_indexing_worker_threads)
            .enable_all()
            .thread_name_fn(|| {
                static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
                let id = ATOMIC_ID.fetch_add(1, Ordering::Relaxed);
                format!("tantivy-indexing-worker-{}", id)
            })
            .build()
            .expect("Failed to create tokio runtime")
    });

    let _ = TOKIO_MERGER_WORKER_POOL.get_or_init(|| {
        runtime::Builder::new_multi_thread()
            .worker_threads(singleton_options.singleton_tokio_merge_worker_threads)
            .enable_all()
            .thread_name_fn(|| {
                static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
                let id = ATOMIC_ID.fetch_add(1, Ordering::Relaxed);
                format!("tantivy-merger-worker-{}", id)
            })
            .build()
            .expect("Failed to create tokio runtime")
    });

    let _ = TOKIO_DOCSTORE_COMPRESS_WORKER_POOL.get_or_init(|| {
        runtime::Builder::new_multi_thread()
            .worker_threads(singleton_options.singleton_tokio_docstore_worker_threads)
            .enable_all()
            .thread_name_fn(|| {
                static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
                let id = ATOMIC_ID.fetch_add(1, Ordering::Relaxed);
                format!("tantivy-docstore-worker-{}", id)
            })
            .build()
            .expect("Failed to create tokio runtime")
    });

    let _ = SEGMENT_UPDATER_POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(singleton_options.singleton_segment_updater_worker_threads)
            .thread_name(|sz| format!("tantivy-segment-updater-{}", sz))
            .build()
            .expect("Failed to create tantivy-writer thread pool")
    });
}

// It must be called after [`init_pool`].
#[inline]
pub fn get_tokio_indexing_worker_pool() -> &'static runtime::Runtime {
    TOKIO_INDEXING_WORKER_POOL.get().unwrap()
}

// It must be called after [`init_pool`].
#[inline]
pub fn get_tokio_merger_worker_pool() -> &'static runtime::Runtime {
    TOKIO_MERGER_WORKER_POOL.get().unwrap()
}

// It must be called after [`init_pool`].
#[inline]
pub fn get_tokio_docstore_compress_worker_pool() -> &'static runtime::Runtime {
    TOKIO_DOCSTORE_COMPRESS_WORKER_POOL.get().unwrap()
}

// It must be called after [`init_pool`].
#[inline]
pub fn get_segment_updater_pool() -> &'static ThreadPool {
    SEGMENT_UPDATER_POOL.get().unwrap()
}
