use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::thread;

use rayon::{ThreadPool, ThreadPoolBuilder};
use tokio::runtime;

use super::index_writer::SingletonIndexWriterOptions;

pub static TOKIO_INDEXING_WORKER_POOL: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
pub static SEGMENT_UPDATER_POOL: OnceLock<ThreadPool> = OnceLock::new();
pub static TOKIO_MERGER_WORKER_POOL: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
pub static TOKIO_DOCSTORE_COMPRESS_WORKER_POOL: OnceLock<runtime::Runtime> = OnceLock::new();
pub static TOKIO_FILE_WATCHER_WORKER_POOL: OnceLock<runtime::Runtime> = OnceLock::new();
pub static MISC_POLL: OnceLock<ThreadPool> = OnceLock::new();

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

    let cpu_cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);

    let _ = TOKIO_FILE_WATCHER_WORKER_POOL.get_or_init(|| {
        runtime::Builder::new_multi_thread()
            // 2 should be enough for file watcher
            .worker_threads(usize::min(2, cpu_cores))
            .enable_all()
            .thread_name_fn(|| {
                static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
                let id = ATOMIC_ID.fetch_add(1, Ordering::Relaxed);
                format!("tantivy-file-watcher-{}", id)
            })
            .build()
            .expect("Failed to create tokio runtime")
    });

    let _ = MISC_POLL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(usize::min(2, cpu_cores))
            .thread_name(|sz| format!("tantivy-misc-{}", sz))
            .build()
            .expect("Failed to create tantivy-misc thread pool")
    });
}

#[inline]
pub fn get_tokio_indexing_worker_pool() -> &'static runtime::Runtime {
    TOKIO_INDEXING_WORKER_POOL.get().unwrap_or_else(|| {
        let config = SingletonIndexWriterOptions::default();
        init_pool(config);
        TOKIO_INDEXING_WORKER_POOL.get().unwrap()
    })
}

#[inline]
pub fn get_tokio_merger_worker_pool() -> &'static runtime::Runtime {
    TOKIO_MERGER_WORKER_POOL.get().unwrap_or_else(|| {
        let config = SingletonIndexWriterOptions::default();
        init_pool(config);
        TOKIO_MERGER_WORKER_POOL.get().unwrap()
    })
}

#[inline]
pub fn get_tokio_docstore_compress_worker_pool() -> &'static runtime::Runtime {
    TOKIO_DOCSTORE_COMPRESS_WORKER_POOL
        .get()
        .unwrap_or_else(|| {
            let config = SingletonIndexWriterOptions::default();
            init_pool(config);
            TOKIO_DOCSTORE_COMPRESS_WORKER_POOL.get().unwrap()
        })
}

#[inline]
pub fn get_segment_updater_pool() -> &'static ThreadPool {
    SEGMENT_UPDATER_POOL.get().unwrap_or_else(|| {
        let config = SingletonIndexWriterOptions::default();
        init_pool(config);
        SEGMENT_UPDATER_POOL.get().unwrap()
    })
}

#[inline]
pub fn get_tokio_file_watcher_worker_pool() -> &'static runtime::Runtime {
    TOKIO_FILE_WATCHER_WORKER_POOL.get().unwrap_or_else(|| {
        let config = SingletonIndexWriterOptions::default();
        init_pool(config);
        TOKIO_FILE_WATCHER_WORKER_POOL.get().unwrap()
    })
}

#[inline]
pub fn get_misc_poll() -> &'static ThreadPool {
    MISC_POLL.get().unwrap_or_else(|| {
        let config = SingletonIndexWriterOptions::default();
        init_pool(config);
        MISC_POLL.get().unwrap()
    })
}
