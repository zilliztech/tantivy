use std::str::FromStr;
use std::{env, thread};

use lazy_static::lazy_static;
use rayon::{ThreadPool, ThreadPoolBuilder};

const MILVUS_TOKIO_MERGER_THREAD_NUM: &str = "MILVUS_TANTIVY_MERGER_THREAD_NUM";
const MILVUS_TANTIVY_WRITER_THREAD_NUM: &str = "MILVUS_TANTIVY_WRITER_THREAD_NUM";
const MILVUS_TOKIO_THREAD_NUM: &str = "MILVUS_TOKIO_THREAD_NUM";
const MILVUS_TOKIO_DOCSTORE_COMPRESS_THREAD_NUM: &str =
    "MILVUS_TANTIVY_DOCSTORE_COMPRESS_THREAD_NUM";

lazy_static! {
    pub static ref TOKIO_RUNTIME: tokio::runtime::Runtime =
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(get_num_thread(MILVUS_TOKIO_THREAD_NUM))
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");
    pub static ref WRITER_THREAD_POOL: ThreadPool = ThreadPoolBuilder::new()
        .num_threads(get_num_thread(MILVUS_TANTIVY_WRITER_THREAD_NUM))
        .thread_name(|sz| format!("tantivy-writer{}", sz))
        .build()
        .expect("Failed to create tantivy-writer thread pool");
    pub static ref TOKIO_MERGER_THREAD_POOL: tokio::runtime::Runtime =
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(get_num_thread(MILVUS_TOKIO_MERGER_THREAD_NUM))
            .thread_name("tantivy-merger")
            .build()
            .expect("Failed to create tantivy-writer thread pool");
    pub static ref TOKIO_DOCSTORE_COMPRESS_RUNTIME: tokio::runtime::Runtime =
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(get_num_thread(MILVUS_TOKIO_DOCSTORE_COMPRESS_THREAD_NUM))
            .thread_name("tantivy-docstore-compress")
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");
}

fn default_num_thread() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn get_num_thread(thread_num_env_key: &str) -> usize {
    // Use the environment variable to change the thread num for high priority.
    if let Some(x @ 1..) = env::var(thread_num_env_key)
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        return x;
    }

    default_num_thread()
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::{
        default_num_thread, get_num_thread, MILVUS_TANTIVY_WRITER_THREAD_NUM,
        MILVUS_TOKIO_MERGER_THREAD_NUM,
    };

    #[test]
    fn test_get_num_thread() {
        let default_num = default_num_thread();
        let test_one = |env_var: &str| {
            let thread_num = get_num_thread(env_var);
            assert_eq!(thread_num, default_num);
            env::set_var(env_var, "2");
            let thread_num = get_num_thread(env_var);
            assert_eq!(thread_num, 2);
            env::set_var(env_var, "16");
            let thread_num = get_num_thread(env_var);
            assert_eq!(thread_num, 16);
            env::set_var(env_var, "0");
            let thread_num = get_num_thread(env_var);
            assert_eq!(thread_num, default_num);
            env::set_var(env_var, "a");
            let thread_num = get_num_thread(env_var);
            assert_eq!(thread_num, default_num);
        };
        test_one(MILVUS_TOKIO_MERGER_THREAD_NUM);
        test_one(MILVUS_TANTIVY_WRITER_THREAD_NUM);
    }
}
