/// Memory profile for IndexWriter configuration.
///
/// Different profiles optimize for different use cases:
/// - `Default`: Standard configuration for high-throughput indexing
/// - `LowMemory`: Optimized for scenarios with many small indexes (e.g., growing segments)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryProfile {
    /// Standard configuration optimized for high-throughput indexing.
    /// - Pipeline size: 10,000 documents
    /// - Arena page size: 1MB
    /// - Hash table max capacity: 2^19 (4MB)
    /// - Min memory budget: 15MB
    #[default]
    Default,

    /// Low memory configuration for scenarios with many small indexes.
    /// - Pipeline size: 1,000 documents
    /// - Arena page size: 256KB
    /// - Hash table max capacity: 2^17 (1MB)
    /// - Min memory budget: 3MB
    LowMemory,
}

impl MemoryProfile {
    /// Returns the maximum number of documents in the indexing pipeline.
    pub fn pipeline_max_size(&self) -> usize {
        match self {
            Self::Default => 10_000,
            Self::LowMemory => 1_000,
        }
    }

    /// Returns the arena page size in bytes.
    pub fn arena_page_size(&self) -> usize {
        match self {
            Self::Default => 1 << 20,  // 1MB
            Self::LowMemory => 1 << 18, // 256KB
        }
    }

    /// Returns the maximum power for hash table capacity (2^n).
    pub fn hash_table_max_power(&self) -> usize {
        match self {
            Self::Default => 20,  // 2^19 max = 4MB
            Self::LowMemory => 18, // 2^17 max = 1MB
        }
    }

    /// Returns the minimum memory budget in bytes.
    pub fn min_memory_budget(&self) -> usize {
        match self {
            Self::Default => 15 * 1024 * 1024,  // 15MB
            Self::LowMemory => 3 * 1024 * 1024,  // 3MB
        }
    }
}
