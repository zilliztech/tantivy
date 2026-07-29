use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use smallvec::smallvec;
use tokio::task::JoinHandle;

use super::index_writer::{error_in_index_worker_thread, SingletonIndexWriterOptions};
use super::pool::{get_tokio_indexing_worker_pool, init_pool};
use super::AddBatch;
use crate::indexer::merger::MAX_DOC_LIMIT;
use crate::indexer::operation::AddOperation;
use crate::indexer::segment_updater::save_metas;
use crate::indexer::SegmentWriter;
use crate::schema::document::Document;
use crate::{Directory, Index, IndexMeta, Opstamp, Segment, TantivyDocument, TantivyError};

struct SingleSegmentWriterState {
    mem_usage: AtomicUsize,
    worker_error: Mutex<Option<TantivyError>>,
    worker_alive: AtomicBool,
}

impl SingleSegmentWriterState {
    fn new() -> Self {
        Self {
            mem_usage: AtomicUsize::new(0),
            worker_error: Mutex::new(None),
            worker_alive: AtomicBool::new(true),
        }
    }

    fn record_worker_error(&self, error: &TantivyError) -> crate::Result<()> {
        let mut worker_error = self.worker_error.lock()?;
        if worker_error.is_none() {
            *worker_error = Some(error.clone());
        }
        Ok(())
    }

    fn worker_error(&self) -> crate::Result<Option<TantivyError>> {
        Ok(self.worker_error.lock()?.clone())
    }
}

struct WorkerAliveGuard {
    state: Arc<SingleSegmentWriterState>,
}

impl Drop for WorkerAliveGuard {
    fn drop(&mut self) {
        self.state.worker_alive.store(false, Ordering::Release);
    }
}

#[doc(hidden)]
pub struct SingleSegmentIndexWriter<D: Document = TantivyDocument> {
    segment: Segment,
    tx: Arc<async_channel::Sender<AddBatch<D>>>,
    join_handle: JoinHandle<crate::Result<SegmentWriter>>,
    state: Arc<SingleSegmentWriterState>,
    next_opstamp: Opstamp,
    last_doc_id: Option<u32>,
    _phantom: PhantomData<D>,
}

impl<D: Document> SingleSegmentIndexWriter<D> {
    pub fn new(index: Index, mem_budget: usize) -> crate::Result<Self> {
        let config = SingletonIndexWriterOptions::default();
        init_pool(config);

        let segment = index.new_segment();
        let mut segment_writer = SegmentWriter::for_segment(mem_budget, segment.clone())?;
        let (tx, rx) = async_channel::unbounded();
        let state = Arc::new(SingleSegmentWriterState::new());
        let worker_state = Arc::clone(&state);
        let join_handle = get_tokio_indexing_worker_pool().spawn(async move {
            let _worker_alive_guard = WorkerAliveGuard {
                state: Arc::clone(&worker_state),
            };
            while let Ok(add_operations) = rx.recv().await {
                for add_operation in add_operations {
                    if let Err(error) = segment_writer.add_document(add_operation).await {
                        let _ = worker_state.record_worker_error(&error);
                        return Err(error);
                    }
                }
                worker_state
                    .mem_usage
                    .store(segment_writer.mem_usage(), Ordering::Release);
            }
            Ok(segment_writer)
        });
        Ok(Self {
            segment,
            tx: Arc::new(tx),
            join_handle,
            state,
            next_opstamp: 0,
            last_doc_id: None,
            _phantom: PhantomData,
        })
    }

    pub fn add_document(&mut self, document: D) -> crate::Result<()> {
        let add_operation = AddOperation {
            opstamp: self.next_opstamp,
            document,
            doc_id: None,
        };
        self.send_batch(smallvec![add_operation])?;
        self.next_opstamp += 1;
        Ok(())
    }

    /// Adds a batch of documents with user-specified document IDs.
    ///
    /// Document IDs must be strictly increasing across and within batches. Sparse document IDs are
    /// allowed.
    pub fn add_documents_with_doc_ids<I>(&mut self, documents: I) -> crate::Result<()>
    where
        I: IntoIterator<Item = (u32, D)>,
        I::IntoIter: ExactSizeIterator,
    {
        let documents = documents.into_iter();
        if documents.len() == 0 {
            return Ok(());
        }
        if !self.segment.index().schema().user_specified_doc_id() {
            return Err(TantivyError::InvalidArgument(
                "User specified id is not enabled".to_string(),
            ));
        }

        let mut add_operations = AddBatch::with_capacity(documents.len());
        let mut previous_doc_id = self.last_doc_id;
        let mut next_opstamp = self.next_opstamp;
        for (doc_id, document) in documents {
            match doc_id.checked_add(1) {
                Some(max_doc) if max_doc < MAX_DOC_LIMIT => {}
                _ => {
                    return Err(TantivyError::InvalidArgument(format!(
                        "Document ID {doc_id} is out of range: resulting max doc must be less \
                         than {MAX_DOC_LIMIT}"
                    )));
                }
            }
            if let Some(previous_doc_id) = previous_doc_id {
                if doc_id <= previous_doc_id {
                    return Err(TantivyError::InvalidArgument(format!(
                        "Document ID must be strictly ordered: previous doc id {}, current doc id \
                         {}",
                        previous_doc_id, doc_id,
                    )));
                }
            }
            add_operations.push(AddOperation {
                opstamp: next_opstamp,
                document,
                doc_id: Some(doc_id),
            });
            previous_doc_id = Some(doc_id);
            next_opstamp += 1;
        }

        self.send_batch(add_operations)?;
        self.last_doc_id = previous_doc_id;
        self.next_opstamp = next_opstamp;
        Ok(())
    }

    fn send_batch(&self, add_operations: AddBatch<D>) -> crate::Result<()> {
        if !self.state.worker_alive.load(Ordering::Acquire) {
            return Err(self
                .state
                .worker_error()?
                .unwrap_or_else(|| error_in_index_worker_thread("An index writer was closed")));
        }

        let tx = self.tx.clone();
        if get_tokio_indexing_worker_pool()
            .block_on(async move { tx.send(add_operations).await })
            .is_ok()
        {
            return Ok(());
        }
        Err(self
            .state
            .worker_error()?
            .unwrap_or_else(|| error_in_index_worker_thread("An index writer was closed")))
    }

    /// Returns the latest memory usage reported by the indexing worker.
    pub fn mem_usage(&self) -> usize {
        self.state.mem_usage.load(Ordering::Acquire)
    }

    pub fn finalize(self) -> crate::Result<Index> {
        get_tokio_indexing_worker_pool().block_on(async {
            self.tx.close();
            let segment_writer = self
                .join_handle
                .await
                .map_err(|_| error_in_index_worker_thread("Worker thread panicked."))??;

            let max_doc = segment_writer.max_doc();
            segment_writer.finalize().await?;
            let segment: Segment = self.segment.with_max_doc(max_doc);
            let index = segment.index();
            let index_meta = IndexMeta {
                index_settings: index.settings().clone(),
                segments: vec![segment.meta().clone()],
                schema: index.schema(),
                opstamp: 0,
                payload: None,
            };
            save_metas(&index_meta, index.directory())?;
            index.directory().sync_directory()?;
            Ok(segment.index().clone())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::{Duration, Instant};

    use super::MAX_DOC_LIMIT;
    use crate::collector::DocSetCollector;
    use crate::directory::RamDirectory;
    use crate::query::TermQuery;
    use crate::schema::{IndexRecordOption, Schema, Term, INDEXED, TEXT, TEXT_WITH_DOC_ID};
    use crate::{doc, Index, TantivyDocument, TantivyError};

    const MEMORY_BUDGET: usize = 15_000_000;

    fn poll_until<T>(
        description: &str,
        mut observe: impl FnMut() -> Option<T>,
    ) -> crate::Result<T> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(value) = observe() {
                return Ok(value);
            }
            if Instant::now() >= deadline {
                return Err(TantivyError::SystemError(format!(
                    "Timed out waiting for {description}"
                )));
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn schema_error_writer() -> crate::Result<(
        crate::schema::Field,
        super::SingleSegmentIndexWriter<TantivyDocument>,
    )> {
        let mut schema_builder = Schema::builder();
        let number = schema_builder.add_u64_field("number", INDEXED);
        let writer = Index::builder()
            .schema(schema_builder.build())
            .single_segment_index_writer(RamDirectory::default(), MEMORY_BUDGET)?;
        Ok((number, writer))
    }

    fn user_doc_id_writer() -> crate::Result<(
        crate::schema::Field,
        RamDirectory,
        super::SingleSegmentIndexWriter<TantivyDocument>,
    )> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT_WITH_DOC_ID);
        schema_builder.enable_user_specified_doc_id();
        let directory = RamDirectory::default();
        let writer = Index::builder()
            .schema(schema_builder.build())
            .single_segment_index_writer(directory.clone(), MEMORY_BUDGET)?;
        Ok((text, directory, writer))
    }

    #[test]
    fn test_add_documents_with_sparse_doc_ids() -> crate::Result<()> {
        let (text, directory, mut writer) = user_doc_id_writer()?;

        writer.add_documents_with_doc_ids(Vec::<(u32, TantivyDocument)>::new())?;
        writer.add_documents_with_doc_ids(vec![
            (0, doc!(text => "shared")),
            (2, doc!(text => "shared")),
            (5, doc!(text => "shared")),
        ])?;
        writer.finalize()?;

        let index = Index::open(directory)?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        assert_eq!(searcher.segment_reader(0).max_doc(), 6);

        let term_query = TermQuery::new(
            Term::from_field_text(text, "shared"),
            IndexRecordOption::Basic,
        );
        let mut doc_ids: Vec<u32> = searcher
            .search(&term_query, &DocSetCollector)?
            .into_iter()
            .map(|doc_address| doc_address.doc_id)
            .collect();
        doc_ids.sort_unstable();
        assert_eq!(doc_ids, [0, 2, 5]);
        Ok(())
    }

    #[test]
    fn test_add_documents_with_duplicate_doc_ids_is_invalid() -> crate::Result<()> {
        let (text, _directory, mut writer) = user_doc_id_writer()?;

        let error = writer
            .add_documents_with_doc_ids(vec![
                (2, doc!(text => "first")),
                (2, doc!(text => "duplicate")),
            ])
            .unwrap_err();
        assert!(matches!(error, TantivyError::InvalidArgument(_)));
        Ok(())
    }

    #[test]
    fn test_add_documents_with_descending_doc_ids_is_invalid() -> crate::Result<()> {
        let (text, _directory, mut writer) = user_doc_id_writer()?;

        let error = writer
            .add_documents_with_doc_ids(vec![
                (5, doc!(text => "first")),
                (2, doc!(text => "descending")),
            ])
            .unwrap_err();
        assert!(matches!(error, TantivyError::InvalidArgument(_)));
        Ok(())
    }

    #[test]
    fn test_add_documents_with_doc_ids_must_increase_across_batches() -> crate::Result<()> {
        let (text, _directory, mut writer) = user_doc_id_writer()?;

        writer.add_documents_with_doc_ids(vec![(5, doc!(text => "first"))])?;

        for doc_id in [5, 2] {
            let error = writer
                .add_documents_with_doc_ids(vec![(doc_id, doc!(text => "invalid"))])
                .unwrap_err();
            assert!(matches!(error, TantivyError::InvalidArgument(_)));
        }
        Ok(())
    }

    #[test]
    fn test_add_documents_with_doc_ids_requires_user_doc_id_schema() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let directory = RamDirectory::default();
        let mut writer = Index::builder()
            .schema(schema_builder.build())
            .single_segment_index_writer(directory, MEMORY_BUDGET)?;

        let error = writer
            .add_documents_with_doc_ids(vec![(0, doc!(text => "invalid"))])
            .unwrap_err();
        assert!(matches!(error, TantivyError::InvalidArgument(_)));
        Ok(())
    }

    #[test]
    fn test_add_documents_with_doc_ids_accepts_max_doc_boundary() -> crate::Result<()> {
        let (_text, _directory, mut writer) = user_doc_id_writer()?;

        writer.add_documents_with_doc_ids(vec![(MAX_DOC_LIMIT - 2, TantivyDocument::default())])?;
        let index = writer.finalize()?;

        let segment_metas = index.searchable_segment_metas()?;
        assert_eq!(segment_metas.len(), 1);
        assert_eq!(segment_metas[0].max_doc(), MAX_DOC_LIMIT - 1);
        Ok(())
    }

    #[test]
    fn test_add_documents_with_doc_ids_rejects_max_doc_limit() -> crate::Result<()> {
        let (_text, _directory, mut writer) = user_doc_id_writer()?;

        let error = writer
            .add_documents_with_doc_ids(vec![
                (0, TantivyDocument::default()),
                (MAX_DOC_LIMIT - 1, TantivyDocument::default()),
            ])
            .unwrap_err();
        assert!(matches!(error, TantivyError::InvalidArgument(_)));
        writer.add_documents_with_doc_ids(vec![(0, TantivyDocument::default())])?;
        Ok(())
    }

    #[test]
    fn test_add_documents_with_doc_ids_rejects_u32_max() -> crate::Result<()> {
        let (_text, _directory, mut writer) = user_doc_id_writer()?;

        let error = writer
            .add_documents_with_doc_ids(vec![(u32::MAX, TantivyDocument::default())])
            .unwrap_err();
        assert!(matches!(error, TantivyError::InvalidArgument(_)));
        Ok(())
    }

    #[test]
    fn test_add_document_after_worker_failure_returns_worker_error() -> crate::Result<()> {
        let (number, mut writer) = schema_error_writer()?;
        writer.add_document(doc!(number => "not a u64"))?;

        let error = poll_until("the worker error", || {
            writer.add_document(TantivyDocument::default()).err()
        })?;

        assert!(
            matches!(error, TantivyError::SchemaError(_)),
            "unexpected worker error: {error:?}"
        );
        Ok(())
    }

    #[test]
    fn test_finalize_preserves_worker_error() -> crate::Result<()> {
        let (number, mut writer) = schema_error_writer()?;
        writer.add_document(doc!(number => "not a u64"))?;

        let error = match writer.finalize() {
            Ok(_) => {
                return Err(TantivyError::SystemError(
                    "finalize unexpectedly succeeded".to_string(),
                ));
            }
            Err(error) => error,
        };

        assert!(
            matches!(error, TantivyError::SchemaError(_)),
            "unexpected worker error: {error:?}"
        );
        Ok(())
    }

    #[test]
    fn test_mem_usage_is_updated_after_batch_is_processed() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let mut writer = Index::builder()
            .schema(schema_builder.build())
            .single_segment_index_writer(RamDirectory::default(), MEMORY_BUDGET)?;

        writer.add_document(doc!(text => "some text to consume indexing memory"))?;
        let mem_usage = poll_until("non-zero writer memory usage", || {
            let mem_usage = writer.mem_usage();
            (mem_usage > 0).then_some(mem_usage)
        })?;

        assert!(mem_usage > 0);
        writer.finalize()?;
        Ok(())
    }
}
