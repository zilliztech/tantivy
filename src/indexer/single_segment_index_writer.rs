use std::marker::PhantomData;
use std::sync::Arc;

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

#[doc(hidden)]
pub struct SingleSegmentIndexWriter<D: Document = TantivyDocument> {
    segment: Segment,
    tx: Arc<async_channel::Sender<AddBatch<D>>>,
    join_handle: JoinHandle<crate::Result<SegmentWriter>>,
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
        let join_handle = get_tokio_indexing_worker_pool().spawn(async move {
            while let Ok(add_operations) = rx.recv().await {
                for add_operation in add_operations {
                    segment_writer.add_document(add_operation).await?;
                }
            }
            Ok(segment_writer)
        });
        Ok(Self {
            segment,
            tx: Arc::new(tx),
            join_handle,
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
        let tx = self.tx.clone();
        if get_tokio_indexing_worker_pool()
            .block_on(async move { tx.send(add_operations).await })
            .is_ok()
        {
            return Ok(());
        }
        Err(error_in_index_worker_thread(
            "An index writer encounter erros.",
        ))
    }

    pub fn finalize(self) -> crate::Result<Index> {
        get_tokio_indexing_worker_pool().block_on(async {
            self.tx.close();
            let segment_writer = self
                .join_handle
                .await
                .map_err(|_| error_in_index_worker_thread("Worker thread panicked."))?
                .map_err(|_| error_in_index_worker_thread("Worker thread failed."))?;

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
    use super::MAX_DOC_LIMIT;
    use crate::collector::DocSetCollector;
    use crate::directory::RamDirectory;
    use crate::query::TermQuery;
    use crate::schema::{IndexRecordOption, Schema, Term, TEXT, TEXT_WITH_DOC_ID};
    use crate::{doc, Index, TantivyDocument, TantivyError};

    const MEMORY_BUDGET: usize = 15_000_000;

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
}
