use std::any::Any;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};

use smallvec::smallvec;

use super::AddBatch;
use crate::indexer::merger::MAX_DOC_LIMIT;
use crate::indexer::operation::AddOperation;
use crate::indexer::segment_updater::save_metas;
use crate::indexer::SegmentWriter;
use crate::schema::document::Document;
use crate::{Directory, Index, IndexMeta, Opstamp, Segment, TantivyDocument, TantivyError};

fn panic_to_error(context: &str, panic_payload: Box<dyn Any + Send>) -> TantivyError {
    let panic_message = if let Some(message) = panic_payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic_payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    };
    TantivyError::ErrorInThread(format!(
        "Single segment {context} panicked: {panic_message}"
    ))
}

#[doc(hidden)]
pub struct SingleSegmentIndexWriter<D: Document = TantivyDocument> {
    segment: Segment,
    segment_writer: Option<SegmentWriter>,
    first_error: Option<TantivyError>,
    next_opstamp: Opstamp,
    last_doc_id: Option<u32>,
    _phantom: PhantomData<D>,
}

impl<D: Document> SingleSegmentIndexWriter<D> {
    pub fn new(mut index: Index, mem_budget: usize) -> crate::Result<Self> {
        index.settings_mut().docstore_compress_dedicated_thread = false;
        let segment = index.new_segment();
        let segment_writer = SegmentWriter::for_segment(mem_budget, segment.clone())?;
        Ok(Self {
            segment,
            segment_writer: Some(segment_writer),
            first_error: None,
            next_opstamp: 0,
            last_doc_id: None,
            _phantom: PhantomData,
        })
    }

    pub fn add_document(&mut self, document: D) -> crate::Result<()> {
        self.ensure_active()?;
        if self.segment.index().schema().user_specified_doc_id() {
            return Err(TantivyError::InvalidArgument(
                "add_document cannot be used when user specified document IDs are enabled"
                    .to_string(),
            ));
        }
        let next_opstamp = self.next_opstamp.checked_add(1).ok_or_else(|| {
            TantivyError::InvalidArgument("Document opstamp overflow".to_string())
        })?;
        if next_opstamp >= Opstamp::from(MAX_DOC_LIMIT) {
            return Err(TantivyError::InvalidArgument(format!(
                "Document ID {} is out of range: resulting max doc must be less than {}",
                self.next_opstamp, MAX_DOC_LIMIT
            )));
        }
        let add_operation = AddOperation {
            opstamp: self.next_opstamp,
            document,
            doc_id: None,
        };
        self.write_batch(smallvec![add_operation])?;
        self.next_opstamp = next_opstamp;
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
        self.ensure_active()?;
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
            next_opstamp = next_opstamp.checked_add(1).ok_or_else(|| {
                TantivyError::InvalidArgument("Document opstamp overflow".to_string())
            })?;
        }

        self.write_batch(add_operations)?;
        self.last_doc_id = previous_doc_id;
        self.next_opstamp = next_opstamp;
        Ok(())
    }

    fn ensure_active(&self) -> crate::Result<()> {
        if let Some(error) = &self.first_error {
            return Err(error.clone());
        }
        if self.segment_writer.is_none() {
            return Err(TantivyError::InternalError(
                "Single segment writer is unavailable".to_string(),
            ));
        }
        Ok(())
    }

    fn poison(&mut self, error: TantivyError) -> TantivyError {
        self.segment_writer.take();
        self.first_error.get_or_insert(error).clone()
    }

    fn write_batch(&mut self, add_operations: AddBatch<D>) -> crate::Result<()> {
        self.ensure_active()?;
        let mut segment_writer = self.segment_writer.take().ok_or_else(|| {
            TantivyError::InternalError("Single segment writer is unavailable".to_string())
        })?;
        let write_result = catch_unwind(AssertUnwindSafe(|| {
            futures::executor::block_on(async {
                for add_operation in add_operations {
                    segment_writer.add_document(add_operation).await?;
                }
                Ok(())
            })
        }));

        match write_result {
            Ok(Ok(())) => {
                self.segment_writer = Some(segment_writer);
                Ok(())
            }
            Ok(Err(error)) => {
                drop(segment_writer);
                let error = self.poison(error);
                Err(error)
            }
            Err(panic_payload) => {
                drop(segment_writer);
                let error = self.poison(panic_to_error("document indexing", panic_payload));
                Err(error)
            }
        }
    }

    /// Returns the active segment writer's current memory usage estimate.
    ///
    /// A poisoned writer has dropped its partial segment and reports zero memory usage.
    pub fn mem_usage(&self) -> usize {
        self.segment_writer
            .as_ref()
            .map(SegmentWriter::mem_usage)
            .unwrap_or(0)
    }

    pub fn finalize(mut self) -> crate::Result<Index> {
        if let Some(error) = self.first_error {
            return Err(error);
        }
        let segment_writer = self.segment_writer.take().ok_or_else(|| {
            TantivyError::InternalError("Single segment writer is unavailable".to_string())
        })?;
        let max_doc = segment_writer.max_doc();
        match catch_unwind(AssertUnwindSafe(|| {
            futures::executor::block_on(segment_writer.finalize())
        })) {
            Ok(result) => {
                result?;
            }
            Err(panic_payload) => {
                return Err(panic_to_error("finalization", panic_payload));
            }
        }

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
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::MAX_DOC_LIMIT;
    use crate::collector::DocSetCollector;
    use crate::directory::RamDirectory;
    use crate::query::TermQuery;
    use crate::schema::{
        Document, IndexRecordOption, NumericOptions, Schema, Term, TextFieldIndexing, TextOptions,
        INDEXED, TEXT, TEXT_WITH_DOC_ID,
    };
    use crate::tokenizer::{Token, TokenStream, Tokenizer, TokenizerManager};
    use crate::{doc, Index, IndexSettings, TantivyDocument, TantivyError};

    const MEMORY_BUDGET: usize = 15_000_000;

    struct PanicDocument;

    impl Document for PanicDocument {
        type Value<'a> = <TantivyDocument as Document>::Value<'a>;
        type FieldsValuesIter<'a> = <TantivyDocument as Document>::FieldsValuesIter<'a>;

        fn iter_fields_and_values(&self) -> Self::FieldsValuesIter<'_> {
            panic!("panic document sentinel");
        }
    }

    #[derive(Clone)]
    struct PanicTokenizer;

    struct UnusedTokenStream {
        token: Token,
    }

    impl TokenStream for UnusedTokenStream {
        fn advance(&mut self) -> bool {
            false
        }

        fn token(&self) -> &Token {
            &self.token
        }

        fn token_mut(&mut self) -> &mut Token {
            &mut self.token
        }
    }

    impl Tokenizer for PanicTokenizer {
        type TokenStream<'a> = UnusedTokenStream;

        fn token_stream<'a>(&'a mut self, _text: &'a str) -> Self::TokenStream<'a> {
            panic!("panic tokenizer sentinel");
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
        let direct_segment_id = writer.segment.id();

        writer.add_documents_with_doc_ids(Vec::<(u32, TantivyDocument)>::new())?;
        writer.add_documents_with_doc_ids(vec![
            (0, doc!(text => "shared")),
            (2, doc!(text => "shared")),
            (5, doc!(text => "shared")),
        ])?;
        writer.finalize()?;

        let index = Index::open(directory)?;
        let segment_metas = index.searchable_segment_metas()?;
        assert_eq!(segment_metas.len(), 1);
        assert_eq!(segment_metas[0].id(), direct_segment_id);
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
    fn test_add_document_rejects_max_doc_limit_atomically() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let mut writer = Index::builder()
            .schema(schema_builder.build())
            .single_segment_index_writer(RamDirectory::default(), MEMORY_BUDGET)?;
        writer.next_opstamp = u64::from(MAX_DOC_LIMIT - 1);
        let max_doc_before = writer
            .segment_writer
            .as_ref()
            .expect("writer must be active")
            .max_doc();

        let error = writer
            .add_document(doc!(text => "out of range"))
            .expect_err("resulting max doc at MAX_DOC_LIMIT must be rejected");

        assert!(matches!(error, TantivyError::InvalidArgument(_)));
        assert_eq!(writer.next_opstamp, u64::from(MAX_DOC_LIMIT - 1));
        assert_eq!(
            writer
                .segment_writer
                .as_ref()
                .expect("validation errors must not poison the writer")
                .max_doc(),
            max_doc_before
        );

        writer.next_opstamp = 0;
        writer.add_document(doc!(text => "valid"))?;
        writer.finalize()?;
        Ok(())
    }

    #[test]
    fn test_add_document_after_failure_returns_first_error() -> crate::Result<()> {
        let (number, mut writer) = schema_error_writer()?;
        let first_error = writer
            .add_document(doc!(number => "not a u64"))
            .expect_err("the invalid document must fail");

        let error = writer
            .add_document(TantivyDocument::default())
            .expect_err("the writer must stay poisoned");

        assert!(
            matches!(error, TantivyError::SchemaError(_)),
            "unexpected document error: {error:?}"
        );
        assert_eq!(error.to_string(), first_error.to_string());
        Ok(())
    }

    #[test]
    fn test_finalize_preserves_document_error() -> crate::Result<()> {
        let (number, mut writer) = schema_error_writer()?;
        let first_error = writer
            .add_document(doc!(number => "not a u64"))
            .expect_err("the invalid document must fail");

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
            "unexpected document error: {error:?}"
        );
        assert_eq!(error.to_string(), first_error.to_string());
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
        let mem_usage = writer.mem_usage();

        assert!(mem_usage > 0);
        writer.finalize()?;
        Ok(())
    }

    #[test]
    fn test_document_error_is_returned_by_the_add_call() -> crate::Result<()> {
        let (number, mut writer) = schema_error_writer()?;

        let error = writer
            .add_document(doc!(number => "not a u64"))
            .expect_err("the document error must be returned synchronously");

        assert!(
            matches!(error, TantivyError::SchemaError(_)),
            "unexpected document error: {error:?}"
        );
        Ok(())
    }

    #[test]
    fn test_document_error_poisons_writer_with_first_error() -> crate::Result<()> {
        let (number, mut writer) = schema_error_writer()?;
        let index = writer.segment.index().clone();

        let first_error = writer
            .add_document(doc!(number => "not a u64"))
            .expect_err("the first invalid document must fail");
        let second_error = writer
            .add_document(TantivyDocument::default())
            .expect_err("a poisoned writer must reject later documents");
        let empty_batch_error = writer
            .add_documents_with_doc_ids(Vec::<(u32, TantivyDocument)>::new())
            .expect_err("a poisoned writer must reject later empty batches");

        assert_eq!(second_error.to_string(), first_error.to_string());
        assert_eq!(empty_batch_error.to_string(), first_error.to_string());
        assert_eq!(writer.mem_usage(), 0);

        let finalize_error = match writer.finalize() {
            Ok(_) => {
                return Err(TantivyError::SystemError(
                    "poisoned writer unexpectedly finalized".to_string(),
                ));
            }
            Err(error) => error,
        };
        assert_eq!(finalize_error.to_string(), first_error.to_string());
        assert!(index.load_metas()?.segments.is_empty());
        Ok(())
    }

    #[test]
    fn test_batch_document_error_drops_partial_segment_without_publishing_meta() -> crate::Result<()>
    {
        let mut schema_builder = Schema::builder();
        let number =
            schema_builder.add_u64_field("number", NumericOptions::default().set_indexed());
        schema_builder.enable_user_specified_doc_id();
        let directory = RamDirectory::default();
        let mut writer = Index::builder()
            .schema(schema_builder.build())
            .single_segment_index_writer(directory.clone(), MEMORY_BUDGET)?;

        let first_error = writer
            .add_documents_with_doc_ids(vec![
                (0, doc!(number => 123u64)),
                (1, doc!(number => "invalid after first write")),
            ])
            .expect_err("the schema error must be returned by the batch call");
        assert!(matches!(first_error, TantivyError::SchemaError(_)));
        assert_eq!(writer.mem_usage(), 0);

        let finalize_error = match writer.finalize() {
            Ok(_) => {
                return Err(TantivyError::SystemError(
                    "partially written segment unexpectedly finalized".to_string(),
                ));
            }
            Err(error) => error,
        };
        assert_eq!(finalize_error.to_string(), first_error.to_string());

        let index = Index::open(directory)?;
        assert!(index.load_metas()?.segments.is_empty());
        Ok(())
    }

    #[test]
    fn test_add_document_is_invalid_for_user_doc_id_schema_without_panicking() -> crate::Result<()>
    {
        let (text, _directory, mut writer) = user_doc_id_writer()?;

        let call_result = catch_unwind(AssertUnwindSafe(|| {
            writer.add_document(doc!(text => "missing explicit id"))
        }));
        let add_result = call_result.expect("add_document must not panic");
        let error = add_result.expect_err("add_document must reject user specified ID schemas");
        assert!(matches!(error, TantivyError::InvalidArgument(_)));

        writer.add_documents_with_doc_ids(vec![(0, doc!(text => "valid"))])?;
        writer.finalize()?;
        Ok(())
    }

    #[test]
    fn test_document_panic_is_captured_and_poisons_writer() -> crate::Result<()> {
        let mut writer = Index::builder()
            .schema(Schema::builder().build())
            .single_segment_index_writer::<PanicDocument>(RamDirectory::default(), MEMORY_BUDGET)?;

        let call_result = catch_unwind(AssertUnwindSafe(|| writer.add_document(PanicDocument)));
        let add_result = call_result.expect("document panic must not escape add_document");
        let first_error = add_result.expect_err("document panic must become a Tantivy error");
        assert!(matches!(first_error, TantivyError::ErrorInThread(_)));
        assert!(first_error.to_string().contains("panic document sentinel"));

        let second_error = writer
            .add_document(PanicDocument)
            .expect_err("panic must poison the writer");
        assert_eq!(second_error.to_string(), first_error.to_string());
        Ok(())
    }

    #[test]
    fn test_tokenizer_panic_is_captured_and_poisons_writer() -> crate::Result<()> {
        let text_options = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("panic")
                .set_index_option(IndexRecordOption::Basic),
        );
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", text_options);
        let tokenizers = TokenizerManager::default();
        tokenizers.register("panic", PanicTokenizer);
        let mut writer = Index::builder()
            .schema(schema_builder.build())
            .tokenizers(tokenizers)
            .single_segment_index_writer::<TantivyDocument>(
                RamDirectory::default(),
                MEMORY_BUDGET,
            )?;

        let call_result = catch_unwind(AssertUnwindSafe(|| {
            writer.add_document(doc!(text => "panic"))
        }));
        let add_result = call_result.expect("tokenizer panic must not escape add_document");
        let first_error = add_result.expect_err("tokenizer panic must become a Tantivy error");
        assert!(matches!(first_error, TantivyError::ErrorInThread(_)));
        assert!(first_error.to_string().contains("panic tokenizer sentinel"));

        let second_error = writer
            .add_document(TantivyDocument::default())
            .expect_err("tokenizer panic must poison the writer");
        assert_eq!(second_error.to_string(), first_error.to_string());
        Ok(())
    }

    #[test]
    fn test_mem_usage_is_available_synchronously() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("text", TEXT);
        let writer = Index::builder()
            .schema(schema_builder.build())
            .single_segment_index_writer::<TantivyDocument>(
                RamDirectory::default(),
                MEMORY_BUDGET,
            )?;

        assert!(writer.mem_usage() > 0);
        Ok(())
    }

    #[test]
    fn test_default_writer_disables_dedicated_docstore_compression() -> crate::Result<()> {
        let writer = Index::builder()
            .schema(Schema::builder().build())
            .single_segment_index_writer::<TantivyDocument>(
                RamDirectory::default(),
                MEMORY_BUDGET,
            )?;

        assert!(
            !writer
                .segment
                .index()
                .settings()
                .docstore_compress_dedicated_thread
        );
        Ok(())
    }

    #[test]
    fn test_opstamp_overflow_is_prevalidated_without_poisoning() -> crate::Result<()> {
        let (text, _directory, mut writer) = user_doc_id_writer()?;
        writer.next_opstamp = u64::MAX;

        let call_result = catch_unwind(AssertUnwindSafe(|| {
            writer.add_documents_with_doc_ids(vec![(0, doc!(text => "overflow"))])
        }));
        let add_result = call_result.expect("opstamp validation must not panic");
        let error = add_result.expect_err("opstamp overflow must be rejected");
        assert!(matches!(error, TantivyError::InvalidArgument(_)));
        assert_eq!(writer.next_opstamp, u64::MAX);

        writer.next_opstamp = 0;
        writer.add_documents_with_doc_ids(vec![(0, doc!(text => "valid"))])?;
        writer.finalize()?;
        Ok(())
    }

    #[test]
    fn test_finalize_preserves_settings_schema_and_meta_contract() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let mut settings = IndexSettings::default();
        settings.docstore_blocksize = 4_096;
        let mut expected_settings = settings.clone();
        expected_settings.docstore_compress_dedicated_thread = false;
        let directory = RamDirectory::default();
        let mut writer = Index::builder()
            .schema(schema.clone())
            .settings(settings.clone())
            .single_segment_index_writer(directory.clone(), MEMORY_BUDGET)?;

        writer.add_document(doc!(text => "meta"))?;
        writer.finalize()?;

        let index = Index::open(directory)?;
        let meta = index.load_metas()?;
        assert_eq!(meta.segments.len(), 1);
        assert_eq!(meta.segments[0].max_doc(), 1);
        assert_eq!(meta.index_settings, expected_settings);
        assert_eq!(meta.schema, schema);
        assert_eq!(meta.opstamp, 0);
        assert_eq!(meta.payload, None);
        Ok(())
    }
}
