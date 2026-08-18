use std::any::Any;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::AddBatch;
use crate::indexer::merger::MAX_DOC_LIMIT;
use crate::indexer::operation::AddOperation;
use crate::indexer::segment_updater::save_metas;
use crate::indexer::SegmentWriter;
use crate::schema::document::Document;
use crate::{
    Directory, Index, IndexMeta, IndexSettings, Opstamp, Segment, TantivyDocument, TantivyError,
};

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

/// Builds an index made of exactly one segment, without spawning any indexing thread.
///
/// Documents are indexed synchronously, on the calling thread: the `add_*` methods return
/// once the documents have been handed over to the underlying `SegmentWriter`, and any
/// failure is reported to the caller that submitted them.
///
/// # Document IDs
///
/// The schema decides which insertion methods are usable, and the two modes are mutually
/// exclusive:
/// - schemas without user specified document IDs: [`Self::add_document`] and
///   [`Self::add_documents`], which assign sequential document IDs starting at `0`;
/// - schemas built with `SchemaBuilder::enable_user_specified_doc_id`:
///   [`Self::add_documents_with_doc_ids`] only.
///
/// Using the wrong method returns [`TantivyError::InvalidArgument`] instead of writing
/// anything.
///
/// # Failure contract
///
/// Failures come in two flavours:
/// - *validation* failures (wrong method for the schema, document ID out of range or out of order,
///   opstamp overflow) are detected before anything is written. The batch is rejected as a whole
///   and the writer remains usable.
/// - *indexing* failures poison the writer permanently. Panics raised by a document or a tokenizer
///   are caught and converted into [`TantivyError::ErrorInThread`], and are treated as indexing
///   failures too. Every subsequent call, [`Self::finalize`] included, returns that first error,
///   and the partially written segment is never published in `meta.json`. The files it already
///   wrote are left behind: this writer runs no garbage collection, so the caller is expected to
///   discard the whole directory.
#[doc(hidden)]
pub struct SingleSegmentIndexWriter<D: Document = TantivyDocument> {
    segment: Segment,
    /// Settings as configured by the caller, captured before [`Self::new`] disabled the doc
    /// store's dedicated compression thread. This is what gets published in `meta.json` and
    /// restored on the `Index` returned by [`Self::finalize`].
    index_settings: IndexSettings,
    /// Whether the schema requires the caller to supply document IDs.
    ///
    /// Cached at construction time: the schema cannot change over the lifetime of an index,
    /// while reading it back from the index clones an `Arc` on every call.
    user_specified_doc_id: bool,
    /// `None` once the writer has been poisoned by an indexing failure, or once it has been
    /// consumed by [`Self::finalize`].
    segment_writer: Option<SegmentWriter>,
    /// The indexing failure that poisoned this writer, if any.
    first_error: Option<TantivyError>,
    /// Opstamp that will be assigned to the next document. With sequential document IDs it
    /// doubles as the next document ID, and hence as the segment's current max doc.
    next_opstamp: Opstamp,
    /// Largest document ID accepted so far. Used to enforce strictly increasing user
    /// specified document IDs across batches.
    last_doc_id: Option<u32>,
    _phantom: PhantomData<D>,
}

impl<D: Document> SingleSegmentIndexWriter<D> {
    pub fn new(mut index: Index, mem_budget: usize) -> crate::Result<Self> {
        let index_settings = index.settings().clone();
        // The doc store's dedicated compression thread only pays off when documents keep
        // arriving from another thread. Here indexing runs synchronously on the caller's
        // thread, so that thread would only add channel hand-offs per block plus a join at
        // finalization. Disable it for the segment we are about to write.
        //
        // This is a write side knob exclusively: no read path looks at it, and the caller's
        // original value is kept in `index_settings` so that neither `meta.json` nor the
        // `Index` returned by `finalize` advertises a value we changed behind their back.
        index.settings_mut().docstore_compress_dedicated_thread = false;
        let user_specified_doc_id = index.schema().user_specified_doc_id();
        let segment = index.new_segment();
        let segment_writer = SegmentWriter::for_segment(mem_budget, segment.clone())?;
        Ok(Self {
            segment,
            index_settings,
            user_specified_doc_id,
            segment_writer: Some(segment_writer),
            first_error: None,
            next_opstamp: 0,
            last_doc_id: None,
            _phantom: PhantomData,
        })
    }

    /// Adds a single document, assigning it the next sequential document ID.
    ///
    /// Requires a schema without user specified document IDs. See the type level
    /// documentation for the document ID and failure contracts.
    pub fn add_document(&mut self, document: D) -> crate::Result<()> {
        self.add_documents(std::iter::once(document))
    }

    /// Adds a batch of documents, assigning them consecutive document IDs starting right
    /// after the last document added so far.
    ///
    /// Requires a schema without user specified document IDs, and an iterator whose length
    /// is known upfront, so that the resulting max doc can be validated before any document
    /// is written. See the type level documentation for the failure contract.
    pub fn add_documents<I>(&mut self, documents: I) -> crate::Result<()>
    where
        I: IntoIterator<Item = D>,
        I::IntoIter: ExactSizeIterator,
    {
        self.ensure_active()?;
        if self.user_specified_doc_id {
            return Err(TantivyError::InvalidArgument(
                "Sequential document IDs cannot be used when the schema enables user specified \
                 document IDs: use add_documents_with_doc_ids"
                    .to_string(),
            ));
        }
        let documents = documents.into_iter();
        if documents.len() == 0 {
            return Ok(());
        }

        // With sequential document IDs the opstamp counter is also the document ID counter,
        // so the opstamp reached at the end of the batch is the segment's resulting max doc.
        let last_opstamp = self.checked_next_opstamp(documents.len())?;
        if last_opstamp >= Opstamp::from(MAX_DOC_LIMIT) {
            return Err(TantivyError::InvalidArgument(format!(
                "Document ID {} is out of range: resulting max doc must be less than \
                 {MAX_DOC_LIMIT}",
                last_opstamp - 1,
            )));
        }

        let first_opstamp = self.next_opstamp;
        let add_operations = documents
            .enumerate()
            .map(|(offset, document)| AddOperation {
                opstamp: first_opstamp + offset as Opstamp,
                document,
                doc_id: None,
            });
        let num_docs = self.write_batch(add_operations)?;
        // Derived from what was actually written rather than from the announced length, so
        // that the opstamp counter cannot drift from the segment's document count.
        self.next_opstamp = first_opstamp + num_docs as Opstamp;
        Ok(())
    }

    /// Adds a batch of documents with user-specified document IDs.
    ///
    /// Requires a schema with user specified document IDs. Document IDs must be strictly
    /// increasing across and within batches. Sparse document IDs are allowed, and leave the
    /// skipped document IDs empty in the resulting segment. See the type level documentation
    /// for the failure contract.
    pub fn add_documents_with_doc_ids<I>(&mut self, documents: I) -> crate::Result<()>
    where
        I: IntoIterator<Item = (u32, D)>,
        I::IntoIter: ExactSizeIterator,
    {
        self.ensure_active()?;
        if !self.user_specified_doc_id {
            return Err(TantivyError::InvalidArgument(
                "User specified document IDs are not enabled on this schema: use add_documents"
                    .to_string(),
            ));
        }
        let documents = documents.into_iter();
        if documents.len() == 0 {
            return Ok(());
        }
        self.checked_next_opstamp(documents.len())?;

        // Unlike sequential document IDs, user specified ones can only be validated against
        // their predecessor, which is unknown before walking the batch. Materialize the
        // operations first so that an invalid document ID rejects the batch as a whole
        // instead of leaving the documents preceding it indexed.
        let mut add_operations = AddBatch::with_capacity(documents.len());
        let mut previous_doc_id = self.last_doc_id;
        let mut opstamp = self.next_opstamp;
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
                        "Document ID must be strictly ordered: previous doc id {previous_doc_id}, \
                         current doc id {doc_id}"
                    )));
                }
            }
            add_operations.push(AddOperation {
                opstamp,
                document,
                doc_id: Some(doc_id),
            });
            previous_doc_id = Some(doc_id);
            opstamp += 1;
        }

        let first_opstamp = self.next_opstamp;
        let num_docs = self.write_batch(add_operations.into_iter())?;
        self.last_doc_id = previous_doc_id;
        self.next_opstamp = first_opstamp + num_docs as Opstamp;
        Ok(())
    }

    /// Opstamp the counter would reach after appending `num_docs` documents, or an error if
    /// that would overflow.
    fn checked_next_opstamp(&self, num_docs: usize) -> crate::Result<Opstamp> {
        Opstamp::try_from(num_docs)
            .ok()
            .and_then(|num_docs| self.next_opstamp.checked_add(num_docs))
            .ok_or_else(|| TantivyError::InvalidArgument("Document opstamp overflow".to_string()))
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

    /// Indexes `add_operations` and returns how many of them were written.
    ///
    /// Any failure, panics included, poisons the writer and drops the segment writer along
    /// with the partially written segment.
    fn write_batch<I>(&mut self, add_operations: I) -> crate::Result<usize>
    where I: Iterator<Item = AddOperation<D>> {
        self.ensure_active()?;
        let mut segment_writer = self.segment_writer.take().ok_or_else(|| {
            TantivyError::InternalError("Single segment writer is unavailable".to_string())
        })?;
        let write_result = catch_unwind(AssertUnwindSafe(|| {
            futures_executor::block_on(async {
                let mut num_docs = 0usize;
                for add_operation in add_operations {
                    segment_writer.add_document(add_operation).await?;
                    num_docs += 1;
                }
                Ok::<usize, TantivyError>(num_docs)
            })
        }));

        match write_result {
            Ok(Ok(num_docs)) => {
                self.segment_writer = Some(segment_writer);
                Ok(num_docs)
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

    /// Serializes the segment, publishes it in `meta.json` and returns the resulting index.
    ///
    /// Returns the first indexing error instead if the writer has been poisoned; in that case
    /// `meta.json` is left without any segment.
    pub fn finalize(mut self) -> crate::Result<Index> {
        if let Some(error) = self.first_error {
            return Err(error);
        }
        let segment_writer = self.segment_writer.take().ok_or_else(|| {
            TantivyError::InternalError("Single segment writer is unavailable".to_string())
        })?;
        let max_doc = segment_writer.max_doc();
        // Backstop for the per-batch validation: whichever path produced the documents, a
        // segment reaching `MAX_DOC_LIMIT` cannot be merged, so refuse to publish it here
        // rather than emitting an index that only blows up much later.
        if max_doc >= MAX_DOC_LIMIT {
            return Err(TantivyError::InvalidArgument(format!(
                "Segment max doc {max_doc} is out of range: max doc must be less than \
                 {MAX_DOC_LIMIT}"
            )));
        }
        match catch_unwind(AssertUnwindSafe(|| {
            futures_executor::block_on(segment_writer.finalize())
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
        let index_settings = self.index_settings;
        let index_meta = IndexMeta {
            index_settings: index_settings.clone(),
            segments: vec![segment.meta().clone()],
            schema: index.schema(),
            opstamp: 0,
            payload: None,
        };
        save_metas(&index_meta, index.directory())?;
        index.directory().sync_directory()?;
        let mut finalized_index = segment.index().clone();
        *finalized_index.settings_mut() = index_settings;
        Ok(finalized_index)
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

    /// An `ExactSizeIterator` announcing a length it does not honor.
    struct MisreportedLen<I> {
        inner: I,
        claimed_len: usize,
    }

    impl<I: Iterator> Iterator for MisreportedLen<I> {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.claimed_len, Some(self.claimed_len))
        }
    }

    impl<I: Iterator> ExactSizeIterator for MisreportedLen<I> {}

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

    fn default_doc_id_writer() -> crate::Result<(
        crate::schema::Field,
        RamDirectory,
        super::SingleSegmentIndexWriter<TantivyDocument>,
    )> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let directory = RamDirectory::default();
        let writer = Index::builder()
            .schema(schema_builder.build())
            .single_segment_index_writer(directory.clone(), MEMORY_BUDGET)?;
        Ok((text, directory, writer))
    }

    #[test]
    fn test_add_documents_assigns_sequential_doc_ids_across_batches() -> crate::Result<()> {
        let (text, directory, mut writer) = default_doc_id_writer()?;
        let direct_segment_id = writer.segment.id();

        writer.add_documents(Vec::<TantivyDocument>::new())?;
        writer.add_documents(vec![doc!(text => "shared"), doc!(text => "other")])?;
        writer.add_documents(vec![doc!(text => "shared")])?;
        writer.finalize()?;

        let index = Index::open(directory)?;
        let segment_metas = index.searchable_segment_metas()?;
        assert_eq!(segment_metas.len(), 1);
        assert_eq!(segment_metas[0].id(), direct_segment_id);
        assert_eq!(segment_metas[0].max_doc(), 3);

        let searcher = index.reader()?.searcher();
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
        assert_eq!(doc_ids, [0, 2]);
        Ok(())
    }

    #[test]
    fn test_add_documents_counts_the_documents_actually_written() -> crate::Result<()> {
        let (text, directory, mut writer) = default_doc_id_writer()?;

        // No standard iterator lies about its length, but a caller side adapter forwarding
        // the wrong `size_hint` does. Such a batch must not punch a hole in the document ID
        // sequence: the counter follows what was written, not what was announced.
        writer.add_documents(MisreportedLen {
            inner: vec![doc!(text => "first")].into_iter(),
            claimed_len: 5,
        })?;
        assert_eq!(writer.next_opstamp, 1);

        writer.add_documents(vec![doc!(text => "second")])?;
        writer.finalize()?;

        let index = Index::open(directory)?;
        let segment_metas = index.searchable_segment_metas()?;
        assert_eq!(segment_metas.len(), 1);
        assert_eq!(segment_metas[0].max_doc(), 2);
        Ok(())
    }

    #[test]
    fn test_add_documents_requires_default_doc_id_schema() -> crate::Result<()> {
        let (text, _directory, mut writer) = user_doc_id_writer()?;

        let empty_batch_error = writer
            .add_documents(Vec::<TantivyDocument>::new())
            .expect_err("empty batches must reject user-specified document ID schemas too");
        assert!(matches!(
            empty_batch_error,
            TantivyError::InvalidArgument(_)
        ));
        let error = writer
            .add_documents(vec![doc!(text => "invalid")])
            .expect_err("non-empty batches must reject user-specified document ID schemas");
        assert!(matches!(error, TantivyError::InvalidArgument(_)));

        writer.add_documents_with_doc_ids(vec![(0, doc!(text => "valid"))])?;
        writer.finalize()?;
        Ok(())
    }

    #[test]
    fn test_add_documents_rejects_max_doc_limit_atomically() -> crate::Result<()> {
        let (text, _directory, mut writer) = default_doc_id_writer()?;
        writer.next_opstamp = u64::from(MAX_DOC_LIMIT - 2);
        let max_doc_before = writer
            .segment_writer
            .as_ref()
            .expect("writer must be active")
            .max_doc();

        let error = writer
            .add_documents(vec![doc!(text => "first"), doc!(text => "out of range")])
            .expect_err("resulting max doc at MAX_DOC_LIMIT must be rejected");

        assert!(matches!(error, TantivyError::InvalidArgument(_)));
        assert_eq!(writer.next_opstamp, u64::from(MAX_DOC_LIMIT - 2));
        assert_eq!(
            writer
                .segment_writer
                .as_ref()
                .expect("validation errors must not poison the writer")
                .max_doc(),
            max_doc_before
        );

        writer.next_opstamp = 0;
        writer.add_documents(vec![doc!(text => "valid")])?;
        writer.finalize()?;
        Ok(())
    }

    #[test]
    fn test_add_documents_rejects_opstamp_overflow_atomically() -> crate::Result<()> {
        let (text, _directory, mut writer) = default_doc_id_writer()?;
        writer.next_opstamp = u64::MAX;

        let error = writer
            .add_documents(vec![doc!(text => "overflow")])
            .expect_err("opstamp overflow must be rejected");
        assert!(matches!(error, TantivyError::InvalidArgument(_)));
        assert_eq!(writer.next_opstamp, u64::MAX);
        assert_eq!(
            writer
                .segment_writer
                .as_ref()
                .expect("validation errors must not poison the writer")
                .max_doc(),
            0
        );

        writer.next_opstamp = 0;
        writer.add_documents(vec![doc!(text => "valid")])?;
        writer.finalize()?;
        Ok(())
    }

    #[test]
    fn test_add_documents_error_drops_partial_segment_without_publishing_meta() -> crate::Result<()>
    {
        let mut schema_builder = Schema::builder();
        let number =
            schema_builder.add_u64_field("number", NumericOptions::default().set_indexed());
        let directory = RamDirectory::default();
        let mut writer = Index::builder()
            .schema(schema_builder.build())
            .single_segment_index_writer(directory.clone(), MEMORY_BUDGET)?;

        let first_error = writer
            .add_documents(vec![
                doc!(number => 123u64),
                doc!(number => "invalid after first write"),
            ])
            .expect_err("the schema error must be returned by the batch call");
        assert!(matches!(first_error, TantivyError::SchemaError(_)));
        assert!(writer.segment_writer.is_none());

        let empty_batch_error = writer
            .add_documents(Vec::<TantivyDocument>::new())
            .expect_err("a poisoned writer must reject later empty batches");
        assert_eq!(empty_batch_error.to_string(), first_error.to_string());

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

        let empty_batch_error = writer
            .add_documents_with_doc_ids(Vec::<(u32, TantivyDocument)>::new())
            .expect_err("empty batches must reject default document ID schemas too");
        assert!(matches!(
            empty_batch_error,
            TantivyError::InvalidArgument(_)
        ));
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
        assert!(writer.segment_writer.is_none());

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
        assert!(writer.segment_writer.is_none());

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
        let directory = RamDirectory::default();
        let mut writer = Index::builder()
            .schema(schema.clone())
            .settings(settings.clone())
            .single_segment_index_writer(directory.clone(), MEMORY_BUDGET)?;

        writer.add_document(doc!(text => "meta"))?;
        let finalized_index = writer.finalize()?;
        assert_eq!(finalized_index.settings(), &settings);

        let index = Index::open(directory)?;
        let meta = index.load_metas()?;
        assert_eq!(meta.segments.len(), 1);
        assert_eq!(meta.segments[0].max_doc(), 1);
        assert_eq!(meta.index_settings, settings);
        assert_eq!(meta.schema, schema);
        assert_eq!(meta.opstamp, 0);
        assert_eq!(meta.payload, None);
        Ok(())
    }
}
