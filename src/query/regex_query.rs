use std::clone::Clone;
use std::sync::Arc;

use tantivy_fst::Regex;

use crate::error::TantivyError;
use crate::query::{AutomatonWeight, EnableScoring, Query, Weight};
use crate::schema::Field;
use crate::Term;

/// A Regex Query matches all of the documents
/// containing a specific term that matches
/// a regex pattern.
///
/// Wildcard queries (e.g. ho*se) can be achieved
/// by converting them to their regex counterparts.
///
/// ```rust
/// use tantivy::collector::Count;
/// use tantivy::query::RegexQuery;
/// use tantivy::schema::{Schema, TEXT};
/// use tantivy::{doc, Index, IndexWriter, Term};
///
/// # fn test() -> tantivy::Result<()> {
/// let mut schema_builder = Schema::builder();
/// let title = schema_builder.add_text_field("title", TEXT);
/// let schema = schema_builder.build();
/// let index = Index::create_in_ram(schema);
/// {
///     let mut index_writer: IndexWriter = index.writer(15_000_000)?;
///     index_writer.add_document(doc!(
///         title => "The Name of the Wind",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "The Diary of Muadib",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "A Dairy Cow",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "The Diary of a Young Girl",
///     ))?;
///     index_writer.commit()?;
/// }
///
/// let reader = index.reader()?;
/// let searcher = reader.searcher();
///
/// let term = Term::from_field_text(title, "Diary");
/// let query = RegexQuery::from_pattern("d[ai]{2}ry", title)?;
/// let count = searcher.search(&query, &Count)?;
/// assert_eq!(count, 3);
/// Ok(())
/// # }
/// # assert!(test().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct RegexQuery {
    regex: Arc<Regex>,
    field: Field,
    json_path_bytes: Option<Vec<u8>>,
}

impl RegexQuery {
    /// Creates a new RegexQuery from a given pattern
    pub fn from_pattern(regex_pattern: &str, field: Field) -> crate::Result<Self> {
        let regex = Regex::new(regex_pattern)
            .map_err(|err| TantivyError::InvalidArgument(format!("RegexQueryError: {err}")))?;
        Ok(RegexQuery::from_regex(regex, field))
    }

    /// Creates a new RegexQuery from a fully built Regex
    pub fn from_regex<T: Into<Arc<Regex>>>(regex: T, field: Field) -> Self {
        RegexQuery {
            regex: regex.into(),
            field,
            json_path_bytes: None,
        }
    }

    /// Creates a new RegexQuery from a given pattern with a json path
    pub fn from_pattern_with_json_path(
        regex_pattern: &str,
        field: Field,
        json_path: &str,
    ) -> crate::Result<Self> {
        // tantivy-fst does not support ^ and $ in regex pattern so it is valid to append regex
        // pattern to the end of the json path
        let mut term = Term::from_field_json_path(field, json_path, false);
        term.append_type_and_str(regex_pattern);
        let regex_text = std::str::from_utf8(term.serialized_value_bytes()).map_err(|err| {
            TantivyError::InvalidArgument(format!(
                "Failed to convert json term value bytes to utf8 string: {err}"
            ))
        })?;
        let regex = Regex::new(regex_text).unwrap();

        if let Some((json_path_bytes, _)) = term.value().as_json() {
            Ok(RegexQuery {
                regex: regex.into(),
                field,
                json_path_bytes: Some(json_path_bytes.to_vec()),
            })
        } else {
            Err(TantivyError::InvalidArgument(format!(
                "The regex query requires a json path for a json term."
            )))
        }
    }

    fn specialized_weight(&self) -> AutomatonWeight<Regex> {
        match &self.json_path_bytes {
            Some(json_path_bytes) => AutomatonWeight::new_for_json_path(
                self.field,
                self.regex.clone(),
                json_path_bytes.as_slice(),
            ),
            None => AutomatonWeight::new(self.field, self.regex.clone()),
        }
    }
}

impl Query for RegexQuery {
    fn weight(&self, _enabled_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(self.specialized_weight()))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use tantivy_fst::Regex;

    use super::RegexQuery;
    use crate::collector::TopDocs;
    use crate::schema::{Field, Schema, STORED, TEXT};
    use crate::{assert_nearly_equals, Index, IndexReader, IndexWriter, TantivyDocument};

    fn build_test_index() -> crate::Result<(IndexReader, Field)> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
            index_writer.add_document(doc!(
                country_field => "japan",
            ))?;
            index_writer.add_document(doc!(
                country_field => "korea",
            ))?;
            index_writer.commit()?;
        }
        let reader = index.reader()?;

        Ok((reader, country_field))
    }

    fn verify_regex_query(
        query_matching_one: RegexQuery,
        query_matching_zero: RegexQuery,
        reader: IndexReader,
    ) {
        let searcher = reader.searcher();
        {
            let scored_docs = searcher
                .search(&query_matching_one, &TopDocs::with_limit(2))
                .unwrap();
            assert_eq!(scored_docs.len(), 1, "Expected only 1 document");
            let (score, _) = scored_docs[0];
            assert_nearly_equals!(1.0, score);
        }
        let top_docs = searcher
            .search(&query_matching_zero, &TopDocs::with_limit(2))
            .unwrap();
        assert!(top_docs.is_empty(), "Expected ZERO document");
    }

    #[test]
    pub fn test_regex_query() -> crate::Result<()> {
        let (reader, field) = build_test_index()?;

        let matching_one = RegexQuery::from_pattern("jap[ao]n", field)?;
        let matching_zero = RegexQuery::from_pattern("jap[A-Z]n", field)?;
        verify_regex_query(matching_one, matching_zero, reader);
        Ok(())
    }

    #[test]
    pub fn test_construct_from_regex() -> crate::Result<()> {
        let (reader, field) = build_test_index()?;

        let matching_one = RegexQuery::from_regex(Regex::new("jap[ao]n").unwrap(), field);
        let matching_zero = RegexQuery::from_regex(Regex::new("jap[A-Z]n").unwrap(), field);

        verify_regex_query(matching_one, matching_zero, reader);
        Ok(())
    }

    #[test]
    pub fn test_construct_from_reused_regex() -> crate::Result<()> {
        let r1 = Arc::new(Regex::new("jap[ao]n").unwrap());
        let r2 = Arc::new(Regex::new("jap[A-Z]n").unwrap());

        let (reader, field) = build_test_index()?;

        let matching_one = RegexQuery::from_regex(r1.clone(), field);
        let matching_zero = RegexQuery::from_regex(r2.clone(), field);

        verify_regex_query(matching_one, matching_zero, reader.clone());

        let matching_one = RegexQuery::from_regex(r1, field);
        let matching_zero = RegexQuery::from_regex(r2, field);

        verify_regex_query(matching_one, matching_zero, reader);
        Ok(())
    }

    #[test]
    pub fn test_pattern_error() {
        let (_reader, field) = build_test_index().unwrap();

        match RegexQuery::from_pattern(r"(foo", field) {
            Err(crate::TantivyError::InvalidArgument(msg)) => {
                assert!(msg.contains("error: unclosed group"))
            }
            res => panic!("unexpected result: {res:?}"),
        }
    }

    #[test]
    pub fn test_regex_query_with_json_path() -> crate::Result<()> {
        std::env::set_var("RUST_BACKTRACE", "1");
        let mut schema_builder = Schema::builder();
        let attributes_field = schema_builder.add_json_field("attributes", TEXT | STORED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        {
            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();

            let doc = TantivyDocument::parse_json(
                &schema,
                r#"{
                "attributes": {
                    "country": {"name": "japan"}
                }
            }"#,
            )?;

            index_writer.add_document(doc)?;
            let doc = TantivyDocument::parse_json(
                &schema,
                r#"{
                "attributes": {
                    "country": {"name": "korea"}
                }
            }"#,
            )?;

            index_writer.add_document(doc)?;
            index_writer.commit()?;
        }
        let reader = index.reader()?;

        let matching_one =
            RegexQuery::from_pattern_with_json_path("j.*", attributes_field, "country.name")?;
        let matching_zero =
            RegexQuery::from_pattern_with_json_path("jap[A-Z]n", attributes_field, "country")?;
        verify_regex_query(matching_one, matching_zero, reader);
        Ok(())
    }
}
