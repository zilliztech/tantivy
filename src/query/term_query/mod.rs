mod term_query;
mod term_scorer;
mod term_weight;

pub use self::term_query::TermQuery;
pub use self::term_scorer::TermScorer;
#[cfg(test)]
mod tests {

    use std::collections::{BTreeMap, HashSet};

    use crate::collector::{DocSetCollector, TopDocs};
    use crate::docset::{DocSet, COLLECT_BLOCK_BUFFER_LEN};
    use crate::merge_policy::NoMergePolicy;
    use crate::postings::compression::COMPRESSION_BLOCK_SIZE;
    use crate::query::weight::for_each_docset_buffered;
    use crate::query::{EnableScoring, Query, QueryParser, Scorer, TermQuery};
    use crate::schema::{
        Field, IndexRecordOption, OwnedValue, Schema, TextFieldIndexing, TextOptions, Value,
        INDEXED, STORED, STRING, TEXT,
    };
    use crate::{
        assert_nearly_equals, DocAddress, Index, IndexWriter, TantivyDocument, Term, TERMINATED,
    };

    #[test]
    fn test_term_query_no_score_batches() -> crate::Result<()> {
        use IndexRecordOption::{Basic, WithFreqs, WithFreqsAndPositions};

        let mut builder = Schema::builder();
        let string = builder.add_text_field("string", STRING);
        let freqs = builder.add_text_field(
            "freqs",
            TextOptions::default()
                .set_indexing_options(TextFieldIndexing::default().set_index_option(WithFreqs)),
        );
        let positions = builder.add_text_field("positions", TEXT);
        let number = builder.add_u64_field("number", INDEXED);
        let json = builder.add_json_field("json", TEXT);
        let schema = builder.build();
        let mut json_text = Term::from_field_json_path(json, "text", false);
        json_text.append_type_and_str("needle");
        let mut json_number = Term::from_field_json_path(json, "number", false);
        json_number.append_type_and_fast_value(7i64);
        let terms = [
            (Term::from_field_text(string, "needle"), Basic),
            (Term::from_field_text(freqs, "needle"), WithFreqs),
            (
                Term::from_field_text(positions, "needle"),
                WithFreqsAndPositions,
            ),
            (Term::from_field_text(positions, "needle"), Basic),
            (Term::from_field_u64(number, 7), Basic),
            (json_text, WithFreqsAndPositions),
            (json_number, Basic),
        ];

        for hits in [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257] {
            let index = Index::create_in_ram(schema.clone());
            let mut writer: IndexWriter = index.writer_for_tests()?;
            // Interleaved nonmatches make skipped, duplicated and padded IDs visible.
            for row in 0..2 * hits + 3 {
                let matches = row % 2 == 1 && row < 2 * hits;
                let word = if matches { "needle" } else { "other" };
                let words = format!("{word} {word} filler");
                let value = if matches { 7u64 } else { 11u64 };
                let mut document = doc!(
                    string => word, freqs => words.clone(), positions => words.clone(),
                    number => value
                );
                document.add_object(
                    json,
                    BTreeMap::from([
                        ("text".to_string(), OwnedValue::Str(words)),
                        ("number".to_string(), OwnedValue::I64(value as i64)),
                    ]),
                );
                writer.add_document(document)?;
            }
            writer.commit()?;
            let searcher = index.reader()?.searcher();
            assert_eq!(searcher.segment_readers().len(), 1);
            let segment = searcher.segment_reader(0);
            let expected: Vec<u32> = (0..hits).map(|i| 2 * i + 1).collect();
            let mut expected_batches: Vec<_> = expected
                .chunks(COLLECT_BLOCK_BUFFER_LEN)
                .map(<[u32]>::to_vec)
                .collect();
            // The buffered traversal emits an empty tail for zero/exactly full batches.
            if hits as usize % COLLECT_BLOCK_BUFFER_LEN == 0 {
                expected_batches.push(Vec::new());
            }
            for (term, option) in &terms {
                let query = TermQuery::new(term.clone(), *option);
                for scoring in [
                    EnableScoring::disabled_from_schema(searcher.schema()),
                    EnableScoring::disabled_from_searcher(&searcher),
                    EnableScoring::enabled_from_searcher(&searcher),
                ] {
                    let weight = query.weight(scoring)?;
                    let mut actual = Vec::new();
                    weight.for_each_no_score(segment, &mut |docs| actual.push(docs.to_vec()))?;
                    assert_eq!(actual, expected_batches, "hits={hits}, term={term:?}");
                    let mut buffered = Vec::new();
                    for_each_docset_buffered(
                        &mut weight.scorer(segment, 1.0)?,
                        &mut [0; COLLECT_BLOCK_BUFFER_LEN],
                        |docs| buffered.push(docs.to_vec()),
                    );
                    assert_eq!(actual, buffered, "hits={hits}, term={term:?}");
                }
            }
            // Deletion also uses this traversal. A later insert must survive, and
            // the already-open searcher must keep its original snapshot.
            let term = Term::from_field_text(string, "needle");
            writer.add_document(doc!(string => "needle"))?;
            writer.delete_term(term.clone());
            writer.add_document(doc!(string => "needle"))?;
            writer.commit()?;
            let query = TermQuery::new(term, Basic);
            assert_eq!(
                searcher.search(&query, &DocSetCollector)?.len(),
                hits as usize
            );
            let new_searcher = index.reader()?.searcher();
            // Check the alive count independently of the query traversal.
            assert_eq!(new_searcher.num_docs(), hits as u64 + 4);
            assert_eq!(new_searcher.search(&query, &DocSetCollector)?.len(), 1);
        }
        Ok(())
    }

    #[test]
    fn test_term_query_no_score_collector_with_deletes() -> crate::Result<()> {
        let mut builder = Schema::builder();
        let text = builder.add_text_field("text", STRING);
        let row_id = builder.add_u64_field("row", INDEXED | STORED);
        let index = Index::create_in_ram(builder.build());
        let mut writer: IndexWriter = index.writer_for_tests()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for row in 0..771u64 {
            let word = if row % 2 == 1 { "needle" } else { "other" };
            writer.add_document(doc!(text => word, row_id => row))?;
            if row % 257 == 256 {
                writer.commit()?;
            }
        }
        // Keep the middle segment free of deletions: both collector paths must work.
        let deleted = [1, 127, 256, 515, 767];
        for row in deleted {
            writer.delete_term(Term::from_field_u64(row_id, row));
        }
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 3);
        assert_eq!(
            searcher
                .segment_readers()
                .iter()
                .filter(|s| s.alive_bitset().is_some())
                .count(),
            2
        );
        let query = TermQuery::new(
            Term::from_field_text(text, "needle"),
            IndexRecordOption::Basic,
        );
        let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut expected_live = HashSet::new();
        for (ordinal, segment) in searcher.segment_readers().iter().enumerate() {
            let mut expected_raw = Vec::new();
            for doc_id in 0..segment.max_doc() {
                let address = DocAddress::new(ordinal as u32, doc_id);
                let document: TantivyDocument = searcher.doc(address)?;
                let row = document.get_first(row_id).unwrap().as_u64().unwrap();
                // Derive the oracle from stored input, not postings or the alive bitset.
                if row % 2 == 1 {
                    expected_raw.push(doc_id);
                    if !deleted.contains(&row) {
                        expected_live.insert(address);
                    }
                }
            }
            let mut actual_raw = Vec::new();
            weight.for_each_no_score(segment, &mut |docs| actual_raw.extend_from_slice(docs))?;
            assert_eq!(actual_raw, expected_raw);
        }
        assert_eq!(expected_live.len(), 381);
        assert_eq!(searcher.search(&query, &DocSetCollector)?, expected_live);
        Ok(())
    }

    #[test]
    pub fn test_term_query_no_freq() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", STRING);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            // writing the segment
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            let doc = doc!(text_field => "a");
            index_writer.add_document(doc)?;
            index_writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let term_query = TermQuery::new(
            Term::from_field_text(text_field, "a"),
            IndexRecordOption::Basic,
        );
        let term_weight = term_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let segment_reader = searcher.segment_reader(0);
        let mut term_scorer = term_weight.scorer(segment_reader, 1.0)?;
        assert_eq!(term_scorer.doc(), 0);
        assert_nearly_equals!(term_scorer.score(), 0.28768212);
        Ok(())
    }

    #[test]
    pub fn test_term_query_multiple_of_block_len() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", STRING);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            // writing the segment
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            for _ in 0..COMPRESSION_BLOCK_SIZE {
                let doc = doc!(text_field => "a");
                index_writer.add_document(doc)?;
            }
            index_writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let term_query = TermQuery::new(
            Term::from_field_text(text_field, "a"),
            IndexRecordOption::Basic,
        );
        let term_weight = term_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let segment_reader = searcher.segment_reader(0);
        let mut term_scorer = term_weight.scorer(segment_reader, 1.0)?;
        for i in 0u32..COMPRESSION_BLOCK_SIZE as u32 {
            assert_eq!(term_scorer.doc(), i);
            if i == COMPRESSION_BLOCK_SIZE as u32 - 1u32 {
                assert_eq!(term_scorer.advance(), TERMINATED);
            } else {
                assert_eq!(term_scorer.advance(), i + 1);
            }
        }
        assert_eq!(term_scorer.doc(), TERMINATED);
        Ok(())
    }

    #[test]
    pub fn test_term_weight() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let left_field = schema_builder.add_text_field("left", TEXT);
        let right_field = schema_builder.add_text_field("right", TEXT);
        let large_field = schema_builder.add_text_field("large", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(
                left_field => "left1 left2 left2 left2f2 left2f2 left3 abcde abcde abcde abcde abcde abcde abcde abcde abcde abcewde abcde abcde",
                right_field => "right1 right2",
                large_field => "large0 large1 large2 large3 large4 large5 large6 large7 large8 large9 large10 large11 large12 large13 large14 large15 large16 large17 large18 large19 large20 large21 large22 large23 large24 large25 large26 large27 large28 large29 large30 large31 large32 large33 large34 large35 large36 large37 large38 large39 large40 large41 large42 large43 large44 large45 large46 large47 large48 large49 large50 large51 large52 large53 large54 large55 large56 large57 large58 large59 large60 large61 large62 large63 large64 large65 large66 large67 large68 large69 large70 large71 large72 large73 large74 large75 large76 large77 large78 large79 large80 large81 large82 large83 large84 large85 large86 large87 large88 large89 large90 large91 large92 large93 large94 large95 large96 large97 large98 large99 large100 large101 large102 large103 large104 large105 large106 large107 large108 large109 large110 large111 large112 large113 large114 large115 large116 large117 large118 large119 large120 large121 large122 large123 large124 large125 large126 large127 large128 large129 large130 large131 large132 large133 large134 large135 large136 large137 large138 large139 large140 large141 large142 large143 large144 large145 large146 large147 large148 large149 large150 large151 large152 large153 large154 large155 large156 large157 large158 large159 large160 large161 large162 large163 large164 large165 large166 large167 large168 large169 large170 large171 large172 large173 large174 large175 large176 large177 large178 large179 large180 large181 large182 large183 large184 large185 large186 large187 large188 large189 large190 large191 large192 large193 large194 large195 large196 large197 large198 large199 large200 large201 large202 large203 large204 large205 large206 large207 large208 large209 large210 large211 large212 large213 large214 large215 large216 large217 large218 large219 large220 large221 large222 large223 large224 large225 large226 large227 large228 large229 large230 large231 large232 large233 large234 large235 large236 large237 large238 large239 large240 large241 large242 large243 large244 large245 large246 large247 large248 large249 large250 large251 large252 large253 large254 large255 large256 large257 large258 large259 large260 large261 large262 large263 large264 large265 large266 large267 large268 large269 large270 large271 large272 large273 large274 large275 large276 large277 large278 large279 large280 large281 large282 large283 large284 large285 large286"
            ))?;
            index_writer.add_document(doc!(left_field => "left4 left1"))?;
            index_writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        {
            let term = Term::from_field_text(left_field, "left2");
            let term_query = TermQuery::new(term, IndexRecordOption::WithFreqs);
            let topdocs = searcher.search(&term_query, &TopDocs::with_limit(2))?;
            assert_eq!(topdocs.len(), 1);
            let (score, _) = topdocs[0];
            assert_nearly_equals!(0.77802235, score);
        }
        {
            let term = Term::from_field_text(left_field, "left1");
            let term_query = TermQuery::new(term, IndexRecordOption::WithFreqs);
            let top_docs = searcher.search(&term_query, &TopDocs::with_limit(2))?;
            assert_eq!(top_docs.len(), 2);
            let (score1, _) = top_docs[0];
            assert_nearly_equals!(0.27101856, score1);
            let (score2, _) = top_docs[1];
            assert_nearly_equals!(0.13736556, score2);
        }
        {
            let query_parser = QueryParser::for_index(&index, Vec::new());
            let query = query_parser.parse_query("left:left2 left:left1")?;
            let top_docs = searcher.search(&query, &TopDocs::with_limit(2))?;
            assert_eq!(top_docs.len(), 2);
            let (score1, _) = top_docs[0];
            assert_nearly_equals!(0.9153879, score1);
            let (score2, _) = top_docs[1];
            assert_nearly_equals!(0.27101856, score2);
        }
        Ok(())
    }

    #[test]
    fn test_term_query_count_when_there_are_deletes() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(text_field=>"a b"))?;
        index_writer.add_document(doc!(text_field=>"a c"))?;
        index_writer.delete_term(Term::from_field_text(text_field, "b"));
        index_writer.commit()?;
        let term_a = Term::from_field_text(text_field, "a");
        let term_query = TermQuery::new(term_a, IndexRecordOption::Basic);
        let reader = index.reader()?;
        assert_eq!(term_query.count(&reader.searcher())?, 1);
        Ok(())
    }

    #[test]
    fn test_term_query_simple_seek() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(text_field=>"a"))?;
        index_writer.add_document(doc!(text_field=>"a"))?;
        index_writer.commit()?;
        let term_a = Term::from_field_text(text_field, "a");
        let term_query = TermQuery::new(term_a, IndexRecordOption::Basic);
        let searcher = index.reader()?.searcher();
        let term_weight =
            term_query.weight(EnableScoring::disabled_from_schema(searcher.schema()))?;
        let mut term_scorer = term_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(term_scorer.doc(), 0u32);
        term_scorer.seek(1u32);
        assert_eq!(term_scorer.doc(), 1u32);
        Ok(())
    }

    #[test]
    fn test_term_query_debug() {
        let term_query = TermQuery::new(
            Term::from_field_text(Field::from_field_id(1), "hello"),
            IndexRecordOption::WithFreqs,
        );
        assert_eq!(
            format!("{term_query:?}"),
            r#"TermQuery(Term(field=1, type=Str, "hello"))"#
        );
    }

    #[test]
    fn test_term_query_explain() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(text_field=>"b"))?;
        index_writer.add_document(doc!(text_field=>"a"))?;
        index_writer.add_document(doc!(text_field=>"a"))?;
        index_writer.add_document(doc!(text_field=>"b"))?;
        index_writer.commit()?;
        let term_a = Term::from_field_text(text_field, "a");
        let term_query = TermQuery::new(term_a, IndexRecordOption::Basic);
        let searcher = index.reader()?.searcher();
        {
            let explanation = term_query.explain(&searcher, DocAddress::new(0u32, 1u32))?;
            assert_nearly_equals!(explanation.value(), std::f32::consts::LN_2);
        }
        {
            let explanation_err = term_query.explain(&searcher, DocAddress::new(0u32, 0u32));
            assert!(matches!(
                explanation_err,
                Err(crate::TantivyError::InvalidArgument(_msg))
            ));
        }
        {
            let explanation_err = term_query.explain(&searcher, DocAddress::new(0u32, 3u32));
            assert!(matches!(
                explanation_err,
                Err(crate::TantivyError::InvalidArgument(_msg))
            ));
        }
        Ok(())
    }
}
