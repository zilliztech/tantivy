use std::sync::Arc;

use super::RepeatedPhraseScorer;
use crate::index::SegmentReader;
use crate::postings::SegmentPostings;
use crate::query::explanation::does_not_match;
use crate::query::{EmptyScorer, Explanation, Scorer, Weight};
use crate::schema::{IndexRecordOption, Term};
use crate::{DocId, DocSet, Score};

pub(crate) struct RepeatedPhraseWeight {
    phrase_terms: Vec<(usize, Term)>,
    term_ids: Arc<[usize]>,
    slop: u32,
}

impl RepeatedPhraseWeight {
    pub(crate) fn new(phrase_terms: Vec<(usize, Term)>, term_ids: Arc<[usize]>, slop: u32) -> Self {
        Self {
            phrase_terms,
            term_ids,
            slop,
        }
    }

    fn phrase_scorer(
        &self,
        reader: &SegmentReader,
    ) -> crate::Result<Option<RepeatedPhraseScorer<SegmentPostings>>> {
        let mut term_postings_list = Vec::new();
        for ((offset, term), &term_id) in self.phrase_terms.iter().zip(self.term_ids.iter()) {
            if let Some(postings) = reader
                .inverted_index(term.field())?
                .read_postings(term, IndexRecordOption::WithFreqsAndPositions)?
            {
                term_postings_list.push((*offset, term_id, postings));
            } else {
                return Ok(None);
            }
        }
        Ok(Some(RepeatedPhraseScorer::new(
            term_postings_list,
            self.slop,
        )))
    }
}

impl Weight for RepeatedPhraseWeight {
    fn scorer(&self, reader: &SegmentReader, _boost: Score) -> crate::Result<Box<dyn Scorer>> {
        if let Some(scorer) = self.phrase_scorer(reader)? {
            Ok(Box::new(scorer))
        } else {
            Ok(Box::new(EmptyScorer))
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let Some(mut scorer) = self.phrase_scorer(reader)? else {
            return Err(does_not_match(doc));
        };
        if scorer.doc() > doc || scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        Ok(Explanation::new("Phrase Scorer", scorer.score()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::phrase_query::tests::create_index;
    use crate::query::{EnableScoring, PhraseQuery, Query};

    #[test]
    fn test_explain_repeated_phrase_without_scoring() -> crate::Result<()> {
        let index = create_index(&["a", "a a", "a x a"])?;
        let field = index.schema().get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let term = Term::from_field_text(field, "a");
        let query = PhraseQuery::new(vec![term.clone(), term]);
        let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let reader = searcher.segment_reader(0);
        assert!(weight.explain(reader, 0).is_err());
        assert!(weight.explain(reader, 1).is_ok());
        assert!(weight.explain(reader, 2).is_err());

        let term = Term::from_field_text(field, "missing");
        let absent = PhraseQuery::new(vec![term.clone(), term]);
        let weight = absent.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        assert!(weight.explain(reader, 0).is_err());

        let term = Term::from_field_text(field, "a");
        let exhausted = PhraseQuery::new(vec![term.clone(), term.clone(), term]);
        let weight = exhausted.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        assert!(weight.explain(reader, 0).is_err());
        Ok(())
    }
}
