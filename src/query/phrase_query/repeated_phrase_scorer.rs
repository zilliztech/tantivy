use smallvec::SmallVec;

use crate::docset::{DocSet, TERMINATED};
use crate::postings::Postings;
use crate::query::{Intersection, Scorer};
use crate::{DocId, Score};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PositionSpan {
    left: u32,
    right: u32,
}

struct PostingsWithOffset<TPostings> {
    offset: u32,
    postings: TPostings,
}

impl<TPostings: Postings> PostingsWithOffset<TPostings> {
    fn new(postings: TPostings, offset: u32) -> Self {
        Self { offset, postings }
    }

    fn positions(&mut self, output: &mut Vec<u32>) {
        self.postings.positions_with_offset(self.offset, output)
    }

    fn raw_positions(&mut self, output: &mut Vec<u32>) {
        self.postings.positions(output)
    }
}

impl<TPostings: Postings> DocSet for PostingsWithOffset<TPostings> {
    fn advance(&mut self) -> DocId {
        self.postings.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.postings.seek(target)
    }

    fn doc(&self) -> DocId {
        self.postings.doc()
    }

    fn size_hint(&self) -> u32 {
        self.postings.size_hint()
    }
}

fn push_span_frontier(frontier: &mut Vec<PositionSpan>, candidate: PositionSpan) {
    if let Some(last) = frontier.last() {
        debug_assert!(last.left <= candidate.left);
        if last.left == candidate.left && last.right <= candidate.right {
            return;
        }
    }
    while frontier
        .last()
        .is_some_and(|last| last.right >= candidate.right)
    {
        frontier.pop();
    }
    debug_assert!(frontier.last().map_or(true, |last| {
        last.left < candidate.left && last.right < candidate.right
    }));
    frontier.push(candidate);
}

#[derive(Debug)]
struct RepeatedTermGroup {
    postings_index: usize,
    offsets: SmallVec<[u32; 2]>,
}

pub(crate) struct RepeatedPhraseScorer<TPostings: Postings> {
    intersection_docset: Intersection<PostingsWithOffset<TPostings>, PostingsWithOffset<TPostings>>,
    groups: Vec<RepeatedTermGroup>,
    slop: u32,
    right_positions: Vec<u32>,
    position_indices: Vec<usize>,
    current_spans: Vec<PositionSpan>,
    spans_buffer: Vec<PositionSpan>,
    term_spans: Vec<PositionSpan>,
}

/// Builds the minimal span frontier for one term used at multiple query offsets.
///
/// Raw positions are assigned in increasing order, so one token occurrence cannot satisfy two
/// query occurrences. For a lower bound on the adjusted positions, the earliest valid assignment
/// minimizes the right endpoint. Raising the bound past the resulting left endpoint enumerates
/// exactly the non-dominated frontier.
///
/// Each offset's lower bound and predecessor index only increase, so its position cursor never
/// needs to move backwards. With n positions, k offsets and F candidate rounds, this takes
/// O(k * n + F * k) time and O(k) reusable cursor space.
fn build_repeated_term_frontier(
    positions: &[u32],
    offsets: &[u32],
    max_slop: u32,
    frontier: &mut Vec<PositionSpan>,
    position_indices: &mut Vec<usize>,
) {
    debug_assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    debug_assert!(offsets.windows(2).all(|pair| pair[0] >= pair[1]));
    debug_assert!(offsets.len() > 1);
    frontier.clear();
    if positions.len() < offsets.len() {
        return;
    }
    if positions.len() == offsets.len() {
        // The increasing assignment is the only non-dominated choice when every position is used.
        let mut left = u32::MAX;
        let mut right = 0u32;
        for (&position, &offset) in positions.iter().zip(offsets) {
            let adjusted_position = position + offset;
            left = left.min(adjusted_position);
            right = right.max(adjusted_position);
        }
        if right - left <= max_slop {
            frontier.push(PositionSpan { left, right });
        }
        return;
    }

    position_indices.clear();
    position_indices.resize(offsets.len(), 0);
    let mut min_adjusted_position = 0u32;
    'frontier: loop {
        let mut next_position_index = 0;
        let mut left = u32::MAX;
        let mut right = 0u32;
        for (&offset, position_index) in offsets.iter().zip(position_indices.iter_mut()) {
            *position_index = (*position_index).max(next_position_index);
            let min_position = min_adjusted_position.saturating_sub(offset);
            let position = loop {
                let Some(&position) = positions.get(*position_index) else {
                    break 'frontier;
                };
                if position >= min_position {
                    break position;
                }
                *position_index += 1;
            };
            let adjusted_position = position + offset;
            left = left.min(adjusted_position);
            right = right.max(adjusted_position);
            next_position_index = *position_index + 1;
        }

        let candidate = PositionSpan { left, right };
        if right - left <= max_slop {
            push_span_frontier(frontier, candidate);
        }
        let Some(next_minimum) = left.checked_add(1) else {
            break;
        };
        min_adjusted_position = next_minimum;
    }
}

/// Merges two strict span frontiers in linear time.
///
/// For each current span, first find the rightmost term span whose right endpoint is at or before
/// the current right endpoint. It either proves containment or is the best left-only expansion.
/// Later term spans are consumed while they expand both sides; the first right-only expansion is
/// retained because a later current span can still improve it.
fn merge_span_frontiers(
    current_spans: &mut Vec<PositionSpan>,
    next_spans: &[PositionSpan],
    max_slop: u32,
    spans_buffer: &mut Vec<PositionSpan>,
) {
    debug_assert!(spans_buffer.is_empty());
    debug_assert!(current_spans
        .windows(2)
        .all(|pair| { pair[0].left < pair[1].left && pair[0].right < pair[1].right }));
    debug_assert!(next_spans
        .windows(2)
        .all(|pair| { pair[0].left < pair[1].left && pair[0].right < pair[1].right }));

    let mut next_index = 0;
    for &span in current_spans.iter() {
        let mut first_after_right = next_index;
        while first_after_right < next_spans.len()
            && next_spans[first_after_right].right <= span.right
        {
            first_after_right += 1;
        }

        if first_after_right > next_index {
            let predecessor_index = first_after_right - 1;
            let predecessor = next_spans[predecessor_index];
            if predecessor.left >= span.left {
                next_index = if predecessor.left == span.left {
                    first_after_right
                } else {
                    predecessor_index
                };
                push_span_frontier(spans_buffer, span);
                continue;
            }

            let candidate = PositionSpan {
                left: predecessor.left,
                right: span.right,
            };
            if candidate.right - candidate.left <= max_slop {
                push_span_frontier(spans_buffer, candidate);
            }
            next_index = first_after_right;
        }

        while next_index < next_spans.len() && next_spans[next_index].left < span.left {
            let candidate = next_spans[next_index];
            debug_assert!(candidate.right > span.right);
            if candidate.right - candidate.left <= max_slop {
                push_span_frontier(spans_buffer, candidate);
            }
            next_index += 1;
        }

        if let Some(&right_expansion) = next_spans.get(next_index) {
            debug_assert!(right_expansion.right > span.right);
            let candidate = PositionSpan {
                left: span.left,
                right: right_expansion.right,
            };
            if candidate.right - candidate.left <= max_slop {
                push_span_frontier(spans_buffer, candidate);
            }
            if right_expansion.left == span.left {
                next_index += 1;
            }
        }
    }

    std::mem::swap(current_spans, spans_buffer);
    spans_buffer.clear();
}

impl<TPostings: Postings> RepeatedPhraseScorer<TPostings> {
    pub(crate) fn new(
        term_postings_with_ids: Vec<(usize, usize, TPostings)>,
        slop: u32,
    ) -> RepeatedPhraseScorer<TPostings> {
        let max_offset = term_postings_with_ids
            .iter()
            .map(|&(offset, _, _)| offset)
            .max()
            .unwrap_or(0);
        let mut postings_with_ids = term_postings_with_ids
            .into_iter()
            .map(|(offset, term_id, postings)| {
                (
                    PostingsWithOffset::new(postings, (max_offset - offset) as u32),
                    term_id,
                )
            })
            .collect::<Vec<_>>();
        // `Intersection::new` uses the same stable sort. Sorting first lets the repeated-term
        // metadata refer to its final postings index without burdening the common scorer.
        postings_with_ids.sort_by_key(|(postings, _)| postings.size_hint());

        let mut groups = Vec::<RepeatedTermGroup>::new();
        let mut group_indices = vec![None::<usize>; postings_with_ids.len()];
        for (postings_index, (postings, term_id)) in postings_with_ids.iter().enumerate() {
            if let Some(group_index) = group_indices[*term_id] {
                groups[group_index].offsets.push(postings.offset);
            } else {
                group_indices[*term_id] = Some(groups.len());
                groups.push(RepeatedTermGroup {
                    postings_index,
                    offsets: SmallVec::from_slice(&[postings.offset]),
                });
            }
        }
        debug_assert!(groups.iter().any(|group| group.offsets.len() > 1));
        for group in &mut groups {
            group
                .offsets
                .sort_unstable_by(|left, right| right.cmp(left));
        }

        let postings = postings_with_ids
            .into_iter()
            .map(|(postings, _)| postings)
            .collect();
        let mut scorer = RepeatedPhraseScorer {
            intersection_docset: Intersection::new(postings),
            groups,
            slop,
            right_positions: Vec::with_capacity(100),
            position_indices: Vec::new(),
            current_spans: Vec::with_capacity(100),
            spans_buffer: Vec::with_capacity(100),
            term_spans: Vec::with_capacity(100),
        };
        if scorer.doc() != TERMINATED && !scorer.phrase_match() {
            scorer.advance();
        }
        scorer
    }

    fn phrase_match(&mut self) -> bool {
        self.current_spans.clear();
        for (group_index, group) in self.groups.iter().enumerate() {
            let postings = self
                .intersection_docset
                .docset_mut_specialized(group.postings_index);
            if group.offsets.len() == 1 {
                postings.positions(&mut self.right_positions);
                self.right_positions.dedup();
                self.term_spans.clear();
                self.term_spans.reserve(self.right_positions.len());
                self.term_spans
                    .extend(self.right_positions.iter().map(|&position| PositionSpan {
                        left: position,
                        right: position,
                    }));
            } else {
                postings.raw_positions(&mut self.right_positions);
                self.right_positions.dedup();
                build_repeated_term_frontier(
                    &self.right_positions,
                    &group.offsets,
                    self.slop,
                    &mut self.term_spans,
                    &mut self.position_indices,
                );
            }

            if group_index == 0 {
                std::mem::swap(&mut self.current_spans, &mut self.term_spans);
            } else {
                merge_span_frontiers(
                    &mut self.current_spans,
                    &self.term_spans,
                    self.slop,
                    &mut self.spans_buffer,
                );
            }
            if self.current_spans.is_empty() {
                return false;
            }
        }
        true
    }
}

impl<TPostings: Postings> DocSet for RepeatedPhraseScorer<TPostings> {
    fn advance(&mut self) -> DocId {
        loop {
            let doc = self.intersection_docset.advance();
            if doc == TERMINATED || self.phrase_match() {
                return doc;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        debug_assert!(target >= self.doc());
        let doc = self.intersection_docset.seek(target);
        if doc == TERMINATED || self.phrase_match() {
            return doc;
        }
        self.advance()
    }

    fn doc(&self) -> DocId {
        self.intersection_docset.doc()
    }

    fn size_hint(&self) -> u32 {
        self.intersection_docset.size_hint()
    }
}

impl<TPostings: Postings> Scorer for RepeatedPhraseScorer<TPostings> {
    fn score(&mut self) -> Score {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::postings::LoadedPostings;

    fn brute_force_phrase_exists(
        positions: &[Vec<u32>],
        query: &[(usize, usize)],
        max_slop: u32,
    ) -> bool {
        fn visit(
            positions: &[Vec<u32>],
            query: &[(usize, usize)],
            used: &mut Vec<(usize, u32)>,
            span: Option<PositionSpan>,
            max_offset: usize,
            max_slop: u32,
        ) -> bool {
            let Some((&(term, offset), remaining)) = query.split_first() else {
                return true;
            };
            for &position in &positions[term] {
                if used.contains(&(term, position)) {
                    continue;
                }
                let adjusted = position + (max_offset - offset) as u32;
                let expanded = span.map_or(
                    PositionSpan {
                        left: adjusted,
                        right: adjusted,
                    },
                    |span| PositionSpan {
                        left: span.left.min(adjusted),
                        right: span.right.max(adjusted),
                    },
                );
                if expanded.right - expanded.left > max_slop {
                    continue;
                }
                used.push((term, position));
                let matched = visit(
                    positions,
                    remaining,
                    used,
                    Some(expanded),
                    max_offset,
                    max_slop,
                );
                used.pop();
                if matched {
                    return true;
                }
            }
            false
        }

        let max_offset = query.iter().map(|&(_, offset)| offset).max().unwrap();
        visit(
            positions,
            query,
            &mut Vec::new(),
            None,
            max_offset,
            max_slop,
        )
    }

    fn minimal_frontier(mut spans: Vec<PositionSpan>, max_slop: u32) -> Vec<PositionSpan> {
        spans.retain(|span| span.right - span.left <= max_slop);
        spans.sort_unstable_by_key(|span| (span.left, span.right));
        let mut frontier = Vec::new();
        for span in spans {
            push_span_frontier(&mut frontier, span);
        }
        frontier
    }

    fn brute_force_repeated_term_frontier(
        positions: &[u32],
        offsets: &[u32],
        max_slop: u32,
    ) -> Vec<PositionSpan> {
        fn visit(
            positions: &[u32],
            offsets: &[u32],
            offset_index: usize,
            used_positions: u64,
            left: u32,
            right: u32,
            spans: &mut Vec<PositionSpan>,
        ) {
            if offset_index == offsets.len() {
                spans.push(PositionSpan { left, right });
                return;
            }
            for index in 0..positions.len() {
                if used_positions & (1 << index) != 0 {
                    continue;
                }
                let adjusted_position = positions[index] + offsets[offset_index];
                visit(
                    positions,
                    offsets,
                    offset_index + 1,
                    used_positions | (1 << index),
                    left.min(adjusted_position),
                    right.max(adjusted_position),
                    spans,
                );
            }
        }

        let mut spans = Vec::new();
        visit(positions, offsets, 0, 0, u32::MAX, 0, &mut spans);
        minimal_frontier(spans, max_slop)
    }

    #[test]
    fn test_repeated_term_frontier_matches_brute_force() {
        let offset_sets = [
            &[1, 0][..],
            &[1, 1],
            &[3, 0],
            &[2, 1, 1],
            &[2, 1, 0],
            &[4, 2, 0],
        ];
        let mut position_indices = vec![usize::MAX; 4];
        for position_mask in 1u32..1 << 6 {
            let positions = (0..6)
                .filter(|position| position_mask & (1 << position) != 0)
                .collect::<Vec<_>>();
            for offsets in offset_sets {
                for max_slop in 0..=6 {
                    let mut actual = Vec::new();
                    build_repeated_term_frontier(
                        &positions,
                        offsets,
                        max_slop,
                        &mut actual,
                        &mut position_indices,
                    );
                    assert_eq!(
                        actual,
                        brute_force_repeated_term_frontier(&positions, offsets, max_slop),
                        "positions={positions:?}, offsets={offsets:?}, max_slop={max_slop}"
                    );
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1_000))]
        #[test]
        fn test_repeated_phrase_scorer_matches_brute_force(
            mut query in proptest::collection::vec((0usize..3, 0usize..16), 1..6),
            repeated_offset in 0usize..16,
            mut documents in proptest::collection::vec(
                proptest::collection::vec(
                    proptest::collection::vec(0u32..32, 0..5),
                    3..=3,
                ),
                1..7,
            ),
            max_slop in 0u32..64,
        ) {
            query.push((query[0].0, repeated_offset));
            for document in &mut documents {
                for positions in document {
                    positions.sort_unstable();
                }
            }
            let mut first_occurrences = [None; 3];
            let term_ids = query
                .iter()
                .enumerate()
                .map(|(index, &(term, _))| *first_occurrences[term].get_or_insert(index))
                .collect::<Vec<_>>();
            for slop in [max_slop.saturating_sub(1), max_slop, max_slop + 1] {
                let expected = documents
                    .iter()
                    .enumerate()
                    .filter(|(_, document)| brute_force_phrase_exists(document, &query, slop))
                    .map(|(doc, _)| doc as DocId)
                    .collect::<Vec<_>>();
                let make_scorer = || {
                    let postings = query
                        .iter()
                        .zip(&term_ids)
                        .map(|(&(term, offset), &term_id)| {
                            let (doc_ids, positions) = documents
                                .iter()
                                .enumerate()
                                .filter(|(_, document)| !document[term].is_empty())
                                .map(|(doc, document)| (doc as DocId, document[term].clone()))
                                .unzip();
                            (offset, term_id, LoadedPostings::from((doc_ids, positions)))
                        })
                        .collect();
                    RepeatedPhraseScorer::new(postings, slop)
                };
                let mut scorer = make_scorer();
                let mut actual = Vec::new();
                while scorer.doc() != TERMINATED {
                    actual.push(scorer.doc());
                    scorer.advance();
                }
                prop_assert_eq!(&actual, &expected, "query={:?}, slop={}", query, slop);

                let mut seeker = make_scorer();
                for target in 0..documents.len() as DocId {
                    if seeker.doc() != TERMINATED {
                        seeker.seek(target.max(seeker.doc()));
                    }
                    let expected_doc = expected
                        .iter()
                        .copied()
                        .find(|&doc| doc >= target)
                        .unwrap_or(TERMINATED);
                    prop_assert_eq!(seeker.doc(), expected_doc);
                }
            }
        }

        #[test]
        fn test_repeated_term_frontier_matches_random_brute_force(
            position_set in prop_oneof![
                proptest::collection::btree_set(0u32..64, 1..8),
                proptest::collection::btree_set(0u32..4_096, 1..8),
            ],
            mut offsets in prop_oneof![
                proptest::collection::vec(0u32..16, 2..5),
                proptest::collection::vec(0u32..256, 2..5),
            ],
            max_slop in prop_oneof![0u32..16, 0u32..4_096],
        ) {
            let positions = position_set.into_iter().collect::<Vec<_>>();
            offsets.sort_unstable_by(|left, right| right.cmp(left));
            let mut actual = Vec::new();
            build_repeated_term_frontier(
                &positions,
                &offsets,
                max_slop,
                &mut actual,
                &mut Vec::new(),
            );
            prop_assert_eq!(
                actual,
                brute_force_repeated_term_frontier(&positions, &offsets, max_slop)
            );
        }

        #[test]
        fn test_span_frontier_merge_matches_cartesian_product(
            left_spans in proptest::collection::vec((0u32..64, 0u32..16), 1..10),
            right_spans in proptest::collection::vec((0u32..64, 0u32..16), 1..10),
            max_slop in 0u32..64,
        ) {
            let make_frontier = |spans: Vec<(u32, u32)>| {
                minimal_frontier(
                    spans
                        .into_iter()
                        .map(|(left, width)| PositionSpan {
                            left,
                            right: left + width,
                        })
                        .collect(),
                    max_slop,
                )
            };
            let left = make_frontier(left_spans);
            let right = make_frontier(right_spans);
            let expected = minimal_frontier(
                left.iter()
                    .flat_map(|left_span| {
                        right.iter().map(|right_span| PositionSpan {
                            left: left_span.left.min(right_span.left),
                            right: left_span.right.max(right_span.right),
                        })
                    })
                    .collect(),
                max_slop,
            );
            let mut actual = left.clone();
            merge_span_frontiers(&mut actual, &right, max_slop, &mut Vec::new());
            prop_assert_eq!(actual, expected);
        }
    }
}
