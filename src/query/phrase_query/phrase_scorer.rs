use std::cmp::Ordering;

use smallvec::SmallVec;

use crate::docset::{DocSet, TERMINATED};
use crate::fieldnorm::FieldNormReader;
use crate::postings::Postings;
use crate::query::bm25::Bm25Weight;
use crate::query::{Intersection, Scorer};
use crate::{DocId, Score};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PositionSpan {
    left: u32,
    right: u32,
}

impl PositionSpan {
    fn distance(&self) -> u32 {
        self.right - self.left
    }

    fn non_overlap(&self, other: &PositionSpan) -> bool {
        self.right < other.left || self.left > other.right
    }
}

struct PostingsWithOffset<TPostings> {
    offset: u32,
    postings: TPostings,
}

impl<TPostings: Postings> PostingsWithOffset<TPostings> {
    pub fn new(segment_postings: TPostings, offset: u32) -> PostingsWithOffset<TPostings> {
        PostingsWithOffset {
            offset,
            postings: segment_postings,
        }
    }

    pub fn positions(&mut self, output: &mut Vec<u32>) {
        self.postings.positions_with_offset(self.offset, output)
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

pub struct PhraseScorer<TPostings: Postings> {
    intersection_docset: Intersection<PostingsWithOffset<TPostings>, PostingsWithOffset<TPostings>>,
    num_terms: usize,
    left_positions: Vec<u32>,
    right_positions: Vec<u32>,
    phrase_count: u32,
    fieldnorm_reader: FieldNormReader,
    similarity_weight_opt: Option<Bm25Weight>,
    slop: u32,

    current_spans: Vec<PositionSpan>,
    spans_buffer: Vec<PositionSpan>,
}

/// Returns true if and only if the two sorted arrays contain a common element
fn intersection_exists(left: &[u32], right: &[u32]) -> bool {
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() && right_index < right.len() {
        let left_val = left[left_index];
        let right_val = right[right_index];
        match left_val.cmp(&right_val) {
            Ordering::Less => {
                left_index += 1;
            }
            Ordering::Equal => {
                return true;
            }
            Ordering::Greater => {
                right_index += 1;
            }
        }
    }
    false
}

pub(crate) fn intersection_count(left: &[u32], right: &[u32]) -> usize {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut count = 0;
    while left_index < left.len() && right_index < right.len() {
        let left_val = left[left_index];
        let right_val = right[right_index];
        match left_val.cmp(&right_val) {
            Ordering::Less => {
                left_index += 1;
            }
            Ordering::Equal => {
                count += 1;
                left_index += 1;
                right_index += 1;
            }
            Ordering::Greater => {
                right_index += 1;
            }
        }
    }
    count
}

/// Intersect twos sorted arrays `left` and `right` and outputs the
/// resulting array in left.
///
/// Returns the length of the intersection
#[inline]
fn intersection(left: &mut Vec<u32>, right: &[u32]) {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut count = 0;
    let left_len = left.len();
    let right_len = right.len();
    while left_index < left_len && right_index < right_len {
        let left_val = left[left_index];
        let right_val = right[right_index];
        match left_val.cmp(&right_val) {
            Ordering::Less => {
                left_index += 1;
            }
            Ordering::Equal => {
                left[count] = left_val;
                count += 1;
                left_index += 1;
                right_index += 1;
            }
            Ordering::Greater => {
                right_index += 1;
            }
        }
    }
    left.truncate(count);
}

/// Intersect twos sorted arrays `left` and `right` and outputs the
/// resulting array in left_positions if update_left is true.
///
/// Condition for match is that the distance between left and right is less than or equal to `slop`.
///
/// Returns the length of the intersection
#[inline]
fn intersection_count_with_slop(
    left_positions: &mut Vec<u32>,
    right_positions: &[u32],
    slop: u32,
    update_left: bool,
) -> usize {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut count = 0;
    let left_len = left_positions.len();
    let right_len = right_positions.len();
    while left_index < left_len && right_index < right_len {
        let left_val = left_positions[left_index];
        let right_val = right_positions[right_index];

        let distance = left_val.abs_diff(right_val);
        if distance <= slop {
            while left_index + 1 < left_len {
                // there could be a better match
                let next_left_val = left_positions[left_index + 1];
                if next_left_val > right_val {
                    // the next value is outside the range, so current one is the best.
                    break;
                }
                // the next value is better.
                left_index += 1;
            }

            // store the match in left.
            if update_left {
                left_positions[count] = right_val;
            }
            count += 1;
            left_index += 1;
            right_index += 1;
        } else if left_val < right_val {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    if update_left {
        left_positions.truncate(count);
    }

    count
}

/// Identifies matching spans within a positional slop constraint and builds expanded position
/// spans.
///
/// This function analyzes positional relationships between existing position spans
/// (`current_spans`) and new positions (`next_positions`), identifying matches that fall within the
/// allowed distance (`max_slop`). For each existing span, it finds the best matching position based
/// on minimum distance, and constructs new spans that potentially expand the coverage.
///
/// Return the number of non-overlapping spans found during matching
fn intersection_count_with_slop_with_spans(
    current_spans: &mut Vec<PositionSpan>,
    next_positions: &[u32],
    max_slop: u32,
    spans_buffer: &mut Vec<PositionSpan>,
) -> u32 {
    let mut count = 0;
    let mut start_index = 0;
    for prev_qualified in current_spans.iter() {
        let mut best_match = SmallVec::<[PositionSpan; 4]>::new();
        let mut best_match_distance = u32::MAX;
        let mut record_no_expansion = false;
        for idx in start_index..next_positions.len() {
            let pos = next_positions[idx];
            if pos < prev_qualified.left {
                start_index = idx;
                let distance = prev_qualified.right - pos;
                if distance <= max_slop {
                    if distance < best_match_distance {
                        best_match_distance = distance;
                        best_match.clear();
                        best_match.push(PositionSpan {
                            left: pos,
                            right: prev_qualified.right,
                        });
                    } else if distance == best_match_distance {
                        // Record the span if the distance is the same as the best match distance
                        // Ex: [8] and [7, 9] with slop 1
                        // we should have [{7, 8}] as well as [{8, 9}]
                        best_match.push(PositionSpan {
                            left: pos,
                            right: prev_qualified.right,
                        });
                    }
                }
            } else if pos > prev_qualified.right {
                let distance = pos - prev_qualified.left;
                if distance <= max_slop {
                    if distance < best_match_distance {
                        best_match_distance = distance;
                        best_match.clear();
                        best_match.push(PositionSpan {
                            left: prev_qualified.left,
                            right: pos,
                        });
                    } else if distance == best_match_distance {
                        best_match.push(PositionSpan {
                            left: prev_qualified.left,
                            right: pos,
                        });
                    }
                } else {
                    break;
                }
            } else {
                if pos == prev_qualified.left {
                    start_index = idx;
                }
                best_match_distance = prev_qualified.distance();
                if !record_no_expansion {
                    best_match.clear();
                    best_match.push(*prev_qualified);
                    record_no_expansion = true;
                }
            }
        }
        if !best_match.is_empty() {
            for span in best_match.into_iter() {
                // only inc count if the new span is not overlap with the last span
                if spans_buffer
                    .last()
                    .map_or(true, |last: &PositionSpan| last.non_overlap(&span))
                {
                    count += 1;
                }
                spans_buffer.push(span);
            }
        }
    }
    std::mem::swap(current_spans, spans_buffer);
    spans_buffer.clear();
    count
}

fn intersection_exists_with_slop_with_spans(
    current_spans: &[PositionSpan],
    next_positions: &[u32],
    max_slop: u32,
) -> bool {
    let mut span_index = 0;
    let mut positions_index = 0;
    while span_index < current_spans.len() && positions_index < next_positions.len() {
        let span = current_spans[span_index];
        let position = next_positions[positions_index];
        if position < span.left {
            let distance = span.right - position;
            if distance <= max_slop {
                return true;
            }
            positions_index += 1;
        } else if position > span.right {
            let distance = position - span.left;
            if distance <= max_slop {
                return true;
            }
            span_index += 1;
        } else {
            return true;
        }
    }
    false
}

impl<TPostings: Postings> PhraseScorer<TPostings> {
    // If similarity_weight is None, then scoring is disabled.
    pub fn new(
        term_postings: Vec<(usize, TPostings)>,
        similarity_weight_opt: Option<Bm25Weight>,
        fieldnorm_reader: FieldNormReader,
        slop: u32,
    ) -> PhraseScorer<TPostings> {
        Self::new_with_offset(
            term_postings,
            similarity_weight_opt,
            fieldnorm_reader,
            slop,
            0,
        )
    }

    pub(crate) fn new_with_offset(
        term_postings_with_offset: Vec<(usize, TPostings)>,
        similarity_weight_opt: Option<Bm25Weight>,
        fieldnorm_reader: FieldNormReader,
        slop: u32,
        offset: usize,
    ) -> PhraseScorer<TPostings> {
        let max_offset = term_postings_with_offset
            .iter()
            .map(|&(offset, _)| offset)
            .max()
            .unwrap_or(0)
            + offset;
        let num_docsets = term_postings_with_offset.len();
        let postings_with_offsets = term_postings_with_offset
            .into_iter()
            .map(|(offset, postings)| {
                PostingsWithOffset::new(postings, (max_offset - offset) as u32)
            })
            .collect::<Vec<_>>();
        let mut scorer = PhraseScorer {
            intersection_docset: Intersection::new(postings_with_offsets),
            num_terms: num_docsets,
            left_positions: Vec::with_capacity(100),
            right_positions: Vec::with_capacity(100),
            phrase_count: 0u32,
            similarity_weight_opt,
            fieldnorm_reader,
            slop,
            current_spans: Vec::with_capacity(100),
            spans_buffer: Vec::with_capacity(100),
        };
        if scorer.doc() != TERMINATED && !scorer.phrase_match() {
            scorer.advance();
        }
        scorer
    }

    pub fn phrase_count(&self) -> u32 {
        self.phrase_count
    }

    pub(crate) fn get_intersection(&mut self) -> &[u32] {
        intersection(&mut self.left_positions, &self.right_positions);
        &self.left_positions
    }

    fn phrase_match(&mut self) -> bool {
        if self.similarity_weight_opt.is_some() {
            let count = self.compute_phrase_count();
            self.phrase_count = count;
            println!("phrase_count: {}", count);
            count > 0u32
        } else {
            self.phrase_exists()
        }
    }

    fn phrase_exists(&mut self) -> bool {
        self.compute_phrase_match();
        if self.has_slop() {
            intersection_exists_with_slop_with_spans(
                &self.current_spans,
                &self.right_positions[..],
                self.slop,
            )
        } else {
            intersection_exists(&self.left_positions, &self.right_positions[..])
        }
    }

    fn compute_phrase_count(&mut self) -> u32 {
        self.compute_phrase_match();
        if self.has_slop() {
            if self.num_terms > 2 {
                intersection_count_with_slop_with_spans(
                    &mut self.current_spans,
                    &self.right_positions,
                    self.slop,
                    &mut self.spans_buffer,
                )
            } else {
                intersection_count_with_slop(
                    &mut self.left_positions,
                    &self.right_positions[..],
                    self.slop,
                    false,
                ) as u32
            }
        } else {
            intersection_count(&self.left_positions, &self.right_positions[..]) as u32
        }
    }

    fn compute_phrase_match(&mut self) {
        self.intersection_docset
            .docset_mut_specialized(0)
            .positions(&mut self.left_positions);

        if self.num_terms == 2 {
            // we actually just prepare positions when there are only two terms in this method
            self.intersection_docset
                .docset_mut_specialized(1)
                .positions(&mut self.right_positions);

            return;
        }

        if self.has_slop() {
            // If having slop, we should keep the position span info to consider all possible
            // situations
            self.current_spans.clear();
            self.current_spans.reserve(self.left_positions.len());
            self.current_spans
                .extend(self.left_positions.iter().map(|&pos| PositionSpan {
                    left: pos,
                    right: pos,
                }));
            for i in 1..self.num_terms - 1 {
                self.intersection_docset
                    .docset_mut_specialized(i)
                    .positions(&mut self.right_positions);
                intersection_count_with_slop_with_spans(
                    &mut self.current_spans,
                    &self.right_positions,
                    self.slop,
                    &mut self.spans_buffer,
                );

                if self.current_spans.is_empty() {
                    return;
                }
            }
        } else {
            for i in 1..self.num_terms - 1 {
                self.intersection_docset
                    .docset_mut_specialized(i)
                    .positions(&mut self.right_positions);

                intersection(&mut self.left_positions, &self.right_positions);

                if self.left_positions.is_empty() {
                    return;
                }
            }
        }
        self.intersection_docset
            .docset_mut_specialized(self.num_terms - 1)
            .positions(&mut self.right_positions);
    }

    fn has_slop(&self) -> bool {
        self.slop > 0
    }
}

impl<TPostings: Postings> DocSet for PhraseScorer<TPostings> {
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

impl<TPostings: Postings> Scorer for PhraseScorer<TPostings> {
    fn score(&mut self) -> Score {
        let doc = self.doc();
        let fieldnorm_id = self.fieldnorm_reader.fieldnorm_id(doc);
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            similarity_weight.score(fieldnorm_id, self.phrase_count)
        } else {
            1.0f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_intersection_sym(left: &[u32], right: &[u32], expected: &[u32]) {
        test_intersection_aux(left, right, expected, 0);
        test_intersection_aux(right, left, expected, 0);
    }

    fn test_intersection_aux(left: &[u32], right: &[u32], expected: &[u32], slop: u32) {
        let mut left_vec = Vec::from(left);
        if slop == 0 {
            assert_eq!(intersection_count(&left_vec, right), expected.len());
            intersection(&mut left_vec, right);
            assert_eq!(&left_vec, expected);
        } else {
            let mut right_vec = Vec::from(right);
            let right_mut = &mut right_vec[..];
            intersection_count_with_slop(&mut left_vec, right_mut, slop, true);
            assert_eq!(&left_vec, expected);
        }
    }

    #[test]
    fn test_intersection() {
        test_intersection_sym(&[1], &[1], &[1]);
        test_intersection_sym(&[1], &[2], &[]);
        test_intersection_sym(&[], &[2], &[]);
        test_intersection_sym(&[5, 7], &[1, 5, 10, 12], &[5]);
        test_intersection_sym(&[1, 5, 6, 9, 10, 12], &[6, 8, 9, 12], &[6, 9, 12]);
    }
    #[test]
    fn test_slop() {
        // The slop is not symmetric. It does not allow for the phrase to be out of order.
        test_intersection_aux(&[1], &[2], &[2], 1);
        test_intersection_aux(&[1], &[3], &[], 1);
        test_intersection_aux(&[1], &[3], &[3], 2);
        test_intersection_aux(&[], &[2], &[], 100000);
        test_intersection_aux(&[5, 7, 11], &[1, 5, 10, 12], &[5, 10], 1);
        test_intersection_aux(&[1, 5, 6, 9, 10, 12], &[6, 8, 9, 12], &[6, 8, 9, 12], 1);
        test_intersection_aux(&[1, 5, 6, 9, 10, 12], &[6, 8, 9, 12], &[6, 8, 9, 12], 10);
        test_intersection_aux(&[1, 3, 5], &[2, 4, 6], &[2, 4, 6], 1);
        test_intersection_aux(&[1, 3, 5], &[2, 4, 6], &[], 0);
    }

    fn test_merge(left: &[u32], right: &[u32], expected_left: &[u32], slop: u32) {
        let mut left_vec = Vec::from(left);
        let mut right_vec = Vec::from(right);
        let right_mut = &mut right_vec[..];
        intersection_count_with_slop(&mut left_vec, right_mut, slop, true);
        assert_eq!(&left_vec, expected_left);
    }

    #[test]
    fn test_merge_slop() {
        test_merge(&[1, 2], &[1], &[1], 1);
        test_merge(&[3], &[4], &[4], 2);
        test_merge(&[3], &[4], &[4], 2);
        test_merge(&[1, 5, 6, 9, 10, 12], &[6, 8, 9, 12], &[6, 8, 9, 12], 10);
    }

    fn test_carry_slop_intersection_aux(
        right: &[&[u32]],
        expected: &[PositionSpan],
        slop: u32,
        expected_count: u32,
    ) {
        let left_vec = right[0].to_vec();
        let mut position_spans = left_vec
            .iter()
            .map(|&pos| PositionSpan {
                left: pos,
                right: pos,
            })
            .collect::<Vec<_>>();
        let mut count = 0;
        for right in &right[1..] {
            count = intersection_count_with_slop_with_spans(
                &mut position_spans,
                right,
                slop,
                &mut Vec::new(),
            );
        }
        assert_eq!(&position_spans, expected);
        assert_eq!(count, expected_count);
    }

    #[test]
    fn test_carry_slop_intersection() {
        test_carry_slop_intersection_aux(&[&[1], &[]], &[], 1, 0);
        test_carry_slop_intersection_aux(&[&[1], &[3]], &[], 1, 0);
        test_carry_slop_intersection_aux(
            &[&[1], &[2]],
            &[PositionSpan { left: 1, right: 2 }],
            1,
            1,
        );

        test_carry_slop_intersection_aux(
            &[&[1], &[2], &[2]],
            &[PositionSpan { left: 1, right: 2 }],
            1,
            1,
        );
        test_carry_slop_intersection_aux(
            &[&[2], &[1], &[2]],
            &[PositionSpan { left: 1, right: 2 }],
            1,
            1,
        );
        test_carry_slop_intersection_aux(
            &[&[2], &[2], &[1]],
            &[PositionSpan { left: 1, right: 2 }],
            1,
            1,
        );

        test_carry_slop_intersection_aux(
            &[&[2], &[2], &[1], &[2]],
            &[PositionSpan { left: 1, right: 2 }],
            1,
            1,
        );
        test_carry_slop_intersection_aux(
            &[&[1], &[2], &[2], &[2]],
            &[PositionSpan { left: 1, right: 2 }],
            1,
            1,
        );

        test_carry_slop_intersection_aux(
            &[&[1], &[2], &[1]],
            &[PositionSpan { left: 1, right: 2 }],
            1,
            1,
        );

        test_carry_slop_intersection_aux(
            &[&[11], &[10, 12]],
            &[
                PositionSpan {
                    left: 10,
                    right: 11,
                },
                PositionSpan {
                    left: 11,
                    right: 12,
                },
            ],
            1,
            1,
        );
        test_carry_slop_intersection_aux(
            &[&[10, 12], &[11]],
            &[
                PositionSpan {
                    left: 10,
                    right: 11,
                },
                PositionSpan {
                    left: 11,
                    right: 12,
                },
            ],
            1,
            1,
        );

        test_carry_slop_intersection_aux(
            &[&[5, 7, 11], &[1, 5, 10, 12]],
            &[
                PositionSpan { left: 5, right: 5 },
                PositionSpan {
                    left: 10,
                    right: 11,
                },
                PositionSpan {
                    left: 11,
                    right: 12,
                },
            ],
            1,
            2,
        );

        test_carry_slop_intersection_aux(
            &[&[5, 7, 11], &[1, 5, 10, 12]],
            &[
                PositionSpan { left: 5, right: 5 },
                PositionSpan { left: 5, right: 7 },
                PositionSpan {
                    left: 10,
                    right: 11,
                },
                PositionSpan {
                    left: 11,
                    right: 12,
                },
            ],
            3,
            2,
        );
    }
}

#[cfg(all(test, feature = "unstable"))]
mod bench {

    use test::Bencher;

    use super::*;

    #[bench]
    fn bench_intersection_short_with_slop_with_spans(b: &mut Bencher) {
        let mut left = Vec::new();
        let mut left_slops = Vec::new();
        let mut buffer = Vec::new();
        let mut spans = Vec::new();
        b.iter(|| {
            left.clear();
            left.extend_from_slice(&[1, 5, 10, 12]);
            left_slops.extend_from_slice(&[0, 0, 0, 0]);
            let right = [5, 7];
            intersection(&mut left, &right);

            spans.clear();
            spans.extend(left.iter().map(|&pos| PositionSpan {
                left: pos,
                right: pos,
            }));
            intersection_count_with_slop_with_spans(&mut spans, &right, 2, &mut buffer)
        });
    }

    #[bench]
    fn bench_intersection_short(b: &mut Bencher) {
        let mut left = Vec::new();
        b.iter(|| {
            left.clear();
            left.extend_from_slice(&[1, 5, 10, 12]);
            let right = [5, 7];
            intersection(&mut left, &right);
        });
    }

    #[bench]
    fn bench_intersection_medium_with_slop_with_spans(b: &mut Bencher) {
        let mut buffer = Vec::new();
        let mut spans = Vec::new();
        let left_data: Vec<u32> = (0..100).collect();
        b.iter(|| {
            let right = [5, 7, 55, 200];
            spans.clear();
            spans.extend(left_data.iter().map(|&pos| PositionSpan {
                left: pos,
                right: pos,
            }));
            intersection_count_with_slop_with_spans(&mut spans, &right, 2, &mut buffer)
        });
    }

    #[bench]
    fn bench_intersection_medium_slop(b: &mut Bencher) {
        let mut left = Vec::new();
        let left_data: Vec<u32> = (0..100).collect();

        b.iter(|| {
            left.clear();
            left.extend_from_slice(&left_data);
            let right = [5, 7, 55, 200];
            intersection_count_with_slop(&mut left, &right[..], 2, true) as u32
        });
    }

    #[bench]
    fn bench_intersection_medium(b: &mut Bencher) {
        let mut left = Vec::new();
        let left_data: Vec<u32> = (0..100).collect();
        b.iter(|| {
            left.clear();
            left.extend_from_slice(&left_data);
            let right = [5, 7, 55, 200];
            intersection(&mut left, &right);
        });
    }

    #[bench]
    fn bench_intersection_count_short(b: &mut Bencher) {
        b.iter(|| {
            let left = [1, 5, 10, 12];
            let right = [5, 7];
            intersection_count(&left, &right);
        });
    }
}
