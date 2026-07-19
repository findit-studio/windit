use super::{FixedWindow, SplitPolicy};
use crate::{
  plan::{WindowOptions, WindowPlan},
  WinditError,
};

#[test]
fn fixed_window_split_delegates_to_plan() {
  let opts = WindowOptions::new(4);
  let via_policy = FixedWindow.split(12, &opts).unwrap();
  let via_plan = WindowPlan::spans(&opts, 12).unwrap();
  assert_eq!(via_policy, via_plan);
  assert_eq!(via_policy.len(), 3);
}

#[test]
fn fixed_window_propagates_plan_errors() {
  // A zero window is rejected identically to WindowPlan::spans.
  let opts = WindowOptions::new(0);
  assert!(matches!(
    FixedWindow.split(10, &opts),
    Err(WinditError::ZeroWindow)
  ));
}

#[cfg(feature = "text")]
mod content_aware {
  use alloc::string::String;

  use super::super::ContentAware;
  use crate::plan::WindowOptions;

  /// The mock tokenizer: whitespace-delimited word count.
  fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
  }

  #[test]
  fn packs_words_within_window() {
    let text = "a b c d e f g h";
    let len_fn: &dyn Fn(&str) -> usize = &word_count;
    let chunks = ContentAware { len_fn }.chunk(text, &WindowOptions::new(5));

    let slices: alloc::vec::Vec<&str> = chunks.iter().map(|&(s, e)| &text[s..e]).collect();
    assert_eq!(slices, alloc::vec!["a b c d e", "f g h"]);
    for &(s, e) in &chunks {
      assert!(word_count(&text[s..e]) <= 5);
    }
    // Full coverage of the token span: first chunk at the start, last at the end.
    assert_eq!(chunks[0].0, 0);
    assert_eq!(chunks.last().unwrap().1, text.len());
  }

  #[test]
  fn respects_sentence_boundaries() {
    // Two sentences (2 and 4 words) that together exceed window 5: the packer
    // breaks at the sentence boundary rather than mid-sentence.
    let text = "One two. Three four five six.";
    let len_fn: &dyn Fn(&str) -> usize = &word_count;
    let chunks = ContentAware { len_fn }.chunk(text, &WindowOptions::new(5));

    assert_eq!(chunks.len(), 2);
    let first = &text[chunks[0].0..chunks[0].1];
    let second = &text[chunks[1].0..chunks[1].1];
    assert_eq!(word_count(first), 2);
    assert_eq!(word_count(second), 4);
    assert!(first.contains("two"));
    assert!(!first.contains("Three"));
    assert!(second.trim_start().starts_with("Three"));
  }

  #[test]
  fn overlap_repeats_tail_tokens() {
    // overlap 1: the second chunk repeats the last token of the first.
    let text = "a b c d e f g h";
    let len_fn: &dyn Fn(&str) -> usize = &word_count;
    let opts = WindowOptions::new(5).with_overlap(1);
    let chunks = ContentAware { len_fn }.chunk(text, &opts);

    let slices: alloc::vec::Vec<&str> = chunks.iter().map(|&(s, e)| &text[s..e]).collect();
    assert_eq!(slices, alloc::vec!["a b c d e", "e f g h"]);
  }

  #[test]
  fn oversized_single_unit_falls_back() {
    // 100 whitespace-separated tokens, no sentence boundary: forced to the word
    // level and packed 5 per chunk into 20 chunks.
    let mut text = String::new();
    for i in 0..100 {
      if i > 0 {
        text.push(' ');
      }
      text.push('x');
    }
    let len_fn: &dyn Fn(&str) -> usize = &word_count;
    let chunks = ContentAware { len_fn }.chunk(&text, &WindowOptions::new(5));

    assert_eq!(chunks.len(), 20);
    for &(s, e) in &chunks {
      assert!(word_count(&text[s..e]) <= 5);
    }
    assert_eq!(chunks[0].0, 0);
    assert_eq!(chunks.last().unwrap().1, text.len());
  }

  #[test]
  fn empty_and_whitespace_yield_no_chunks() {
    let len_fn: &dyn Fn(&str) -> usize = &word_count;
    assert!(ContentAware { len_fn }
      .chunk("", &WindowOptions::new(5))
      .is_empty());
    assert!(ContentAware { len_fn }
      .chunk("   \n\n  ", &WindowOptions::new(5))
      .is_empty());
  }
}
