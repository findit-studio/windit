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
  use std::string::String;

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
    let chunks = ContentAware::new(len_fn).chunk(text, &WindowOptions::new(5));

    let slices: std::vec::Vec<&str> = chunks.iter().map(|&(s, e)| &text[s..e]).collect();
    assert_eq!(slices, std::vec!["a b c d e", "f g h"]);
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
    let chunks = ContentAware::new(len_fn).chunk(text, &WindowOptions::new(5));

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
    // overlap 1: the second chunk repeats the last token of the first. Words are
    // single-token atoms, so a token budget and an atom count coincide here.
    let text = "a b c d e f g h";
    let len_fn: &dyn Fn(&str) -> usize = &word_count;
    let opts = WindowOptions::new(5).with_overlap(1);
    let chunks = ContentAware::new(len_fn).chunk(text, &opts);

    let slices: std::vec::Vec<&str> = chunks.iter().map(|&(s, e)| &text[s..e]).collect();
    assert_eq!(slices, std::vec!["a b c d e", "e f g h"]);
  }

  /// Measure the tokens repeated between each pair of consecutive chunks.
  fn repeated_tokens(text: &str, chunks: &[(usize, usize)]) -> std::vec::Vec<usize> {
    chunks
      .windows(2)
      .map(|w| {
        let (_, e0) = w[0];
        let (s1, _) = w[1];
        if s1 < e0 {
          word_count(&text[s1..e0])
        } else {
          0
        }
      })
      .collect()
  }

  #[test]
  fn overlap_is_a_token_budget_over_sentence_atoms() {
    // Six capitalized 3-token sentences (18 tokens), window 12, overlap 4. With
    // multi-token atoms the overlap is a TOKEN budget, not an atom count: the
    // packer repeats at most 4 tokens of trailing whole sentences (one sentence,
    // 3 tokens), giving two chunks -- not the advance-by-one slide (3 chunks,
    // 9 repeated tokens) that an atom-count overlap produced.
    let text = "Aa bb cc. Dd ee ff. Gg hh ii. Jj kk ll. Mm nn oo. Pp qq rr.";
    let len_fn: &dyn Fn(&str) -> usize = &word_count;
    let opts = WindowOptions::new(12).with_overlap(4);
    let chunks = ContentAware::new(len_fn).chunk(text, &opts);

    assert_eq!(chunks.len(), 2, "token-budget packing yields 2 chunks");
    let repeats = repeated_tokens(text, &chunks);
    assert_eq!(repeats, std::vec![3]);
    assert!(
      repeats.iter().all(|&r| r <= 4),
      "no repeat may exceed the 4-token overlap budget, got {repeats:?}"
    );
  }

  #[test]
  fn overlap_budget_over_paragraph_atoms() {
    // Five 8-token paragraphs (40 tokens), window 24, overlap 6. A whole
    // paragraph (8 tokens) exceeds the 6-token budget, so no atom is repeated
    // and the packer advances a full chunk: two chunks, zero repeated tokens --
    // NOT the atom-count slide (3 chunks) that repeated 16 tokens per step.
    let para = "The quick brown fox jumps over lazy dogs";
    let text = [para; 5].join("\n\n");
    let len_fn: &dyn Fn(&str) -> usize = &word_count;
    let opts = WindowOptions::new(24).with_overlap(6);
    let chunks = ContentAware::new(len_fn).chunk(&text, &opts);

    assert_eq!(chunks.len(), 2, "token-budget packing yields 2 chunks");
    let repeats = repeated_tokens(&text, &chunks);
    assert!(
      repeats.iter().all(|&r| r <= 6),
      "no repeat may exceed the 6-token overlap budget, got {repeats:?}"
    );
  }

  #[test]
  fn chunk_rejects_invalid_window() {
    // A zero window is invalid geometry; chunk short-circuits to no chunks rather
    // than emitting per-atom ranges that all violate the "<= window" guarantee.
    let len_fn: &dyn Fn(&str) -> usize = &word_count;
    let chunks = ContentAware::new(len_fn).chunk("hello world", &WindowOptions::new(0));
    assert!(chunks.is_empty());
  }

  #[test]
  fn single_oversized_char_is_the_sole_exception() {
    // A single char that alone measures more than the window cannot be split
    // further, so it is emitted as-is: the one documented exception to the
    // "every range measures <= window" guarantee.
    let text = "ab";
    let len3 = |s: &str| s.chars().count() * 3; // each char measures 3 tokens
    let len_fn: &dyn Fn(&str) -> usize = &len3;
    let chunks = ContentAware::new(len_fn).chunk(text, &WindowOptions::new(2));

    let slices: std::vec::Vec<&str> = chunks.iter().map(|&(s, e)| &text[s..e]).collect();
    assert_eq!(slices, std::vec!["a", "b"]);
    for &(s, e) in &chunks {
      assert!(
        len3(&text[s..e]) > 2,
        "the oversize char is emitted despite exceeding the window"
      );
    }
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
    let chunks = ContentAware::new(len_fn).chunk(&text, &WindowOptions::new(5));

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
    assert!(ContentAware::new(len_fn)
      .chunk("", &WindowOptions::new(5))
      .is_empty());
    assert!(ContentAware::new(len_fn)
      .chunk("   \n\n  ", &WindowOptions::new(5))
      .is_empty());
  }
}
