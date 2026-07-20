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
  use core::cell::Cell;
  use std::string::String;

  use super::super::{Chunk, ContentAware, MeasureText};
  use crate::{plan::WindowOptions, WinditError};

  /// The mock tokenizer: whitespace-delimited word count.
  fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
  }

  /// `n` single-token words, the shape the high-overlap packing bound is stated
  /// over.
  fn words(n: usize) -> String {
    let mut text = String::new();
    for i in 0..n {
      if i > 0 {
        text.push(' ');
      }
      text.push('w');
    }
    text
  }

  /// The unit a [`Meter`] tallies.
  #[derive(Clone, Copy)]
  enum Unit {
    Words,
    Chars,
  }

  /// A [`MeasureText`] that tallies how often it is queried, how many units it
  /// consumes, and how far into the source it reaches, so a work bound can be
  /// asserted rather than described.
  ///
  /// [`measure_within`](MeasureText::measure_within) stops at the first unit past
  /// the limit — the early stop a real incremental tokenizer has and a plain
  /// closure does not — so `units` records the work actually spent, not the
  /// length of the text handed in. That is what lets a low-cap regression assert
  /// total measured units track the cap rather than the input.
  ///
  /// `reach` is the highest byte offset at which a measured slice *started*.
  /// Counting calls alone cannot separate "measured the whole input once" from
  /// "measured every atom of it", because the preamble measurements all start at
  /// offset zero however much text follows them; the starting offset says how far
  /// the atom production actually walked. Every slice a chunking measures is a
  /// sub-slice of the one `text` it was handed, so subtracting that text's
  /// address recovers the offset — the only handle on production progress a
  /// caller-supplied measurer has.
  struct Meter {
    base: usize,
    unit: Unit,
    calls: Cell<usize>,
    units: Cell<usize>,
    reach: Cell<usize>,
  }

  impl Meter {
    fn new(text: &str, unit: Unit) -> Self {
      Self {
        base: text.as_ptr() as usize,
        unit,
        calls: Cell::new(0),
        units: Cell::new(0),
        reach: Cell::new(0),
      }
    }

    /// A meter that counts whitespace-delimited words.
    fn words(text: &str) -> Self {
      Self::new(text, Unit::Words)
    }

    /// A meter that counts `char`s.
    fn chars(text: &str) -> Self {
      Self::new(text, Unit::Chars)
    }

    /// Record one query that consumed `units` units of `s`.
    fn note(&self, s: &str, units: usize) {
      self.calls.set(self.calls.get() + 1);
      self.units.set(self.units.get() + units);
      // Saturating because nothing stops a caller from measuring a slice of some
      // other string; every measurement this module makes is of the text it
      // constructed the meter from.
      let start = (s.as_ptr() as usize).saturating_sub(self.base);
      self.reach.set(self.reach.get().max(start));
    }
  }

  impl MeasureText for Meter {
    fn measure(&self, s: &str) -> usize {
      let n = match self.unit {
        Unit::Words => word_count(s),
        Unit::Chars => s.chars().count(),
      };
      self.note(s, n);
      n
    }

    fn measure_within(&self, s: &str, limit: usize) -> Option<usize> {
      // Count units one at a time, stopping the instant the tally passes
      // `limit`. `units` then grows with the limit, not with the length of `s`:
      // the early stop a plain `Fn(&str) -> usize` closure cannot express.
      let mut n = 0usize;
      let over = match self.unit {
        Unit::Words => count_until(s.split_whitespace(), limit, &mut n),
        Unit::Chars => count_until(s.chars(), limit, &mut n),
      };
      self.note(s, n);
      (!over).then_some(n)
    }
  }

  /// Advance `iter`, incrementing `*n`, until it is exhausted (returns `false`)
  /// or `*n` first exceeds `limit` (returns `true`, having stopped early).
  fn count_until<I: Iterator>(iter: I, limit: usize, n: &mut usize) -> bool {
    for _ in iter {
      *n += 1;
      if *n > limit {
        return true;
      }
    }
    false
  }

  #[test]
  fn high_overlap_packing_stays_within_its_measurement_budget() {
    // 1000 one-token words, window 500, overlap 499. Advancing one atom per
    // chunk is the geometry the caller asked for, so ~501 chunks is the correct
    // output; the defect was the work spent producing it. Both packing loops
    // re-measured every growing prefix and every growing suffix, which a
    // loop-exact model puts at 499_999 `len_fn` calls over 125_375_249
    // token-units -- cubic once the window and overlap scale with the input.
    let text = words(1000);
    let meter = Meter::words(&text);
    let opts = WindowOptions::new(500).with_overlap(499);

    let chunks = ContentAware::new(&meter).chunk(&text, &opts).unwrap();

    assert_eq!(chunks.len(), 501, "the packing itself must not change");
    assert!(
      meter.calls.get() < 10_000,
      "packing 1000 atoms at overlap 499 took {} measure queries",
      meter.calls.get()
    );
    assert!(
      meter.units.get() < 2_000_000,
      "packing 1000 atoms at overlap 499 rescanned {} token-units",
      meter.units.get()
    );
  }

  #[test]
  fn chunk_enforces_the_window_cap() {
    // `WindowOptions::max_windows` is honoured by `WindowPlan::spans` but was
    // ignored here, so the one configured bound on how much work a chunking may
    // cost did not reach the chunker at all.
    let text = "a b c d e f g h i j k l";
    let len_fn: &dyn MeasureText = &word_count;

    // Twelve words at window 4 want three chunks; a cap of two aborts as soon as
    // it is first exceeded, reporting the same `got` / `max` pair the planner does.
    let capped = WindowOptions::new(4).with_max_windows(2);
    assert_eq!(
      ContentAware::new(len_fn).chunk(text, &capped),
      Err(WinditError::TooManyWindows { got: 3, max: 2 })
    );

    // A cap the packing stays within leaves the chunks untouched.
    let roomy = WindowOptions::new(4).with_max_windows(3);
    assert_eq!(
      ContentAware::new(len_fn).chunk(text, &roomy).unwrap().len(),
      3
    );
  }

  #[test]
  fn a_cap_bounds_the_atom_production_it_caps() {
    // One long word, `char`-count `len_fn`, window 1, cap 1: every char is its
    // own atom and every atom its own chunk, so the cap is settled by the third
    // atom. The cap used to be read only after the whole input had been split
    // into atoms -- one allocated range per character, copied out through the
    // recursion's nested vectors -- so the work a cap exists to forbid was
    // already spent by the time it was consulted. `TooManyWindows` came back
    // either way, which is why the verdict alone cannot witness this.
    let text = "a".repeat(20_000);
    let meter = Meter::chars(&text);
    let opts = WindowOptions::new(1).with_max_windows(1);

    assert_eq!(
      ContentAware::new(&meter).chunk(&text, &opts),
      Err(WinditError::TooManyWindows { got: 2, max: 1 })
    );
    assert!(
      meter.calls.get() < 64,
      "a cap of 1 cost {} measure queries over a 20_000-char word",
      meter.calls.get()
    );
    assert!(
      meter.reach.get() < 64,
      "a cap of 1 walked {} bytes into a 20_000-byte word",
      meter.reach.get()
    );
    // The meter-based bound Codex R4 asked for: pre-change, the descent measured
    // the whole 20_000-char word before the cap could apply (80_010 char-units);
    // the bounded query stops at window + 1 per parent range, so the units spent
    // track the cap and window, not the input length.
    assert!(
      meter.units.get() < 100,
      "a cap of 1 measured {} char-units of a 20_000-char word; must track the cap, not the input",
      meter.units.get()
    );
  }

  #[test]
  fn a_low_cap_bounds_the_work_on_a_large_input() {
    // The same class at the word level, where the atoms are whole words rather
    // than characters: 50_000 one-token words at window 4 under a cap of 3 is
    // settled by the seventeenth word. Tokenizing the other 49_983 is work the
    // cap was set to prevent.
    let text = words(50_000);
    let meter = Meter::words(&text);
    let opts = WindowOptions::new(4).with_max_windows(3);

    assert_eq!(
      ContentAware::new(&meter).chunk(&text, &opts),
      Err(WinditError::TooManyWindows { got: 4, max: 3 })
    );
    assert!(
      meter.calls.get() < 128,
      "a cap of 3 cost {} measure queries over 50_000 words",
      meter.calls.get()
    );
    assert!(
      meter.reach.get() < 256,
      "a cap of 3 walked {} bytes into a {}-byte input",
      meter.reach.get(),
      text.len()
    );
    // The meter-based bound: pre-change, the descent measured the whole 50_000
    // word input (150_085 token-units) before the cap of 3 could apply; the
    // bounded query keeps the units spent proportional to the cap and window.
    assert!(
      meter.units.get() < 1_000,
      "a cap of 3 measured {} token-units of 50_000 words; must track the cap, not the input",
      meter.units.get()
    );
  }

  #[test]
  fn measured_units_track_the_cap_not_the_input() {
    // Codex R4, verbatim: the first atom request called split on the whole input
    // and measured that entire substring "before producing anything or checking
    // the window cap", so the meter "necessarily records at least 50_000
    // token-units from this initial call" even at a cap of one. The fix is the
    // interface: MeasureText::measure_within lets the descent measure a parent
    // range only far enough to learn it overflows the window, so the units spent
    // are set by the cap and window, not the input length. Ten times the input
    // therefore costs the same bounded work.
    let opts = WindowOptions::new(4).with_max_windows(1);

    let small = words(5_000);
    let m_small = Meter::words(&small);
    assert_eq!(
      ContentAware::new(&m_small).chunk(&small, &opts),
      Err(WinditError::TooManyWindows { got: 2, max: 1 })
    );

    let large = words(50_000);
    let m_large = Meter::words(&large);
    assert_eq!(
      ContentAware::new(&m_large).chunk(&large, &opts),
      Err(WinditError::TooManyWindows { got: 2, max: 1 })
    );

    // A small constant, not ~5_000 and ~50_000: the units spent track the cap,
    // and a 10x-larger input is measured no more than the smaller one.
    assert!(
      m_large.units.get() < 200,
      "large-input units {} must track the cap, not the input",
      m_large.units.get()
    );
    assert_eq!(
      m_small.units.get(),
      m_large.units.get(),
      "a 10x-larger input under the same cap measured the same bounded work: {} vs {}",
      m_small.units.get(),
      m_large.units.get()
    );
  }

  #[test]
  fn packs_words_within_window() {
    let text = "a b c d e f g h";
    let len_fn: &dyn MeasureText = &word_count;
    let chunks = ContentAware::new(len_fn)
      .chunk(text, &WindowOptions::new(5))
      .unwrap();

    let slices: std::vec::Vec<&str> = chunks.iter().map(|c| c.as_str(text).unwrap()).collect();
    assert_eq!(slices, std::vec!["a b c d e", "f g h"]);
    for c in &chunks {
      assert!(word_count(c.as_str(text).unwrap()) <= 5);
    }
    // Full coverage of the token span: first chunk at the start, last at the end.
    assert_eq!(chunks[0].start(), 0);
    assert_eq!(chunks.last().unwrap().end(), text.len());
  }

  #[test]
  fn respects_sentence_boundaries() {
    // Two sentences (2 and 4 words) that together exceed window 5: the packer
    // breaks at the sentence boundary rather than mid-sentence.
    let text = "One two. Three four five six.";
    let len_fn: &dyn MeasureText = &word_count;
    let chunks = ContentAware::new(len_fn)
      .chunk(text, &WindowOptions::new(5))
      .unwrap();

    assert_eq!(chunks.len(), 2);
    let first = chunks[0].as_str(text).unwrap();
    let second = chunks[1].as_str(text).unwrap();
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
    let len_fn: &dyn MeasureText = &word_count;
    let opts = WindowOptions::new(5).with_overlap(1);
    let chunks = ContentAware::new(len_fn).chunk(text, &opts).unwrap();

    let slices: std::vec::Vec<&str> = chunks.iter().map(|c| c.as_str(text).unwrap()).collect();
    assert_eq!(slices, std::vec!["a b c d e", "e f g h"]);
  }

  /// Measure the tokens repeated between each pair of consecutive chunks.
  fn repeated_tokens(text: &str, chunks: &[Chunk]) -> std::vec::Vec<usize> {
    chunks
      .windows(2)
      .map(|w| {
        let e0 = w[0].end();
        let s1 = w[1].start();
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
    let len_fn: &dyn MeasureText = &word_count;
    let opts = WindowOptions::new(12).with_overlap(4);
    let chunks = ContentAware::new(len_fn).chunk(text, &opts).unwrap();

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
    let len_fn: &dyn MeasureText = &word_count;
    let opts = WindowOptions::new(24).with_overlap(6);
    let chunks = ContentAware::new(len_fn).chunk(&text, &opts).unwrap();

    assert_eq!(chunks.len(), 2, "token-budget packing yields 2 chunks");
    let repeats = repeated_tokens(&text, &chunks);
    assert!(
      repeats.iter().all(|&r| r <= 6),
      "no repeat may exceed the 6-token overlap budget, got {repeats:?}"
    );
  }

  /// Codex R5 finding 4: a context-sensitive measurer whose token count does not
  /// fall monotonically as a suffix shortens. Word count, except two trailing
  /// suffixes of the four-atom chunk `"a b c d"` are pinned so its overlap-suffix
  /// measures are `t=1 -> "b c d" = 3`, `t=2 -> "c d" = 1`, `t=3 -> "d" = 2`. BPE
  /// and wordpiece tokenizers are non-monotonic in exactly this way.
  struct NonMonotone;

  impl MeasureText for NonMonotone {
    fn measure(&self, s: &str) -> usize {
      match s {
        "c d" => 1,
        "d" => 2,
        _ => word_count(s),
      }
    }
  }

  #[test]
  fn overlap_search_finds_earliest_fit_when_measure_is_not_monotone() {
    // The overlap-boundary search must repeat the EARLIEST trailing suffix that
    // fits the overlap budget, for an arbitrary measurer. For "a b c d" (window
    // 4) at overlap 1 the suffix measures are t=1 -> "b c d" = 3, t=2 -> "c d" =
    // 1, t=3 -> "d" = 2; the earliest that fits the budget of 1 is t=2 ("c d").
    // A doubling+bisection search probes t=1 then t=3, skips t=2, and falls
    // through to the t=4 "repeat nothing" sentinel -- silently dropping the
    // configured overlap. A linear scan over the closed chunk's trailing atoms
    // finds t=2.
    let text = "a b c d e";
    let opts = WindowOptions::new(4).with_overlap(1);
    let chunks = ContentAware::new(&NonMonotone).chunk(text, &opts).unwrap();

    let slices: std::vec::Vec<&str> = chunks.iter().map(|c| c.as_str(text).unwrap()).collect();
    assert_eq!(
      slices,
      std::vec!["a b c d", "c d e"],
      "the second chunk must repeat the earliest fitting suffix \"c d\" (t=2), not \
       start fresh at \"e\" (the t=4 no-overlap sentinel)"
    );
    // The overlap boundary itself: byte 4 is atom "c", the start of the repeated
    // "c d". The pre-fix search returns byte 8 ("e"), repeating nothing.
    assert_eq!(chunks[1].start(), 4);
  }

  #[test]
  fn overlap_boundary_is_unchanged_for_a_monotone_measurer() {
    // The same input and geometry under a monotonic measurer (word count): the
    // suffix measures fall monotonically (t=1 -> 3, t=2 -> 2, t=3 -> 1), so the
    // earliest fitting the budget of 1 is t=3 ("d") and the second chunk repeats
    // one token. Linear scan and bisection agree whenever the measure is
    // monotone, which is what keeps every existing chunk boundary identical.
    let text = "a b c d e";
    let len_fn: &dyn MeasureText = &word_count;
    let opts = WindowOptions::new(4).with_overlap(1);
    let chunks = ContentAware::new(len_fn).chunk(text, &opts).unwrap();

    let slices: std::vec::Vec<&str> = chunks.iter().map(|c| c.as_str(text).unwrap()).collect();
    assert_eq!(slices, std::vec!["a b c d", "d e"]);
    assert_eq!(chunks[1].start(), 6);
  }

  #[test]
  fn chunk_rejects_invalid_window() {
    // Invalid geometry cannot produce a range that honours the window, so it is
    // reported rather than answered with per-atom ranges that all violate the
    // "<= window" guarantee.
    let len_fn: &dyn MeasureText = &word_count;
    assert_eq!(
      ContentAware::new(len_fn).chunk("hello world", &WindowOptions::new(0)),
      Err(WinditError::ZeroWindow)
    );
    assert!(matches!(
      ContentAware::new(len_fn).chunk("hello world", &WindowOptions::new(4).with_overlap(4)),
      Err(WinditError::OverlapGeWindow { .. })
    ));
  }

  #[test]
  fn single_oversized_char_is_the_sole_exception() {
    // A single char that alone measures more than the window cannot be split
    // further, so it is emitted as-is: the one documented exception to the
    // "every range measures <= window" guarantee.
    let text = "ab";
    let len3 = |s: &str| s.chars().count() * 3; // each char measures 3 tokens
    let len_fn: &dyn MeasureText = &len3;
    let chunks = ContentAware::new(len_fn)
      .chunk(text, &WindowOptions::new(2))
      .unwrap();

    let slices: std::vec::Vec<&str> = chunks.iter().map(|c| c.as_str(text).unwrap()).collect();
    assert_eq!(slices, std::vec!["a", "b"]);
    for c in &chunks {
      assert!(
        len3(c.as_str(text).unwrap()) > 2,
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
    let len_fn: &dyn MeasureText = &word_count;
    let chunks = ContentAware::new(len_fn)
      .chunk(&text, &WindowOptions::new(5))
      .unwrap();

    assert_eq!(chunks.len(), 20);
    for c in &chunks {
      assert!(word_count(c.as_str(&text).unwrap()) <= 5);
    }
    assert_eq!(chunks[0].start(), 0);
    assert_eq!(chunks.last().unwrap().end(), text.len());
  }

  #[test]
  fn empty_and_whitespace_yield_no_chunks() {
    let len_fn: &dyn MeasureText = &word_count;
    assert!(ContentAware::new(len_fn)
      .chunk("", &WindowOptions::new(5))
      .unwrap()
      .is_empty());
    assert!(ContentAware::new(len_fn)
      .chunk("   \n\n  ", &WindowOptions::new(5))
      .unwrap()
      .is_empty());
  }
}
