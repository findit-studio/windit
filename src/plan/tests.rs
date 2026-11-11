use super::*;
use std::{vec, vec::Vec};

#[test]
fn exact_windows_no_overlap() {
  let o = WindowOptions::new(4);
  let s = WindowPlan::spans(&o, 12).unwrap();
  assert_eq!(s.len(), 3);
  assert_eq!((s[0].start(), s[0].len()), (0, 4));
  assert_eq!((s[2].start(), s[2].len()), (8, 4));
  assert_eq!(s[0].coverage(), 1.0);
}

#[test]
fn ragged_tail_keeps_partial_with_coverage() {
  let o = WindowOptions::new(4); // 10 elems -> [0..4],[4..8],[8..10 len2 cov .5]
  let s = WindowPlan::spans(&o, 10).unwrap();
  assert_eq!(s.len(), 3);
  assert_eq!((s[2].start(), s[2].len()), (8, 2));
  assert!((s[2].coverage() - 0.5).abs() < 1e-6);
}

#[test]
fn overlap_hops_correctly() {
  let o = WindowOptions::new(4).with_overlap(2); // hop 2
  let s = WindowPlan::spans(&o, 8).unwrap(); // [0..4],[2..6],[4..8]
  assert_eq!(
    s.iter().map(|x| x.start()).collect::<Vec<_>>(),
    vec![0, 2, 4]
  );
}

#[test]
fn drop_below_min_drops_short_tail() {
  let o = WindowOptions::new(4).with_tail(TailPolicy::DropBelowMin(2));
  let s = WindowPlan::spans(&o, 9).unwrap(); // tail len 1 < 2 -> dropped
  assert_eq!(s.last().map(|x| (x.start(), x.len())), Some((4, 4)));
}

#[test]
fn fits_in_one_window_is_single_span() {
  let o = WindowOptions::new(512);
  let s = WindowPlan::spans(&o, 40).unwrap();
  assert_eq!(s.len(), 1);
  assert_eq!((s[0].start(), s[0].len()), (0, 40));
}

#[test]
fn max_windows_errors() {
  let o = WindowOptions::new(4).with_max_windows(2);
  assert!(matches!(
    WindowPlan::spans(&o, 100),
    Err(WinditError::TooManyWindows { .. })
  ));
}

#[test]
fn spans_no_overflow_near_usize_max() {
  // `input_len` within `window` of usize::MAX must not overflow the tail check
  // (a debug panic) nor wrap into an unbounded loop (release). The tail is
  // detected on the second window and packing stops.
  let opts = WindowOptions::new(usize::MAX - 10).with_hop(100);
  let s = WindowPlan::spans(&opts, usize::MAX - 5).unwrap();
  assert_eq!(s.len(), 2);
  assert_eq!((s[0].start(), s[1].start()), (0, 100));

  // A hop whose advance would overflow `start + hop` breaks cleanly instead of
  // panicking: two windows are placed, then the advance saturates and stops.
  let hop = usize::MAX / 2 + 1;
  let s = WindowPlan::spans(&WindowOptions::new(10).with_hop(hop), usize::MAX).unwrap();
  assert_eq!(s.len(), 2);
  assert_eq!((s[0].start(), s[1].start()), (0, hop));
}

#[test]
fn zero_window_and_bad_overlap_error() {
  assert!(matches!(
    WindowOptions::new(0).validate(),
    Err(WinditError::ZeroWindow)
  ));
}

#[test]
fn zero_window_span_is_unconstructible_so_coverage_stays_finite() {
  // A zero window is ruled out by `0 < len <= window` rather than by a separate
  // check, which is what lets `coverage` divide unguarded.
  assert!(matches!(
    Span::try_new(0, 5, 0),
    Err(WinditError::InvalidSpan {
      start: 0,
      len: 5,
      window: 0
    })
  ));
  assert!(matches!(
    Span::try_new(0, 0, 0),
    Err(WinditError::InvalidSpan { .. })
  ));

  let cov = Span::new(0, 1, 4).coverage();
  assert!(cov.is_finite() && cov > 0.0 && cov <= 1.0);
}

#[test]
fn span_new_exposes_geometry_through_accessors() {
  let span = Span::new(8, 3, 4);
  assert_eq!((span.start(), span.len(), span.window()), (8, 3, 4));
  assert!((span.coverage() - 0.75).abs() < 1e-6);
}

/// A geometry past `f32`'s integer-exact range must report the exactly-correct
/// ratio, not one whose operands rounded before the division.
///
/// A window of `16_777_217` (`2^24 + 1`, the first integer `f32` cannot hold)
/// over `16_777_216` elements plans a single ragged tail, one element short of
/// full. Narrowing the operands first rounds that window down to `2^24` and the
/// tail reports as a full window at exactly `1.0` — the ragged tail and the full
/// window become indistinguishable. Both operands are exact in `f64`, so the
/// quotient there is the correctly-rounded ratio.
#[test]
fn coverage_past_f32_integer_range_is_the_exact_ratio() {
  let s = WindowPlan::spans(&WindowOptions::new(16_777_217), 16_777_216).unwrap();
  assert_eq!(s.len(), 1);
  assert_eq!((s[0].len(), s[0].window()), (16_777_216, 16_777_217));

  let cov = s[0].coverage();
  assert!(
    cov < 1.0,
    "a tail one element short of the window must not report full coverage, got {cov:?}"
  );
  assert_eq!(
    cov,
    16_777_216.0 / 16_777_217.0,
    "coverage must be the exact ratio, not a ratio of rounded operands"
  );
}

/// The same class one domain wider, and this is the one that survived widening:
/// `f64` holds every integer only to `2^53`, so a window of `2^53 + 1` casts to
/// the very same `f64` as the `2^53`-element tail inside it and the quotient is
/// exactly `1.0` again.
///
/// The defect was never "`f32` rounds too early". It is that **both operands are
/// rounded independently before the division**, and no wider domain removes
/// that — it only moves the first geometry that shows it. Reached through the
/// public planner in a single allocation (`input_len` is a count, not a slice
/// length, and this plan is one span), so nothing about it needs a hand-built
/// `Span`. 64-bit only: where `usize` is 32 bits every count is exact in `f64`
/// and the regime does not exist.
#[test]
#[cfg(target_pointer_width = "64")]
fn coverage_past_f64_integer_range_is_below_one_for_a_ragged_tail() {
  let len = 1_usize << 53;
  let window = len + 1;
  assert_eq!(
    len as f64, window as f64,
    "the premise: both operands cast to one f64"
  );

  let s = WindowPlan::spans(&WindowOptions::new(window), len).unwrap();
  assert_eq!(s.len(), 1);
  assert_eq!((s[0].len(), s[0].window()), (len, window));

  let cov = s[0].coverage();
  assert!(
    cov < 1.0,
    "a tail one element short of a 2^53 + 1 window must not report full coverage, got {cov:?}"
  );
  assert_eq!(
    cov,
    f64::from_bits(0x3fef_ffff_ffff_ffff),
    "coverage must be the correctly rounded 2^53 / (2^53 + 1)"
  );
}

/// The integer path against an oracle: `Fraction(len, window)` rounded to the
/// nearest `f64` by an implementation with unbounded rationals (CPython), for ten
/// geometries whose window is past `2^53` and whose ratios span `2^-64` to within
/// two ulps of one.
///
/// The table is the whole point: nothing in this crate produced these bits, so
/// agreeing with them is evidence rather than a restatement.
#[test]
#[cfg(target_pointer_width = "64")]
fn coverage_past_f64_integer_range_is_correctly_rounded() {
  const ORACLE: [(usize, usize, u64); 10] = [
    (1, usize::MAX, 0x3bf0_0000_0000_0000),
    (1, (1 << 63) + 1, 0x3c00_0000_0000_0000),
    (3, (1 << 54) + 1, 0x3ca8_0000_0000_0000),
    (7, 1 << 60, 0x3c5c_0000_0000_0000),
    (usize::MAX / 3, usize::MAX, 0x3fd5_5555_5555_5555),
    ((1 << 53) - 1, (1 << 54) + 1, 0x3fdf_ffff_ffff_ffff),
    ((1 << 62) + 12_345, (1 << 63) - 7, 0x3fe0_0000_0000_000c),
    (1 << 55, (1 << 55) + 1024, 0x3fef_ffff_ffff_ff00),
    ((1 << 53) + 1, (1 << 53) + 3, 0x3fef_ffff_ffff_fffe),
    (1 << 53, (1 << 53) + 1, 0x3fef_ffff_ffff_ffff),
  ];
  for (len, window, bits) in ORACLE {
    let got = Span::new(0, len, window).coverage();
    assert_eq!(
      got.to_bits(),
      bits,
      "{len}/{window}: got {got:?} ({:#018x}), oracle {:?}",
      got.to_bits(),
      f64::from_bits(bits)
    );
  }
}

/// The integer path against the `f64` division it replaces, over a sweep.
///
/// A ratio is unchanged by scaling both counts by a power of two, so
/// `ratio_to_f64(l << k, w << k)` must equal `ratio_to_f64(l, w)` — and for `w`
/// small the second is the exact-operand `f64` division, which IEEE already
/// requires to be correctly rounded. That makes this a cross-check of the two
/// paths against each other over 2485 geometries rather than a table of pinned
/// constants.
///
/// The shift decides which path the scaled pair takes, and `56` is the one that
/// matters: `window << 56` is past `2^53` for every window here, so every
/// geometry is checked on the integer path at least once (the test counts them
/// and asserts the count). `10` keeps the pair on the fast path, checking that a
/// ratio is scale-free there too, and `50` straddles the two.
///
/// The saturation regime is reached at no shift. Its test,
/// `(w - l) * 2^54 <= w`, has both sides scaled by `2^k` and so is invariant
/// under the shift — it reduces to the unshifted geometry, where a deficit of at
/// least `1` against a window of at most `70` cannot satisfy it.
#[test]
#[cfg(target_pointer_width = "64")]
fn the_integer_path_agrees_with_exact_operand_division() {
  let (mut checked, mut on_the_integer_path) = (0_u32, 0_u32);
  for window in 1_usize..=70 {
    for len in 1..=window {
      let direct = ratio_to_f64(len, window);
      assert_eq!(
        direct,
        len as f64 / window as f64,
        "{len}/{window} must take the exact-operand fast path"
      );
      for k in [10, 50, 56] {
        let (l, w) = (len << k, window << k);
        let scaled = ratio_to_f64(l, w);
        assert_eq!(
          scaled, direct,
          "{len}/{window} scaled by 2^{k} changed the ratio: {scaled:?} vs {direct:?}"
        );
        if w > 1 << 53 {
          on_the_integer_path += 1;
        }
      }
      checked += 1;
    }
  }
  assert_eq!(
    checked, 2485,
    "the sweep must cover every len <= window <= 70"
  );
  assert!(
    on_the_integer_path >= checked,
    "every geometry must reach the integer path at least once, got \
     {on_the_integer_path} integer-path checks over {checked} geometries"
  );
}

/// `coverage() == 1.0` if and only if `len == window`, including where the true
/// ratio is inside half an ulp of one and would round there.
///
/// `2^54 - 1` real elements in a window of `2^54` is the exact midpoint
/// `1 - 2^-54`, which ties to `1.0` — and `2^64 - 2` in `2^64 - 1` is far past
/// it. Both saturate down to the largest `f64` below one instead, which is what
/// keeps a ragged tail distinguishable from a full window at every geometry
/// rather than only at the ones `f64` resolves. The under-report is at most one
/// ulp and never in the direction of claiming coverage the span does not have.
#[test]
#[cfg(target_pointer_width = "64")]
fn a_ragged_span_never_reports_full_coverage() {
  for (len, window) in [
    ((1_usize << 54) - 1, 1_usize << 54),
    ((1 << 63) - 1, 1 << 63),
    (1 << 63, (1 << 63) + 1),
    (usize::MAX - 1, usize::MAX),
  ] {
    let cov = Span::new(0, len, window).coverage();
    assert_eq!(
      cov, NEAREST_BELOW_ONE,
      "{len}/{window} must saturate below one, got {cov:?}"
    );
    assert!(cov < 1.0);
  }

  // And the equivalence in the other direction, at the same windows: only a full
  // span reports `1.0`.
  for window in [1_usize, 4, (1 << 54) - 1, 1 << 54, usize::MAX] {
    assert_eq!(Span::new(0, window, window).coverage(), 1.0);
  }
}

/// The tie-breaking rule, which only an exactly-halfway ratio can observe.
///
/// A tie needs the window to divide the scaled real length exactly *and* the
/// quotient's dropped bits to be `100…0`; both happen, and the rule is round to
/// nearest with ties to **even**, the same rule IEEE division follows on the fast
/// path. `2^54 - 3` and `2^54 - 5` real elements in a window of `2^54` are the
/// pair that pins it from both sides: their exact ratios are the two midpoints
/// either side of `1 - 2^-52`, so one must round *down* to it and the other *up*,
/// and any rule that always breaks a tie the same way gets one of them wrong.
#[test]
#[cfg(target_pointer_width = "64")]
fn coverage_breaks_an_exact_tie_to_even() {
  for (len, window, bits) in [
    ((1_usize << 54) - 3, 1_usize << 54, 0x3fef_ffff_ffff_fffe),
    ((1 << 54) - 5, 1 << 54, 0x3fef_ffff_ffff_fffe),
    (14_001_415_880_023_897, 1 << 54, 0x3fe8_df1b_55f0_cfac),
  ] {
    let got = Span::new(0, len, window).coverage();
    assert_eq!(
      got.to_bits(),
      bits,
      "{len}/{window} is an exact tie and must round to even: got {got:?} ({:#018x})",
      got.to_bits()
    );
  }
}

#[test]
#[should_panic(expected = "0 < len <= window")]
fn span_new_rejects_len_above_window_in_every_build() {
  let _ = Span::new(0, 2, 1);
}

#[test]
#[should_panic(expected = "0 < len <= window")]
fn span_new_rejects_zero_len_in_every_build() {
  let _ = Span::new(0, 0, 4);
}

#[test]
#[should_panic(expected = "representable start + len")]
fn span_new_rejects_an_unrepresentable_end_in_every_build() {
  let _ = Span::new(usize::MAX, 1, 1);
}

#[test]
fn span_try_new_reports_the_same_invariant_as_a_typed_error() {
  for (start, len, window) in [(0, 2, 1), (0, 0, 4), (usize::MAX, 1, 1)] {
    assert_eq!(
      Span::try_new(start, len, window),
      Err(WinditError::InvalidSpan { start, len, window })
    );
  }

  let span = Span::try_new(8, 3, 4).unwrap();
  assert_eq!((span.start(), span.len(), span.window()), (8, 3, 4));
}

#[test]
fn span_end_is_exact_at_the_usize_boundary() {
  // The largest constructible span: its end is exactly `usize::MAX`, so `end`
  // neither wraps nor saturates away a real element.
  let span = Span::try_new(usize::MAX - 1, 1, 1).unwrap();
  assert_eq!(span.end(), usize::MAX);
  assert_eq!(Span::new(8, 3, 4).end(), 11);
}

#[cfg(feature = "serde")]
#[test]
fn window_options_serde_round_trip() {
  // Every field set to a non-default value, including the tuple tail variant and
  // the window cap, so the round trip covers the whole geometry.
  let opts = WindowOptions::new(512)
    .with_overlap(64)
    .with_tail(TailPolicy::DropBelowMin(10))
    .with_max_windows(8);
  let json = serde_json::to_string(&opts).unwrap();
  let back: WindowOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(opts, back);

  // The default tail and absent cap (a `None`) round-trip as well.
  let simple = WindowOptions::new(4);
  let back: WindowOptions = serde_json::from_str(&serde_json::to_string(&simple).unwrap()).unwrap();
  assert_eq!(simple, back);
}

#[cfg(feature = "serde")]
#[test]
fn window_options_serde_optional_cap_and_rejects_unknown_keys() {
  // `max_windows` may be omitted; it then defaults to no cap.
  let opts: WindowOptions =
    serde_json::from_str(r#"{"window":4,"hop":4,"tail":{"kind":"keep_with_coverage"}}"#).unwrap();
  assert_eq!(opts.max_windows(), None);

  // A required (non-optional) field is still enforced.
  assert!(serde_json::from_str::<WindowOptions>(r#"{"window":4,"hop":4}"#).is_err());

  // An unknown key (e.g. a typo'd `max_window`) is rejected by
  // `deny_unknown_fields` rather than silently ignored.
  assert!(serde_json::from_str::<WindowOptions>(
    r#"{"window":4,"hop":4,"tail":{"kind":"keep_with_coverage"},"max_window":2}"#
  )
  .is_err());
}

#[test]
fn tail_policy_default_is_keep_with_coverage() {
  // Pinned on its own, independent of serde: whichever variant `#[default]`
  // marks is a semantic choice (the planner keeps a ragged tail rather than
  // dropping or padding it), and this regresses if it silently moves.
  assert_eq!(TailPolicy::default(), TailPolicy::KeepWithCoverage);
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_round_trips_through_toml_keep_with_coverage() {
  let original = TailPolicy::KeepWithCoverage;
  let doc = toml::to_string(&original).unwrap();
  assert_eq!(toml::from_str::<TailPolicy>(&doc).unwrap(), original);
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_round_trips_through_toml_drop_below_min() {
  let original = TailPolicy::DropBelowMin(10);
  let doc = toml::to_string(&original).unwrap();
  assert_eq!(toml::from_str::<TailPolicy>(&doc).unwrap(), original);
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_round_trips_through_toml_pad_full() {
  let original = TailPolicy::PadFull;
  let doc = toml::to_string(&original).unwrap();
  assert_eq!(toml::from_str::<TailPolicy>(&doc).unwrap(), original);
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_toml_document_form_is_pinned() {
  // The adjacently tagged wire form, pinned exactly, the same discipline
  // `aggregate::tests::kind_serde_wire_format_is_pinned` applies to
  // `AggregatePolicyKind`: a renamed variant, a retagged field, or a payload
  // moved out of `value` would all fail here rather than silently reaching a
  // downstream document as something else.
  //
  // Internally tagged (`#[serde(tag = "kind")]` alone, no `content`) was
  // rejected for this enum specifically: it only covers struct- and map-shaped
  // payloads, and serializing `DropBelowMin`'s bare `usize` under it fails at
  // run time with "cannot serialize tagged newtype variant
  // TailPolicy::DropBelowMin containing an integer" — proven against a
  // throwaway probe crate before this form was chosen, not assumed.
  assert_eq!(
    toml::to_string(&TailPolicy::KeepWithCoverage).unwrap(),
    "kind = \"keep_with_coverage\"\n"
  );
  assert_eq!(
    toml::to_string(&TailPolicy::DropBelowMin(2)).unwrap(),
    "kind = \"drop_below_min\"\nvalue = 2\n"
  );
  assert_eq!(
    toml::to_string(&TailPolicy::PadFull).unwrap(),
    "kind = \"pad_full\"\n"
  );
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_json_document_form_is_pinned() {
  // The same pin as `tail_policy_toml_document_form_is_pinned`, in the other
  // human-readable format the crate is configured through, and pinned here
  // because the `Deserialize` side is now hand-written: the human-readable
  // branch has to keep reading exactly the document the derive read, byte for
  // byte, or a stored configuration silently stops loading.
  assert_eq!(
    serde_json::to_string(&TailPolicy::KeepWithCoverage).unwrap(),
    r#"{"kind":"keep_with_coverage"}"#
  );
  assert_eq!(
    serde_json::to_string(&TailPolicy::DropBelowMin(2)).unwrap(),
    r#"{"kind":"drop_below_min","value":2}"#
  );
  assert_eq!(
    serde_json::to_string(&TailPolicy::PadFull).unwrap(),
    r#"{"kind":"pad_full"}"#
  );

  // And read back from those exact literals, not merely from what `to_string`
  // just produced: a serializer and a hand-written deserializer that drifted
  // together would still round-trip.
  assert_eq!(
    serde_json::from_str::<TailPolicy>(r#"{"kind":"keep_with_coverage"}"#).unwrap(),
    TailPolicy::KeepWithCoverage
  );
  assert_eq!(
    serde_json::from_str::<TailPolicy>(r#"{"kind":"drop_below_min","value":2}"#).unwrap(),
    TailPolicy::DropBelowMin(2)
  );
  assert_eq!(
    serde_json::from_str::<TailPolicy>(r#"{"kind":"pad_full"}"#).unwrap(),
    TailPolicy::PadFull
  );
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_round_trips_through_postcard() {
  // The half no human-readable format can exercise. Postcard is
  // non-self-describing: it refuses `deserialize_any` with `WontImplement`, and
  // reading an adjacent tag needs exactly that, so every variant of this type
  // failed to deserialize under it before `Deserialize` learned to branch on
  // `is_human_readable` — while `to_allocvec` had been writing bytes all along.
  for policy in [
    TailPolicy::KeepWithCoverage,
    TailPolicy::DropBelowMin(7),
    TailPolicy::PadFull,
  ] {
    let bytes = postcard::to_allocvec(&policy).unwrap();
    assert_eq!(
      postcard::from_bytes::<TailPolicy>(&bytes).unwrap(),
      policy,
      "postcard round trip for {policy:?}"
    );
  }

  // The compact bytes, pinned. They are the externally tagged form — a variant
  // index and then the payload — and no branch on the `Serialize` side produces
  // them: serde writes an adjacent tag through `serialize_unit_variant`, which
  // a format that does not name its variants renders as that index. Pinned so
  // that if this ever stopped being true the diagnosis is here rather than in a
  // round trip that merely stopped closing.
  assert_eq!(
    postcard::to_allocvec(&TailPolicy::KeepWithCoverage).unwrap(),
    vec![0x00]
  );
  assert_eq!(
    postcard::to_allocvec(&TailPolicy::DropBelowMin(7)).unwrap(),
    vec![0x01, 0x07]
  );
  assert_eq!(
    postcard::to_allocvec(&TailPolicy::PadFull).unwrap(),
    vec![0x02]
  );
}

/// A `Serializer` that records the type identities serde is asked to write, and
/// serializes nothing.
///
/// Every data format available here — `serde_json`, `toml`, `postcard`,
/// `ciborium` — discards the name an enum or struct is serialized under, so no
/// round trip can observe it. A schema-aware or name-preserving format does not
/// discard it, which makes that name part of the wire model and a silent break
/// if it changes. Being the serializer is the only way to see it.
#[cfg(feature = "serde")]
mod name_recorder {
  use core::fmt;
  use std::{
    string::{String, ToString},
    vec::Vec,
  };

  use serde::{
    ser::{Impossible, SerializeStruct},
    Serialize, Serializer,
  };

  /// Everything this serializer is not asked to do by a `TailPolicy`.
  #[derive(Debug)]
  pub struct Unsupported(String);

  impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str(&self.0)
    }
  }

  impl core::error::Error for Unsupported {}

  impl serde::ser::Error for Unsupported {
    fn custom<T: fmt::Display>(msg: T) -> Self {
      Self(msg.to_string())
    }
  }

  /// The identities seen, in the order serde asked for them.
  #[derive(Default)]
  pub struct Names {
    /// `(name, field count)` per `serialize_struct`.
    pub structs: Vec<(String, usize)>,
    /// `(enum name, variant index, variant name)` per `serialize_unit_variant`.
    pub unit_variants: Vec<(String, u32, String)>,
  }

  /// Serialize `value` for its names alone.
  pub fn record<T: Serialize>(value: &T) -> Names {
    let mut names = Names::default();
    value
      .serialize(Recorder { names: &mut names })
      .expect("a TailPolicy serializes as a struct holding a unit variant and at most a usize");
    names
  }

  struct Recorder<'a> {
    names: &'a mut Names,
  }

  macro_rules! refuse {
    ($($method:ident($($arg:ty),*);)*) => {
      $(
        fn $method(self, $(_: $arg),*) -> Result<(), Unsupported> {
          Err(Unsupported(::core::stringify!($method).to_string()))
        }
      )*
    };
  }

  impl Serializer for Recorder<'_> {
    type Ok = ();
    type Error = Unsupported;
    type SerializeSeq = Impossible<(), Unsupported>;
    type SerializeTuple = Impossible<(), Unsupported>;
    type SerializeTupleStruct = Impossible<(), Unsupported>;
    type SerializeTupleVariant = Impossible<(), Unsupported>;
    type SerializeMap = Impossible<(), Unsupported>;
    type SerializeStruct = Self;
    type SerializeStructVariant = Impossible<(), Unsupported>;

    refuse! {
      serialize_bool(bool);
      serialize_i8(i8);
      serialize_i16(i16);
      serialize_i32(i32);
      serialize_i64(i64);
      serialize_u8(u8);
      serialize_u16(u16);
      serialize_u32(u32);
      serialize_f32(f32);
      serialize_f64(f64);
      serialize_char(char);
      serialize_str(&str);
      serialize_bytes(&[u8]);
      serialize_unit_struct(&'static str);
    }

    /// The one scalar a `TailPolicy` carries: `DropBelowMin`'s minimum, a
    /// `usize`, which serde routes here.
    fn serialize_u64(self, _: u64) -> Result<(), Unsupported> {
      Ok(())
    }

    fn serialize_none(self) -> Result<(), Unsupported> {
      Ok(())
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), Unsupported> {
      value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), Unsupported> {
      Ok(())
    }

    fn serialize_unit_variant(
      self,
      name: &'static str,
      index: u32,
      variant: &'static str,
    ) -> Result<(), Unsupported> {
      self
        .names
        .unit_variants
        .push((name.to_string(), index, variant.to_string()));
      Ok(())
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
      self,
      _: &'static str,
      value: &T,
    ) -> Result<(), Unsupported> {
      value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
      self,
      _: &'static str,
      _: u32,
      _: &'static str,
      _: &T,
    ) -> Result<(), Unsupported> {
      Err(Unsupported("serialize_newtype_variant".to_string()))
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Unsupported> {
      Err(Unsupported("serialize_seq".to_string()))
    }

    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Unsupported> {
      Err(Unsupported("serialize_tuple".to_string()))
    }

    fn serialize_tuple_struct(
      self,
      _: &'static str,
      _: usize,
    ) -> Result<Self::SerializeTupleStruct, Unsupported> {
      Err(Unsupported("serialize_tuple_struct".to_string()))
    }

    fn serialize_tuple_variant(
      self,
      _: &'static str,
      _: u32,
      _: &'static str,
      _: usize,
    ) -> Result<Self::SerializeTupleVariant, Unsupported> {
      Err(Unsupported("serialize_tuple_variant".to_string()))
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Unsupported> {
      Err(Unsupported("serialize_map".to_string()))
    }

    fn serialize_struct(
      self,
      name: &'static str,
      len: usize,
    ) -> Result<Self::SerializeStruct, Unsupported> {
      self.names.structs.push((name.to_string(), len));
      Ok(self)
    }

    fn serialize_struct_variant(
      self,
      _: &'static str,
      _: u32,
      _: &'static str,
      _: usize,
    ) -> Result<Self::SerializeStructVariant, Unsupported> {
      Err(Unsupported("serialize_struct_variant".to_string()))
    }
  }

  impl SerializeStruct for Recorder<'_> {
    type Ok = ();
    type Error = Unsupported;

    fn serialize_field<T: ?Sized + Serialize>(
      &mut self,
      _: &'static str,
      value: &T,
    ) -> Result<(), Unsupported> {
      value.serialize(Recorder { names: self.names })
    }

    fn end(self) -> Result<(), Unsupported> {
      Ok(())
    }
  }
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_is_written_under_its_own_name() {
  use std::string::String;

  // The wire model includes the identities serde is asked to write, and the
  // private `Tag` that carries `kind` must not leak its own. `TailPolicy` is
  // what the derived adjacent representation passed to `serialize_unit_variant`
  // before this rewrite, and a format that keeps enum names would see any other
  // answer as a different type. No round trip here can catch it: every dev
  // dependency's format discards the name.
  for (policy, index, variant, fields) in [
    (
      TailPolicy::KeepWithCoverage,
      0u32,
      "keep_with_coverage",
      1usize,
    ),
    (TailPolicy::DropBelowMin(5), 1, "drop_below_min", 2),
    (TailPolicy::PadFull, 2, "pad_full", 1),
  ] {
    let names = name_recorder::record(&policy);
    assert_eq!(
      names.structs,
      std::vec![(String::from("TailPolicy"), fields)],
      "the document's own name and field count for {policy:?}"
    );
    assert_eq!(
      names.unit_variants,
      std::vec![(String::from("TailPolicy"), index, String::from(variant))],
      "the `kind` tag's enum name, index and variant name for {policy:?}"
    );
  }
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_round_trips_through_a_self_describing_binary_format() {
  // The row that separates "not human-readable" from "not self-describing".
  // CBOR reports `is_human_readable() == false` and still writes `kind` and
  // `value` by name, so a reader that switched on that flag would hand it the
  // compact shape and fail — the defect this test exists to keep out. It is also
  // the format the previous derived reader handled through `deserialize_any`,
  // so it must keep working, not merely start working.
  for policy in [
    TailPolicy::KeepWithCoverage,
    TailPolicy::DropBelowMin(7),
    TailPolicy::PadFull,
  ] {
    let mut bytes = Vec::new();
    ciborium::into_writer(&policy, &mut bytes).unwrap();
    assert_eq!(
      ciborium::from_reader::<TailPolicy, _>(bytes.as_slice()).unwrap(),
      policy,
      "CBOR round trip for {policy:?}"
    );
  }

  // And the fields really are named there, not indexed: a two-entry map for the
  // variant with a payload, a one-entry map for one without.
  let mut bytes = Vec::new();
  ciborium::into_writer(&TailPolicy::DropBelowMin(7), &mut bytes).unwrap();
  let value: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
  let map = value.as_map().unwrap();
  let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_text().unwrap()).collect();
  assert_eq!(keys, vec!["kind", "value"]);
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_reads_a_document_whatever_order_its_keys_arrive_in() {
  // A map's order is the format's business, not the document's, so the reader
  // collects both keys before deciding the variant.
  assert_eq!(
    serde_json::from_str::<TailPolicy>(r#"{"value":5,"kind":"drop_below_min"}"#).unwrap(),
    TailPolicy::DropBelowMin(5)
  );

  // The explicit null serde's derived reader tolerated for a payload-free
  // variant is still accepted, though nothing here writes it.
  assert_eq!(
    serde_json::from_str::<TailPolicy>(r#"{"kind":"pad_full","value":null}"#).unwrap(),
    TailPolicy::PadFull
  );
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_rejects_malformed_documents() {
  // A minimum is required by the variant that carries one...
  assert!(serde_json::from_str::<TailPolicy>(r#"{"kind":"drop_below_min"}"#).is_err());
  // ...and refused to the variants that do not.
  assert!(serde_json::from_str::<TailPolicy>(r#"{"kind":"pad_full","value":5}"#).is_err());
  // A repeated key, an unknown variant, and a missing tag are all errors.
  assert!(serde_json::from_str::<TailPolicy>(r#"{"kind":"pad_full","kind":"pad_full"}"#).is_err());
  assert!(serde_json::from_str::<TailPolicy>(r#"{"kind":"drop_everything"}"#).is_err());
  assert!(serde_json::from_str::<TailPolicy>(r#"{"value":5}"#).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_refuses_an_entry_that_is_neither_field() {
  use std::string::ToString;

  // An entry that is neither the tag nor the content is refused by name, as
  // `WindowOptions` refuses an unknown key. The document that made this the
  // rule put the cap inside the tail policy: skipped, it left `max_windows`
  // unbounded without a word.
  let misplaced =
    r#"{"window":512,"hop":512,"tail":{"kind":"keep_with_coverage","max_windows":1}}"#;
  let refused = serde_json::from_str::<WindowOptions>(misplaced).unwrap_err();
  assert!(refused.to_string().contains("max_windows"), "{refused}");
  // Whatever the stray entry holds, and on whichever side of the fields that
  // matter it sits.
  for (document, key) in [
    (r#"{"kind":"pad_full","future":1}"#, "future"),
    (
      r#"{"kind":"drop_below_min","value":5,"$schema":"x"}"#,
      "$schema",
    ),
    (
      r#"{"note":{"a":[1,2,null]},"kind":"drop_below_min","value":5}"#,
      "note",
    ),
  ] {
    let refused = serde_json::from_str::<TailPolicy>(document).unwrap_err();
    assert!(refused.to_string().contains(key), "{document}: {refused}");
  }
}

/// Map and sequence accesses built by hand, for the two grammars no
/// dev-dependency's format produces: a map whose keys arrive as byte strings,
/// and a sequence that carries the trailing unit the derived reader always
/// consumed for a payload-free variant.
#[cfg(feature = "serde")]
mod hand_built {
  use core::cell::RefCell;
  use std::{
    string::{String, ToString},
    vec::Vec,
  };

  use serde::de::{
    value::{
      Error, MapAccessDeserializer, SeqAccessDeserializer, StrDeserializer, U32Deserializer,
      U64Deserializer, UnitDeserializer,
    },
    DeserializeSeed, EnumAccess, Error as _, MapAccess, SeqAccess, VariantAccess, Visitor,
  };
  use serde::{forward_to_deserialize_any, Deserialize, Deserializer};

  use super::TailPolicy;

  /// One element's value, in the shapes a `TailPolicy` document uses.
  ///
  /// The text borrows rather than being `'static`, so a name produced at run
  /// time — the variant name a recording serializer just observed — can be fed
  /// straight back without being leaked to buy it a `'static` lifetime.
  #[derive(Clone, Copy)]
  pub enum Entry<'a> {
    Text(&'a str),
    Number(u64),
    Unit,
    /// A variant named by its ordinal rather than by its name — what a compact
    /// format writes for a tag.
    Variant(u32),
  }

  impl Entry<'_> {
    fn feed<'de, T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, Error> {
      match self {
        Self::Text(text) => seed.deserialize(StrDeserializer::new(text)),
        Self::Number(number) => seed.deserialize(U64Deserializer::new(number)),
        Self::Unit => seed.deserialize(UnitDeserializer::new()),
        Self::Variant(index) => seed.deserialize(VariantIndex(index)),
      }
    }
  }

  /// A deserializer that answers `deserialize_enum` with the unit variant at an
  /// ordinal, the way a non-self-describing format does. Serde's own integer
  /// value deserializers forward `deserialize_enum` to `deserialize_any` and so
  /// cannot stand in for one.
  struct VariantIndex(u32);

  impl<'de> Deserializer<'de> for VariantIndex {
    type Error = Error;

    fn deserialize_enum<V: Visitor<'de>>(
      self,
      _: &'static str,
      _: &'static [&'static str],
      visitor: V,
    ) -> Result<V::Value, Error> {
      visitor.visit_enum(self)
    }

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
      visitor.visit_u32(self.0)
    }

    /// A compact format's answer, which is the format this stands in for.
    fn is_human_readable(&self) -> bool {
      false
    }

    forward_to_deserialize_any! {
      bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf
      option unit unit_struct newtype_struct seq tuple tuple_struct map struct
      identifier ignored_any
    }
  }

  impl<'de> EnumAccess<'de> for VariantIndex {
    type Error = Error;
    type Variant = UnitOnly;

    fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<(V::Value, UnitOnly), Error> {
      Ok((seed.deserialize(U32Deserializer::new(self.0))?, UnitOnly))
    }
  }

  /// The tag carries no payload of its own; the content is the next element.
  pub struct UnitOnly;

  impl<'de> VariantAccess<'de> for UnitOnly {
    type Error = Error;

    fn unit_variant(self) -> Result<(), Error> {
      Ok(())
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, _: T) -> Result<T::Value, Error> {
      Err(Error::custom("a tag carries no payload of its own"))
    }

    fn tuple_variant<V: Visitor<'de>>(self, _: usize, _: V) -> Result<V::Value, Error> {
      Err(Error::custom("a tag carries no payload of its own"))
    }

    fn struct_variant<V: Visitor<'de>>(
      self,
      _: &'static [&'static str],
      _: V,
    ) -> Result<V::Value, Error> {
      Err(Error::custom("a tag carries no payload of its own"))
    }
  }

  /// How a map hands a field name over.
  #[derive(Clone, Copy)]
  pub enum Key {
    Text(&'static str),
    Bytes(&'static [u8]),
  }

  impl Key {
    fn feed<'de, T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, Error> {
      match self {
        Self::Text(text) => seed.deserialize(StrDeserializer::new(text)),
        Self::Bytes(bytes) => seed.deserialize(Bytes(bytes)),
      }
    }
  }

  /// A deserializer that offers its value to `visit_bytes` and nothing else —
  /// the byte-form field identifier a CBOR map with byte-string keys produces.
  struct Bytes(&'static [u8]);

  impl<'de> Deserializer<'de> for Bytes {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
      visitor.visit_bytes(self.0)
    }

    forward_to_deserialize_any! {
      bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf
      option unit unit_struct newtype_struct seq tuple tuple_struct map struct
      enum identifier ignored_any
    }
  }

  struct Keyed<'a> {
    entries: &'a [(Key, Entry<'a>)],
    at: usize,
  }

  impl<'de> MapAccess<'de> for Keyed<'_> {
    type Error = Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
      &mut self,
      seed: K,
    ) -> Result<Option<K::Value>, Error> {
      match self.entries.get(self.at) {
        Some((key, _)) => key.feed(seed).map(Some),
        None => Ok(None),
      }
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, Error> {
      let (_, entry) = self.entries[self.at];
      self.at += 1;
      entry.feed(seed)
    }
  }

  struct Framed<'a> {
    entries: &'a [Entry<'a>],
    at: usize,
  }

  impl<'de> SeqAccess<'de> for Framed<'_> {
    type Error = Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(
      &mut self,
      seed: T,
    ) -> Result<Option<T::Value>, Error> {
      match self.entries.get(self.at) {
        Some(entry) => {
          self.at += 1;
          entry.feed(seed).map(Some)
        }
        None => Ok(None),
      }
    }
  }

  /// Read a `TailPolicy` from a map built entry by entry.
  ///
  /// Every value is offered through one of serde's primitive value
  /// deserializers, which forward `deserialize_option` to `deserialize_any` and
  /// so refuse a bare integer asked for as an option. That is the property
  /// default RON has, and the reason this map access — rather than a `ron`
  /// dev-dependency — is what pins it: `ron` is std-only and would be the first
  /// dev-dependency here that is not `alloc`-clean, and the property, not the
  /// format, is what the reader has to survive.
  pub fn from_map<'a>(entries: &'a [(Key, Entry<'a>)]) -> Result<TailPolicy, Error> {
    TailPolicy::deserialize(MapAccessDeserializer::new(Keyed { entries, at: 0 }))
  }

  /// Read a `TailPolicy` from a sequence of elements.
  pub fn from_sequence<'a>(entries: &'a [Entry<'a>]) -> Result<TailPolicy, Error> {
    TailPolicy::deserialize(SeqAccessDeserializer::new(Framed { entries, at: 0 }))
  }

  /// The type names a deserializer was asked for, in order.
  #[derive(Default)]
  pub struct Requested {
    /// Names passed to `deserialize_struct`.
    pub structs: Vec<String>,
    /// Names passed to `deserialize_enum`.
    pub enums: Vec<String>,
  }

  /// Answer an adjacently tagged document while recording every type name the
  /// reader asks for, as a human-readable format or as one that is not.
  ///
  /// The identity a reader requests is part of the wire model — a schema-aware
  /// or name-enforcing format matches on it — and no format here carries it, so
  /// being the deserializer is the only way to see it. It is also where the
  /// names can diverge invisibly: serde's adjacent-tag derive passes the
  /// *renamed* type to `deserialize_struct` and the enum's **Rust identifier**
  /// to the nested `deserialize_enum`, so a helper that is renamed rather than
  /// named answers under two identities at once.
  pub fn record_requests(
    variant: &'static str,
    minimum: Option<u64>,
    human_readable: bool,
  ) -> (Result<TailPolicy, Error>, Requested) {
    let seen = RefCell::new(Requested::default());
    let policy = TailPolicy::deserialize(Recorder {
      variant,
      minimum,
      human_readable,
      seen: &seen,
    });
    (policy, seen.into_inner())
  }

  #[derive(Clone, Copy)]
  struct Recorder<'a> {
    variant: &'static str,
    minimum: Option<u64>,
    /// What the document and its `kind` answer `is_human_readable` with.
    human_readable: bool,
    seen: &'a RefCell<Requested>,
  }

  impl<'de> Deserializer<'de> for Recorder<'_> {
    type Error = Error;

    fn deserialize_struct<V: Visitor<'de>>(
      self,
      name: &'static str,
      _: &'static [&'static str],
      visitor: V,
    ) -> Result<V::Value, Error> {
      self.seen.borrow_mut().structs.push(name.to_string());
      visitor.visit_map(RecordedMap { at: 0, of: self })
    }

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
      visitor.visit_map(RecordedMap { at: 0, of: self })
    }

    fn is_human_readable(&self) -> bool {
      self.human_readable
    }

    forward_to_deserialize_any! {
      bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf
      option unit unit_struct newtype_struct seq tuple tuple_struct map enum
      identifier ignored_any
    }
  }

  /// The document `Recorder` answers with: `kind`, then `value` if one was asked
  /// for.
  struct RecordedMap<'a> {
    at: usize,
    of: Recorder<'a>,
  }

  impl<'de> MapAccess<'de> for RecordedMap<'_> {
    type Error = Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
      &mut self,
      seed: K,
    ) -> Result<Option<K::Value>, Error> {
      match (self.at, self.of.minimum) {
        (0, _) => seed.deserialize(StrDeserializer::new("kind")).map(Some),
        (1, Some(_)) => seed.deserialize(StrDeserializer::new("value")).map(Some),
        _ => Ok(None),
      }
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, Error> {
      let at = self.at;
      self.at += 1;
      match at {
        0 => seed.deserialize(RecordedTag(self.of)),
        _ => seed.deserialize(U64Deserializer::new(self.of.minimum.unwrap_or_default())),
      }
    }
  }

  /// The `kind` value: a variant name, and the record of the enum identity the
  /// reader asked it for.
  struct RecordedTag<'a>(Recorder<'a>);

  impl<'de> Deserializer<'de> for RecordedTag<'_> {
    type Error = Error;

    fn deserialize_enum<V: Visitor<'de>>(
      self,
      name: &'static str,
      variants: &'static [&'static str],
      visitor: V,
    ) -> Result<V::Value, Error> {
      self.0.seen.borrow_mut().enums.push(name.to_string());
      StrDeserializer::new(self.0.variant).deserialize_enum(name, variants, visitor)
    }

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
      visitor.visit_str(self.0.variant)
    }

    fn is_human_readable(&self) -> bool {
      self.0.human_readable
    }

    forward_to_deserialize_any! {
      bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf
      option unit unit_struct newtype_struct seq tuple tuple_struct map struct
      identifier ignored_any
    }
  }
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_is_read_under_its_own_name() {
  // The reading half of `tail_policy_is_written_under_its_own_name`, and the
  // half that drifted: serde's adjacent-tag derive passes the *renamed* type to
  // `deserialize_struct` but the helper enum's **Rust identifier** to the nested
  // `deserialize_enum`, so a `#[serde(rename = "TailPolicy")]` helper answered
  // under two identities at once and the map road asked for the wrong one. The
  // wire enum is now *named* `TailPolicy` inside a private module instead of
  // renamed, which is what makes both requests agree; nothing but being the
  // deserializer can observe that.
  //
  // A format that is not human-readable is asked for the `kind` as that enum.
  // A human-readable one is asked for the `kind` as the value it is, so that a
  // refusal can name the variant, and so for no enum at all; its document is
  // asked for under the same one name.
  use hand_built::record_requests;
  use std::string::String;

  for (variant, minimum, expected) in [
    ("keep_with_coverage", None, TailPolicy::KeepWithCoverage),
    ("drop_below_min", Some(5), TailPolicy::DropBelowMin(5)),
    ("pad_full", None, TailPolicy::PadFull),
  ] {
    let (policy, seen) = record_requests(variant, minimum, false);
    assert_eq!(policy.unwrap(), expected);
    assert_eq!(
      seen.structs,
      std::vec![String::from("TailPolicy")],
      "the document's own name for {variant}"
    );
    assert_eq!(
      seen.enums,
      std::vec![String::from("TailPolicy")],
      "the enum identity the `kind` value was asked for, for {variant}"
    );

    let (policy, seen) = record_requests(variant, minimum, true);
    assert_eq!(policy.unwrap(), expected);
    assert_eq!(
      seen.structs,
      std::vec![String::from("TailPolicy")],
      "the document's own name for {variant}, human-readable"
    );
    assert!(
      seen.enums.is_empty(),
      "a human-readable `kind` is read as the value it is, for {variant}: {:?}",
      seen.enums
    );
  }
}

#[cfg(feature = "serde")]
#[test]
fn every_road_agrees_on_every_variant_name_and_ordinal() {
  // The wire is declared once and generated from that declaration, and this is
  // what holds the generation honest end to end. For every variant it takes the
  // name and the ordinal the *serializer* actually wrote — observed by being the
  // serializer, since no format here carries them — and feeds each back through
  // both reader roads:
  //
  //   the derived writer's variant name  ->  sequence road, tag as a name
  //                                      ->  map road, tag as a name
  //   the derived writer's variant index ->  sequence road, tag as an index
  //
  // A roster that disagreed with the derive, a sequence arm bound to the wrong
  // ordinal, or a renamed variant that reached one road and not the other would
  // all fail here rather than in a downstream document.
  use hand_built::{from_map, from_sequence, Entry, Key};

  for (policy, payload) in [
    (TailPolicy::KeepWithCoverage, None),
    (TailPolicy::DropBelowMin(5), Some(5u64)),
    (TailPolicy::PadFull, None),
  ] {
    let names = name_recorder::record(&policy);
    let (enum_name, index, variant) = names.unit_variants[0].clone();
    assert_eq!(
      enum_name, "TailPolicy",
      "the tag's enum identity for {policy:?}"
    );

    // The sequence road, tag by name and tag by ordinal.
    let by_name = match payload {
      Some(min) => std::vec![Entry::Text(&variant), Entry::Number(min)],
      None => std::vec![Entry::Text(&variant)],
    };
    assert_eq!(
      from_sequence(&by_name).unwrap_or_else(|e| panic!("{variant} by name: {e}")),
      policy
    );
    let by_index = match payload {
      Some(min) => std::vec![Entry::Variant(index), Entry::Number(min)],
      None => std::vec![Entry::Variant(index)],
    };
    assert_eq!(
      from_sequence(&by_index).unwrap_or_else(|e| panic!("{variant} by index {index}: {e}")),
      policy
    );

    // The map road, with the same name the serializer wrote.
    let entries = match payload {
      Some(min) => std::vec![
        (Key::Text("kind"), Entry::Text(&variant)),
        (Key::Text("value"), Entry::Number(min)),
      ],
      None => std::vec![(Key::Text("kind"), Entry::Text(&variant))],
    };
    assert_eq!(
      from_map(&entries).unwrap_or_else(|e| panic!("{variant} as a map: {e}")),
      policy
    );
  }
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_names_an_unknown_byte_key() {
  use ciborium::Value;
  use std::string::ToString;

  // A CBOR tail map whose stray key is a byte string: the refusal names it,
  // as text where it is UTF-8 and as hex where it is not.
  let read = |key: Value| {
    let map = Value::Map(vec![
      (
        Value::Text("kind".into()),
        Value::Text("keep_with_coverage".into()),
      ),
      (key, Value::Integer(1.into())),
    ]);
    let mut bytes = Vec::new();
    ciborium::into_writer(&map, &mut bytes).unwrap();
    ciborium::from_reader::<TailPolicy, _>(bytes.as_slice())
      .unwrap_err()
      .to_string()
  };
  let named = read(Value::Bytes(b"max_windows".to_vec()));
  assert!(named.contains("max_windows"), "{named}");
  let hex = read(Value::Bytes(vec![0xff, 0xfe]));
  assert!(hex.contains("0xfffe"), "{hex}");
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_reads_a_map_whose_keys_arrive_as_bytes() {
  use std::string::ToString;

  // The derived reader recognises byte-form field identifiers, which a CBOR map
  // with byte-string keys produces. Delegating the map to that reader is what
  // makes this hold; the test is here because no dev-dependency's serializer
  // emits them.
  use hand_built::{from_map, Entry, Key};

  assert_eq!(
    from_map(&[(Key::Bytes(b"kind"), Entry::Text("pad_full"))]).unwrap(),
    TailPolicy::PadFull
  );
  assert_eq!(
    from_map(&[
      (Key::Bytes(b"kind"), Entry::Text("drop_below_min")),
      (Key::Bytes(b"value"), Entry::Number(5)),
    ])
    .unwrap(),
    TailPolicy::DropBelowMin(5)
  );
  // A byte key that is neither field is refused exactly as a string one is,
  // by its name.
  let refused = from_map(&[
    (Key::Bytes(b"future"), Entry::Number(1)),
    (Key::Bytes(b"kind"), Entry::Text("keep_with_coverage")),
  ])
  .unwrap_err();
  assert!(refused.to_string().contains("future"), "{refused}");
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_reads_a_map_without_an_option_grammar() {
  // The regression this pins: asking a format for the minimum as an *option*
  // rejects the bare integer this crate writes, because a format is free to
  // require `Some(..)`/`None` syntax for one — default RON does, and serde's own
  // value deserializers behave the same way by forwarding `deserialize_option`
  // to `deserialize_any`. The values below go through those deserializers, so a
  // reader that reached for an option would fail here.
  //
  // Both key orders are covered, because they take different roads through the
  // derived reader: with the tag first it asks for the payload's own type, and
  // with the content first it buffers and re-reads. Neither asks for an option.
  use hand_built::{from_map, Entry, Key};

  assert_eq!(
    from_map(&[
      (Key::Text("kind"), Entry::Text("drop_below_min")),
      (Key::Text("value"), Entry::Number(5)),
    ])
    .unwrap(),
    TailPolicy::DropBelowMin(5)
  );
  assert_eq!(
    from_map(&[
      (Key::Text("value"), Entry::Number(5)),
      (Key::Text("kind"), Entry::Text("drop_below_min")),
    ])
    .unwrap(),
    TailPolicy::DropBelowMin(5)
  );

  // A payload-free variant, with the unit content the derived reader accepts
  // either side of the tag.
  assert_eq!(
    from_map(&[(Key::Text("kind"), Entry::Text("pad_full"))]).unwrap(),
    TailPolicy::PadFull
  );
  assert_eq!(
    from_map(&[
      (Key::Text("value"), Entry::Unit),
      (Key::Text("kind"), Entry::Text("pad_full")),
    ])
    .unwrap(),
    TailPolicy::PadFull
  );
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_reads_both_sequence_grammars() {
  // A payload-free variant is written as a one-element sequence — that is what a
  // compact format emits and what its round trip depends on — while the derived
  // reader always consumed a second element and accepted a unit there. Both are
  // read, so neither a document this crate writes nor one the previous release
  // wrote is refused.
  use hand_built::{from_sequence, Entry};

  for (variant, policy) in [
    ("keep_with_coverage", TailPolicy::KeepWithCoverage),
    ("pad_full", TailPolicy::PadFull),
  ] {
    assert_eq!(
      from_sequence(&[Entry::Text(variant)]).unwrap_or_else(|e| panic!("{variant} alone: {e}")),
      policy
    );
  }
  assert_eq!(
    from_sequence(&[Entry::Text("keep_with_coverage"), Entry::Unit]).unwrap(),
    TailPolicy::KeepWithCoverage
  );
  assert_eq!(
    from_sequence(&[Entry::Text("pad_full"), Entry::Unit]).unwrap(),
    TailPolicy::PadFull
  );

  // A payload-free variant still refuses a real content element, and the
  // refusal names what it expected instead. `ToString` is named rather than
  // assumed: without the `std` feature this crate is `no_std`, and the core
  // prelude does not carry it.
  use std::string::ToString;

  let error = from_sequence(&[Entry::Text("pad_full"), Entry::Number(5)]).unwrap_err();
  assert!(error.to_string().contains("value"), "{error}");

  // And the variant that carries a minimum still requires it.
  assert_eq!(
    from_sequence(&[Entry::Text("drop_below_min"), Entry::Number(5)]).unwrap(),
    TailPolicy::DropBelowMin(5)
  );
  assert!(from_sequence(&[Entry::Text("drop_below_min")]).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn window_options_round_trips_through_postcard() {
  // The motivating shape again, now on the compact side: a `TailPolicy` reached
  // as a field of the geometry document rather than as a bare value. The field
  // deserializer is postcard's, so the format branch has to hold one level down
  // as well.
  let opts = WindowOptions::new(512)
    .with_overlap(64)
    .with_tail(TailPolicy::DropBelowMin(10))
    .with_max_windows(8);
  let bytes = postcard::to_allocvec(&opts).unwrap();
  assert_eq!(postcard::from_bytes::<WindowOptions>(&bytes).unwrap(), opts);

  let simple = WindowOptions::new(4);
  let bytes = postcard::to_allocvec(&simple).unwrap();
  assert_eq!(
    postcard::from_bytes::<WindowOptions>(&bytes).unwrap(),
    simple
  );
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_round_trips_as_a_window_options_document_field() {
  // The motivating scenario: a `TailPolicy` reached not as a bare value but as
  // the `tail` field of a `WindowOptions` document — the shape a `mediagraph`
  // node-options TOML file actually carries.
  let opts = WindowOptions::new(4).with_tail(TailPolicy::DropBelowMin(2));
  let doc = toml::to_string(&opts).unwrap();
  assert_eq!(toml::from_str::<WindowOptions>(&doc).unwrap(), opts);
}

#[test]
fn tail_policy_display_is_pinned() {
  use std::string::ToString;

  // The exact strings a downstream derivation fingerprint persists —
  // changing any one of them is a breaking change, not a patch.
  assert_eq!(
    TailPolicy::KeepWithCoverage.to_string(),
    "keep_with_coverage"
  );
  assert_eq!(TailPolicy::DropBelowMin(3).to_string(), "drop_below_min(3)");
  assert_eq!(TailPolicy::DropBelowMin(0).to_string(), "drop_below_min(0)");
  assert_eq!(TailPolicy::PadFull.to_string(), "pad_full");
}

#[test]
fn window_options_display_is_pinned() {
  use std::string::ToString;

  // The motivating example: `window`, `hop`, and a `DropBelowMin` tail nested
  // through its own `Display` — the fingerprint is the document spelling, so
  // `hop` prints under its own field name.
  let opts = WindowOptions::new(512)
    .with_hop(256)
    .with_tail(TailPolicy::DropBelowMin(3));
  assert_eq!(
    opts.to_string(),
    "window=512,hop=256,tail=drop_below_min(3),max_windows=none"
  );

  // The default geometry (`hop == window`, `KeepWithCoverage`, no cap): all
  // four segments still appear, `max_windows` folding to `none`.
  assert_eq!(
    WindowOptions::new(4).to_string(),
    "window=4,hop=4,tail=keep_with_coverage,max_windows=none"
  );

  // A configured cap prints its value in place of `none`, in the same
  // position.
  assert_eq!(
    WindowOptions::new(4).with_max_windows(8).to_string(),
    "window=4,hop=4,tail=keep_with_coverage,max_windows=8"
  );
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_refuses_a_trailing_element_in_toml() {
  use std::string::ToString;

  // TOML's arrays do not check their own end, so the sequence road holds
  // itself to the tag and the value: a third element is refused by count,
  // and the two-element form still reads.
  #[derive(Debug, serde::Deserialize)]
  struct Document {
    tail: TailPolicy,
  }

  let refused = toml::from_str::<Document>(r#"tail = ["drop_below_min", 5, 99]"#)
    .unwrap_err()
    .to_string();
  assert!(refused.contains("2 elements expected"), "{refused}");
  assert_eq!(
    toml::from_str::<Document>(r#"tail = ["drop_below_min", 5]"#)
      .unwrap()
      .tail,
    TailPolicy::DropBelowMin(5)
  );
}

#[cfg(feature = "serde")]
#[test]
fn window_options_refuse_a_trailing_element_in_toml() {
  use std::string::ToString;

  // The same for the options' own sequence form: four elements, and a fifth
  // is refused by count. A compact format's sequence still reads.
  #[derive(Debug, serde::Deserialize)]
  struct Document {
    #[allow(dead_code)]
    options: WindowOptions,
  }

  let refused =
    toml::from_str::<Document>(r#"options = [8, 4, {kind = "keep_with_coverage"}, 100, 999]"#)
      .unwrap_err()
      .to_string();
  assert!(refused.contains("4 elements expected"), "{refused}");
  let opts = WindowOptions::new(8).with_hop(4).with_max_windows(100);
  let bytes = postcard::to_allocvec(&opts).unwrap();
  assert_eq!(postcard::from_bytes::<WindowOptions>(&bytes).unwrap(), opts);
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_refuses_a_kind_written_as_a_map() {
  use std::string::ToString;

  // A `kind` is read by the rule `AggregatePolicyKind`'s unit variants are:
  // its name, or a YAML tag on no value. A one-entry map is refused by its
  // variant's name, whatever it holds and whatever stands beside it, in every
  // human-readable format.
  for (format, refused) in [
    (
      "JSON null",
      serde_json::from_str::<TailPolicy>(r#"{"kind":{"pad_full":null}}"#)
        .map_err(|e| e.to_string()),
    ),
    (
      "JSON with a key beside",
      serde_json::from_str::<TailPolicy>(r#"{"kind":{"pad_full":null,"typo":0}}"#)
        .map_err(|e| e.to_string()),
    ),
    (
      "TOML empty table",
      toml::from_str::<TailPolicy>("kind = { pad_full = {} }").map_err(|e| e.to_string()),
    ),
    (
      "TOML empty array",
      toml::from_str::<TailPolicy>("kind = { pad_full = [] }").map_err(|e| e.to_string()),
    ),
    (
      "YAML null",
      serde_yaml_ng::from_str::<TailPolicy>("kind: {pad_full: ~}").map_err(|e| e.to_string()),
    ),
  ] {
    let refused = refused.unwrap_err();
    assert!(
      refused.contains("`pad_full` takes no value, and the document gives it one"),
      "{format}: {refused}"
    );
  }

  // The name reads, in each of them.
  assert_eq!(
    serde_json::from_str::<TailPolicy>(r#"{"kind":"pad_full"}"#).unwrap(),
    TailPolicy::PadFull
  );
  assert_eq!(
    toml::from_str::<TailPolicy>(r#"kind = "pad_full""#).unwrap(),
    TailPolicy::PadFull
  );
  assert_eq!(
    serde_yaml_ng::from_str::<TailPolicy>("kind: pad_full").unwrap(),
    TailPolicy::PadFull
  );
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_names_its_kind_in_a_refused_yaml_tag() {
  use std::string::ToString;

  // A tagged `kind` whose value is refused names the variant, as a tagged
  // `AggregatePolicyKind` does, and YAML locates the refusal at the tagged
  // value itself: the reader that refuses it is the one YAML asked.
  let refused = serde_yaml_ng::from_str::<TailPolicy>("kind: !pad_full 7").unwrap_err();
  let at = refused.location().expect("a located refusal");
  assert_eq!((at.line(), at.column()), (1, 7), "{refused}");
  assert!(refused.to_string().contains("`pad_full`"), "{refused}");

  // A tag on no value reads.
  assert_eq!(
    serde_yaml_ng::from_str::<TailPolicy>("kind: !pad_full").unwrap(),
    TailPolicy::PadFull
  );
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_names_its_kind_on_the_sequence_road() {
  use std::string::ToString;

  // The sequence road reads its tag by the same rule as the map road: a
  // one-entry map is refused by its variant's name, and a name reads.
  #[derive(Debug, serde::Deserialize)]
  struct Document {
    tail: TailPolicy,
  }

  let refused = serde_json::from_str::<TailPolicy>(r#"[{"pad_full":null,"typo":0}]"#)
    .unwrap_err()
    .to_string();
  assert!(refused.contains("`pad_full` takes no value"), "{refused}");
  let refused = toml::from_str::<Document>("tail = [{ pad_full = {} }]")
    .unwrap_err()
    .to_string();
  assert!(refused.contains("`pad_full` takes no value"), "{refused}");
  assert_eq!(
    serde_json::from_str::<TailPolicy>(r#"["pad_full"]"#).unwrap(),
    TailPolicy::PadFull
  );
  assert_eq!(
    toml::from_str::<Document>(r#"tail = ["pad_full"]"#)
      .unwrap()
      .tail,
    TailPolicy::PadFull
  );
}

/// The 1-based column at which serde_json reports a refused `token` in `doc`:
/// the token's last character.
#[cfg(feature = "serde")]
fn json_column(doc: &str, token: &str) -> usize {
  doc.find(token).expect("the token is in the document") + token.len()
}

/// The byte range TOML marks for `token` in `doc`.
#[cfg(feature = "serde")]
fn toml_span(doc: &str, token: &str) -> core::ops::Range<usize> {
  let at = doc.find(token).expect("the token is in the document");
  at..at + token.len()
}

#[cfg(feature = "serde")]
#[test]
fn refusals_are_named_and_located_in_json() {
  use std::string::ToString;

  // A refused value is named by the reader JSON asked for it, in its expected
  // text, and JSON places the refusal at the value: the line and column of
  // its last character.
  for (doc, token, named) in [
    (
      r#"{"window":8,"hop":"4","tail":{"kind":"pad_full"}}"#,
      r#""4""#,
      "expected `hop`, an unsigned integer",
    ),
    (
      r#"{"window":"8","hop":4,"tail":{"kind":"pad_full"}}"#,
      r#""8""#,
      "expected `window`, an unsigned integer",
    ),
    (
      r#"{"kind":"drop_below_min","value":"5"}"#,
      r#""5""#,
      "expected `drop_below_min`'s `value`",
    ),
    (
      r#"["drop_below_min","5"]"#,
      r#""5""#,
      "expected `drop_below_min`'s `value`",
    ),
    (
      r#"{"window":8,"hop":4,"tail":{"kind":"drop_below_min","value":"5"}}"#,
      r#""5""#,
      "expected `drop_below_min`'s `value`",
    ),
    (
      r#"["8",4,{"kind":"pad_full"},null]"#,
      r#""8""#,
      "expected `window`, an unsigned integer",
    ),
  ] {
    let refused = if doc.contains("window") || doc.starts_with(r#"["8""#) {
      serde_json::from_str::<WindowOptions>(doc).unwrap_err()
    } else {
      serde_json::from_str::<TailPolicy>(doc).unwrap_err()
    };
    assert!(refused.to_string().contains(named), "{doc}: {refused}");
    assert_eq!(
      (refused.line(), refused.column()),
      (1, json_column(doc, token)),
      "{doc}: {refused}"
    );
  }
}

#[cfg(feature = "serde")]
#[test]
fn refusals_are_named_and_located_in_toml() {
  // TOML marks a refused value with its own span, an array's element
  // included, because the refusal is made by the reader TOML asked for that
  // value.
  #[derive(Debug, serde::Deserialize)]
  struct Tail {
    #[allow(dead_code)]
    tail: TailPolicy,
  }
  #[derive(Debug, serde::Deserialize)]
  struct Options {
    #[allow(dead_code)]
    options: WindowOptions,
  }

  let doc = r#"tail = ["drop_below_min", "5"]"#;
  let refused = toml::from_str::<Tail>(doc).unwrap_err();
  assert_eq!(refused.span(), Some(toml_span(doc, r#""5""#)), "{refused}");
  assert!(
    refused.message().contains("`drop_below_min`'s `value`"),
    "{refused}"
  );

  let doc = r#"options = ["8", 4, { kind = "pad_full" }]"#;
  let refused = toml::from_str::<Options>(doc).unwrap_err();
  assert_eq!(refused.span(), Some(toml_span(doc, r#""8""#)), "{refused}");
  assert!(refused.message().contains("`window`"), "{refused}");

  let doc = "kind = \"drop_below_min\"\nvalue = \"5\"";
  let refused = toml::from_str::<TailPolicy>(doc).unwrap_err();
  assert_eq!(refused.span(), Some(toml_span(doc, r#""5""#)), "{refused}");
  assert!(
    refused.message().contains("`drop_below_min`'s `value`"),
    "{refused}"
  );

  let doc = "window = 8\nhop = 4\n[tail]\nkind = \"drop_below_min\"\nvalue = \"5\"";
  let refused = toml::from_str::<WindowOptions>(doc).unwrap_err();
  assert_eq!(refused.span(), Some(toml_span(doc, r#""5""#)), "{refused}");
  assert!(
    refused.message().contains("`drop_below_min`'s `value`"),
    "{refused}"
  );
}

#[cfg(feature = "serde")]
#[test]
fn refusals_are_named_and_located_in_yaml() {
  use std::string::ToString;

  // YAML's `location()` is the refused value's own, because the refusal is
  // made by the reader YAML asked for that value.
  for (doc, (line, column), named) in [
    (
      "window: \"8\"\nhop: 4\ntail: {kind: pad_full}",
      (1, 9),
      "expected `window`, an unsigned integer",
    ),
    (
      "kind: drop_below_min\nvalue: \"5\"",
      (2, 8),
      "expected `drop_below_min`'s `value`",
    ),
  ] {
    let refused = if doc.starts_with("window") {
      serde_yaml_ng::from_str::<WindowOptions>(doc).unwrap_err()
    } else {
      serde_yaml_ng::from_str::<TailPolicy>(doc).unwrap_err()
    };
    let at = refused.location().expect("a located refusal");
    assert_eq!((at.line(), at.column()), (line, column), "{doc}: {refused}");
    assert!(refused.to_string().contains(named), "{doc}: {refused}");
  }
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_names_a_held_value() {
  use std::string::ToString;

  // A `value` before its `kind` is held by the derived reader and read again
  // once the `kind` is known, through the same reader as ever, so its refusal
  // is named too. What it is read from holds no location, so JSON places it
  // where it was read: the map's end.
  let doc = r#"{"value":"5","kind":"drop_below_min"}"#;
  let refused = serde_json::from_str::<TailPolicy>(doc).unwrap_err();
  assert!(
    refused
      .to_string()
      .contains("expected `drop_below_min`'s `value`"),
    "{refused}"
  );
  assert_eq!(
    (refused.line(), refused.column()),
    (1, doc.len()),
    "{refused}"
  );

  // A value given to a unit variant is refused by the derived reader, which
  // names the variant by its Rust name, on either side of the `kind`.
  for doc in [
    r#"{"kind":"pad_full","value":5}"#,
    r#"{"value":5,"kind":"pad_full"}"#,
  ] {
    let refused = serde_json::from_str::<TailPolicy>(doc)
      .unwrap_err()
      .to_string();
    assert!(
      refused.contains("unit variant TailPolicy::PadFull"),
      "{doc}: {refused}"
    );
  }
}

#[cfg(feature = "serde")]
#[test]
fn tail_policy_names_a_kind_that_is_a_number() {
  use std::string::ToString;

  // A number for a `kind` is refused naming the `kind`: as itself, and as the
  // map `arbitrary_precision` hands a number over as.
  for doc in [
    r#"{"kind":0.5}"#,
    r#"{"kind":{"$serde_json::private::Number":"0.5"}}"#,
  ] {
    let refused = serde_json::from_str::<TailPolicy>(doc)
      .unwrap_err()
      .to_string();
    assert!(
      refused.contains("expected a `kind` naming one of"),
      "{doc}: {refused}"
    );
  }

  // A known limit: a number JSON cannot hold is refused by JSON itself, before
  // any reader of ours is asked, as "number out of range".
  assert!(serde_json::from_str::<TailPolicy>(r#"{"kind":1e400}"#).is_err());
}
