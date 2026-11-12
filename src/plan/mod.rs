//! Window geometry: configurable options, tail handling, spans, and the planner.
//!
//! A `WindowPlan` turns an input length plus a [`WindowOptions`] into a list of
//! [`Span`]s — plain `usize` element counts, unit-agnostic across samples,
//! tokens, patches, and frames. The same spans drive pre-processing (slice /
//! pad / mask) and post-processing (aggregate / smooth / segment).

use crate::error::WinditError;

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests;

/// A single window's placement over the input: where it starts, how many real
/// elements it covers, and the fixed window size it pads to.
///
/// [`len`](Span::len) is the number of *real* input elements (`0 < len <=
/// window`); the remaining `window - len` positions are padding.
/// [`coverage`](Span::coverage) reports the real fraction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Span {
  start: usize,
  len: usize,
  window: usize,
}

// A span always covers at least one real element, so an `is_empty` companion to
// `len` would be a constant `false` — a misleading addition rather than the
// useful pair the lint assumes.
#[allow(clippy::len_without_is_empty)]
impl Span {
  /// A span starting at `start` that covers `len` real elements of a
  /// `window`-wide window, the remaining `window - len` positions being padding.
  ///
  /// The infallible counterpart to [`try_new`](Span::try_new), for the callers
  /// that know their geometry — a literal in a test, a plan the caller just
  /// validated — and would only unwrap.
  ///
  /// # Panics
  ///
  /// Panics, in every build, unless `0 < len <= window` and `start + len` is
  /// representable as a `usize`. Use [`try_new`](Span::try_new) to handle an
  /// untrusted geometry instead.
  #[must_use]
  pub const fn new(start: usize, len: usize, window: usize) -> Self {
    match Self::try_new(start, len, window) {
      Ok(span) => span,
      Err(_) => panic!("a span must satisfy 0 < len <= window with a representable start + len"),
    }
  }

  /// The checked counterpart of [`new`](Span::new): validate the geometry rather
  /// than panic on it.
  ///
  /// # Errors
  ///
  /// Returns [`WinditError::InvalidSpan`] unless `0 < len <= window` and
  /// `start + len` is representable as a `usize`. Those three conditions are the
  /// span invariant every other method on this type relies on, so they are
  /// enforced identically in debug and release.
  pub const fn try_new(start: usize, len: usize, window: usize) -> Result<Self, WinditError> {
    // `len <= window` with a non-zero `len` also rules out a zero window, so
    // `coverage` divides by a positive denominator; the checked addition is what
    // keeps `end` exact for every span that exists.
    if len == 0 || len > window || start.checked_add(len).is_none() {
      return Err(WinditError::InvalidSpan { start, len, window });
    }
    Ok(Self { start, len, window })
  }

  /// Index of the first real element, in input elements.
  #[must_use]
  pub const fn start(&self) -> usize {
    self.start
  }

  /// Number of real elements in the window (`0 < len <= window`).
  #[must_use]
  pub const fn len(&self) -> usize {
    self.len
  }

  /// One past the last real element the span covers (`start + len`), in input
  /// elements.
  ///
  /// The exclusive end of the real (unpadded) region, in the same element units
  /// as the segmentation ranges — and the single place this crate performs the
  /// `start + len` addition.
  #[must_use]
  pub const fn end(&self) -> usize {
    // Both constructors reject a span whose `start + len` is not representable,
    // in every build, so the saturation is unreachable. It is kept as the last
    // line of defence for the "never wraps" contract this method's callers rely
    // on: `usize::MAX` is out of bounds for any slice and so is caught by the
    // length checks downstream, whereas a wrap would alias a low index and
    // silently select the wrong elements.
    self.start.saturating_add(self.len)
  }

  /// The fixed window size this span pads to, in elements.
  #[must_use]
  pub const fn window(&self) -> usize {
    self.window
  }

  /// The fraction of the window filled by real elements, always in `(0, 1]`.
  ///
  /// Both constructors enforce `0 < len <= window` in every build, so the
  /// denominator is positive and the quotient is finite.
  ///
  /// # The fraction is a weight, so it is `f64`
  ///
  /// This is not a report a caller merely reads: it is what the
  /// `CoverageWeightedMean` aggregation policy multiplies an embedding by, in
  /// the `f64` domain every shipped scalar computes in (named rather than
  /// linked — that policy sits behind `alloc` and this tier is featureless). A
  /// weight resolved more coarsely than the arithmetic it drives loses two
  /// distinct things, and this fraction lost both while it was `f32`:
  ///
  /// - **The operands rounded before the division.** `f32` represents every
  ///   integer only to `2^24`; `f64` to `2^53`. A window of `2^24 + 1` narrowed
  ///   to `2^24`, so a tail one element short of it divided out to exactly `1.0`
  ///   and a ragged tail was indistinguishable from a full window.
  /// - **The quotient was rounded to the `f32` grid.** That grid is `2^-24`
  ///   apart relatively where the fold rounds at `2^-53`, so two window
  ///   geometries whose true coverages differ by less than an `f32` ulp — as
  ///   `8388607/16777213` and `8388608/16777215` do, by `3.6e-15` — arrived at
  ///   an `f64` fold as one weight.
  ///
  /// Both were read as a defect of `f32`, and only the second one was. Widening
  /// the quotient to `f64` did fix the second. The first is not about width at
  /// all: it is that **both operands were rounded independently before the
  /// division**, and `f64` only moves the first geometry that shows it out to
  /// `2^53`. `Span::new(0, 2^53, 2^53 + 1)` — one span of
  /// `WindowPlan::spans(&WindowOptions::new(2^53 + 1), 2^53)` — casts both counts
  /// to the same `f64` and divided out to exactly `1.0`, a ragged tail wearing a
  /// full window's coverage in the crate that documents ragged tails as strictly
  /// below one.
  ///
  /// # The ratio, and where it saturates
  ///
  /// So the division is not performed on rounded operands. This returns the
  /// **correctly rounded** value of the exact rational `len / window` for every
  /// geometry a [`Span`] can hold, with one deliberate exception at the top of
  /// the range:
  ///
  /// `coverage() == 1.0` **if and only if** `len == window`.
  ///
  /// A true ratio within half an ulp of one — `window - len` below
  /// `window * 2^-54`, which needs a window of at least `2^54` — would round to
  /// `1.0` and make a ragged span indistinguishable from a full one again. Those
  /// saturate *downwards* instead, to `1 - 2^-53`, the largest `f64` below one.
  /// The error that introduces is under one ulp and never overstates coverage,
  /// the mapping stays monotone in `len`, and the equivalence above holds for
  /// every geometry rather than for the ones `f64` happens to resolve. Rounding
  /// is otherwise nearest-even, so the two window geometries `8388607/16777213`
  /// and `8388608/16777215` stay `3.6e-15` apart rather than collapsing.
  ///
  /// The precision is what the weighting reads and the *scale* is not:
  /// `CoverageWeightedMean` folds each coverage divided by the largest in the
  /// plan, so what reaches its sum is the ratios between these fractions. Two
  /// geometries an ulp apart must stay an ulp apart; a plan whose coverages are
  /// uniformly small is the same plan as one whose coverages are uniformly
  /// large.
  #[must_use]
  pub fn coverage(&self) -> f64 {
    // The span invariant (`0 < len <= window`, enforced by both constructors in
    // every build) is exactly this helper's precondition.
    ratio_to_f64(self.len, self.window)
  }
}

/// The largest `f64` strictly below `1.0`, `1 - 2^-53`.
const NEAREST_BELOW_ONE: f64 = f64::from_bits(0x3fef_ffff_ffff_ffff);

/// The `f64` significand width, `53`: every integer up to `2^SIGNIFICAND_BITS`
/// is exactly an `f64`, and the next one is not.
const SIGNIFICAND_BITS: u32 = f64::MANTISSA_DIGITS;

/// The `f64` exponent bias, `1023`.
const EXPONENT_BIAS: i32 = 1023;

// `ratio_to_f64` reduces both counts through `u64` and bounds its scaled
// numerator by `2^118` on that basis. A wider `usize` would silently break both,
// so it is refused at compile time rather than mis-divided at run time.
const _: () = assert!(usize::BITS <= 64);

/// The correctly rounded value of the exact rational `numer / denom`, saturating
/// to [`NEAREST_BELOW_ONE`] whenever `numer < denom` but the true ratio rounds to
/// `1.0`.
///
/// The precondition is [`Span`]'s own invariant, `0 < numer <= denom`.
///
/// Casting each count to `f64` first is what this exists to avoid: past
/// `2^SIGNIFICAND_BITS` that cast rounds, and two counts one apart can land on
/// the same value — after which no amount of care in the division recovers the
/// distinction. The quotient is formed from the integers themselves instead.
///
/// Three regimes, in the order they are tested:
///
/// - `numer == denom`: exactly `1.0`, the only input that produces it.
/// - `denom <= 2^SIGNIFICAND_BITS`: both counts are exact `f64`s, so IEEE
///   division *is* the correctly rounded ratio and this is one instruction.
///   Always taken where `usize` is 32 bits, which is why the bare-metal tier
///   never reaches the integer path. A quotient here is at most `1 - 1/denom`,
///   itself an `f64`, so it never rounds up to `1.0`.
/// - otherwise: the saturation test, then an exact integer division. The test is
///   `deficit * 2^54 <= denom`, equivalently a true ratio at or above
///   `1 - 2^-54` — the midpoint whose tie rounds to `1.0`. Because it claims
///   everything from there up, the division below is left with a ratio strictly
///   under `1 - 2^-54`, which cannot round to `1.0` either.
fn ratio_to_f64(numer: usize, denom: usize) -> f64 {
  debug_assert!(numer > 0 && numer <= denom, "0 < numer <= denom");
  if numer == denom {
    return 1.0;
  }
  let (n, d) = (numer as u64, denom as u64);
  if d <= 1 << SIGNIFICAND_BITS {
    return n as f64 / d as f64;
  }
  if u128::from(d - n) << (SIGNIFICAND_BITS + 1) <= u128::from(d) {
    return NEAREST_BELOW_ONE;
  }
  // Scale the numerator so the quotient lands in `[2^53, 2^55)` — one or two bits
  // wider than the significand, whatever the two magnitudes are. `n < d` gives
  // `n.ilog2() <= d.ilog2()`, so the shift is between `54` and `117` and the
  // shifted numerator stays under `2^118`.
  let shift = SIGNIFICAND_BITS + 1 + d.ilog2() - n.ilog2();
  let scaled = u128::from(n) << shift;
  let (quotient, remainder) = (scaled / u128::from(d), scaled % u128::from(d));
  // Round to nearest, ties to even. `remainder` is the sticky bit: nonzero means
  // the exact value sits strictly above the midpoint the dropped bits describe,
  // so a tie is a tie only when the division came out exact.
  let extra = quotient.ilog2() + 1 - SIGNIFICAND_BITS;
  let mut kept = quotient >> extra;
  let dropped = quotient & ((1 << extra) - 1);
  let midpoint = 1 << (extra - 1);
  if dropped > midpoint || (dropped == midpoint && (remainder != 0 || kept & 1 == 1)) {
    kept += 1;
  }
  // `kept * 2^exponent` is the answer, and both factors are exact: `kept` is at
  // most `2^53` — the round-up carries at most one bit past the `[2^52, 2^53)`
  // the shift placed it in, and `2^53` is itself an `f64` — while `exponent`
  // lands between `-116` and `-53`, well inside the normal range. So no
  // renormalization is needed for the carry: `2^53 * 2^e` and `2^52 * 2^(e + 1)`
  // are the same `f64`, and a branch for it would be inert rather than defensive.
  let exponent = extra as i32 - shift as i32;
  kept as f64 * f64::from_bits(((exponent + EXPONENT_BIAS) as u64) << (SIGNIFICAND_BITS - 1))
}

/// How the planner treats a final window that does not fill a whole [`Span`].
///
/// With the `serde` feature this is **adjacently tagged**, in one shape for every
/// format: `kind` names the variant and `value` carries
/// [`DropBelowMin`](TailPolicy::DropBelowMin)'s minimum. Internal tagging
/// (`#[serde(tag = "kind")]` with no `content`) was rejected: that representation
/// covers only struct- and map-shaped payloads, and a bare `usize` is neither,
/// so it fails at serialization time for
/// [`DropBelowMin`](TailPolicy::DropBelowMin) with "cannot serialize tagged
/// newtype variant ... containing an integer". The adjacent `value` field
/// sidesteps that: it holds whatever the variant carries with no merge into a
/// shared map.
///
/// A self-describing format writes the two as named fields —
/// `{"kind":"drop_below_min","value":5}` — and a compact one writes the variant
/// and then the payload, `[01, 05]` under postcard. An entry that is neither
/// field is refused by its key, as [`WindowOptions`] refuses an unknown key: a
/// stray key beside `kind`, such as a `max_windows` misplaced there, would
/// otherwise be dropped without a word.
///
/// `Deserialize` asks for a struct, which is what lets serde route each format
/// to the road it belongs on. A human-readable format's map (JSON, YAML, TOML)
/// is read by windit alone, its keys, its `kind` and its `value` in either
/// order, so no derived reader runs there. A map in a format that is not
/// human-readable (CBOR) goes to serde's own derived adjacently tagged reader,
/// each key checked on the way, so that grammar is `0.4.0`'s but for the
/// stray entry it now refuses. A compact format's sequence goes to a small
/// reader written for it.
///
/// In a human-readable format, on either road, the `kind` is read by the one
/// rule windit holds a unit variant to, as `AggregatePolicyKind`'s are: its
/// name, or a YAML tag on no value (`kind: !pad_full`). Written as a one-entry
/// map it is refused by name, whatever its value, so JSON's
/// `{"kind":{"pad_full":null}}` and TOML's `kind = { pad_full = {} }`, which
/// `0.4.0` read, are refused, as is YAML's `kind: {pad_full: ~}`. A tagged
/// value the variant refuses names the variant. A format that is not
/// human-readable is asked for the enum, as before.
///
/// A refused value is named in the expected text of the reader windit hands
/// the format for it: `` expected `drop_below_min`'s `value` ``,
/// `` expected a `kind` naming one of … ``, `` expected no value under `pad_full` ``.
/// So the format makes the refusal itself, at the value: TOML's span, YAML's
/// `location()`, JSON's line and column. Known limits: a refusal a format makes
/// before asking any of windit's readers (a number it cannot hold, such as
/// `1e400`; a syntax error; in CBOR, a value of the wrong type, which
/// ciborium's typed readers refuse first, and a key that is neither text nor
/// bytes) names nothing of windit's. A `value` written before its `kind` is
/// read at once, as nothing or a count, and any other shape is refused there,
/// at the value, as `` expected `value` written before its `kind` ``; what the
/// variant then asks of it is checked once the `kind` is known, named, with no
/// location of its own. In CBOR, whose map goes to the derive, a value given
/// to a unit variant is refused by serde's derive, which names the variant by
/// its Rust name. Keys are read by windit's own reader: a key that is not
/// `kind` or `value`, whatever its shape, and a repeated one are refused at
/// the key, named.
///
/// The derive alone could not serve both: an adjacent tag is deserialized by
/// buffering the content, which asks for `deserialize_any`, and a format that
/// does not describe itself cannot answer — postcard refuses it outright with
/// `WontImplement`. Splitting on `is_human_readable` would not serve them
/// either, since CBOR and MessagePack report `false` while writing the named
/// fields. The struct request splits them where they actually differ: serde
/// hands it to `visit_map` or `visit_seq` by what the format encodes, not by
/// what it claims to be.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TailPolicy {
  /// Keep the ragged tail as a partial span; its [`Span::coverage`] is below
  /// `1.0`. This is the default.
  #[default]
  KeepWithCoverage,
  /// Drop a ragged tail whose real length is below the given minimum. A full
  /// window (`len == window`) is always kept.
  ///
  /// The `text` chunker `split::ContentAware::chunk` honours this variant too.
  /// A chunk has no element count of its own, so "real length" there is the
  /// chunk's measure in the caller's `MeasureText` units, and a chunk that
  /// fills the window is kept exactly as a full span is.
  DropBelowMin(usize),
  /// Keep the ragged tail; pre-processing pads it to a full window. The produced
  /// span is identical to [`KeepWithCoverage`](TailPolicy::KeepWithCoverage);
  /// the two differ only in downstream padding and weighting intent.
  ///
  /// The padding is the element pipeline's: `pre::slice_pad_mask` pads a
  /// fixed-width `Copy` element into a `window`-length buffer. Text has no such
  /// primitive — a chunk is a byte range over the caller's own `&str`, and
  /// there is no neutral text to append that the caller's measurer would count
  /// as padding — so the `text` chunker `split::ContentAware::chunk` keeps the
  /// tail unpadded here, indistinguishably from
  /// [`KeepWithCoverage`](TailPolicy::KeepWithCoverage). That is a named gap
  /// rather than an oversight: inventing characters to pad with would move the
  /// very boundaries the measurer chose.
  PadFull,
}

/// **This spelling is persisted by downstream derivation fingerprints — change
/// it only with a breaking bump.**
///
/// Unit variants print their snake_case name — `keep_with_coverage`,
/// `pad_full` — and [`DropBelowMin`](TailPolicy::DropBelowMin) prints
/// `drop_below_min(<min>)`, its minimum through `usize`'s own `Display`. These
/// are the same three names the `serde` wire form tags each variant with (see
/// below): one vocabulary, not two.
impl core::fmt::Display for TailPolicy {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Self::KeepWithCoverage => f.write_str("keep_with_coverage"),
      Self::DropBelowMin(min) => write!(f, "drop_below_min({min})"),
      Self::PadFull => f.write_str("pad_full"),
    }
  }
}

// `TailPolicy`'s wire form has exactly one declaration. The `wire_tail_policy!`
// invocation below lists the variants, their wire names, and their payloads
// once, and everything the wire needs is generated from that list: the derived
// adjacently tagged twin that writes and reads the map form, the tag-only twin
// the sequence form resolves against, the type identity, the field names, the
// variant roster, and the sequence reader's arms. Nothing about the wire can be
// respelled in one place and not another, because there is only one place.
//
// The two `From` conversions between the public enum and its twin stay
// hand-written: they are what the compiler checks for payload-shape drift, and a
// variant added to either side stops the build in them.
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
const _: () = {
  use core::fmt;

  use serde::{
    de::{
      value::{
        BorrowedBytesDeserializer, BorrowedStrDeserializer, BytesDeserializer,
        MapAccessDeserializer,
      },
      DeserializeSeed, EnumAccess, Error as _, Expected, IntoDeserializer, MapAccess, SeqAccess,
      Unexpected, VariantAccess, Visitor,
    },
    Deserialize, Deserializer, Serialize, Serializer,
  };

  /// Declare the wire form once.
  ///
  /// `tag` and `content` name the two fields; each variant line gives its Rust
  /// name, its payload if it has one, and the name it travels under. From that:
  ///
  /// - `wire::TailPolicy`, the derived adjacently tagged twin. It is *named*
  ///   `TailPolicy`, not renamed to it, because serde passes a type's Rust
  ///   identifier — not its `#[serde(rename)]` — as the enum name in the nested
  ///   `deserialize_enum` an adjacent tag seed makes, so a renamed twin answers
  ///   under two identities at once.
  /// - `wire::tag::TailPolicy`, the same variants without their payloads, named
  ///   the same way and for the same reason. The sequence reader resolves its tag
  ///   by deserializing this, so the tag's names and ordinals are the derive's,
  ///   never a second table.
  /// - `wire::NAME`, `wire::FIELDS` and `wire::VARIANTS`, each from the same
  ///   tokens the derives were given.
  /// - `from_tag`, the sequence reader's arms: one per variant, reading the
  ///   payload the variant declares and no other.
  macro_rules! wire_tail_policy {
    (
      tag: $tag:literal, content: $content:literal;
      $( $variant:ident $( ( $payload:ty ) )? => $name:literal ),+ $(,)?
    ) => {
      mod wire {
        use serde::{Deserialize, Serialize};

        /// The adjacently tagged document, derived: the reader of a map in a
        /// format that is not human-readable, handed its map through
        /// `KnownKeys`, which refuses an entry that is neither the tag nor the
        /// content by name before this reader sees it. A human-readable map is
        /// read by `human_map`, which asks no derived reader.
        #[derive(Clone, Copy, Deserialize, Serialize)]
        #[serde(tag = $tag, content = $content)]
        pub(super) enum TailPolicy {
          $( #[serde(rename = $name)] $variant $( ( $payload ) )?, )+
        }

        /// The `kind` alone, for the reader that has to resolve a tag before it
        /// knows what payload to ask for.
        pub(super) mod tag {
          use serde::Deserialize;

          #[derive(Clone, Copy, Deserialize)]
          pub enum TailPolicy {
            $( #[serde(rename = $name)] $variant, )+
          }
        }

        /// The type identity, from the same token the types above are named by.
        pub(super) const NAME: &str = stringify!(TailPolicy);
        /// The two field names, from the same tokens the derive was given.
        pub(super) const FIELDS: &[&str] = &[$tag, $content];
        /// The variant roster in wire order, from the same names.
        pub(super) const VARIANTS: &[&str] = &[$( $name ),+];
      }

      /// The policy a resolved tag names, with its payload read from `seq`.
      ///
      /// Exhaustive over the tag twin, and each arm reads exactly what its
      /// variant declares — so a variant cannot be added, renamed, or given a
      /// payload without this following it.
      fn from_tag<'de, A: SeqAccess<'de>>(
        tag: wire::tag::TailPolicy,
        seq: &mut A,
        expected: &dyn Expected,
      ) -> Result<TailPolicy, A::Error> {
        Ok(match tag {
          $(
            wire::tag::TailPolicy::$variant =>
              wire_tail_policy!(@element $variant $( ( $payload ) )?, seq, expected),
          )+
        })
      }

      /// The tag a wire name names, from the same list.
      fn tag_named(name: &str) -> Option<wire::tag::TailPolicy> {
        match name {
          $( $name => Some(wire::tag::TailPolicy::$variant), )+
          _ => None,
        }
      }

      /// A tag's wire name, from the same list.
      const fn tag_name(tag: wire::tag::TailPolicy) -> &'static str {
        match tag {
          $( wire::tag::TailPolicy::$variant => $name, )+
        }
      }
    };

    // A variant with a payload: the element after the tag is that payload, and
    // it is required.
    (@element $variant:ident ( $payload:ty ), $seq:expr, $expected:expr) => {
      TailPolicy::$variant(
        $seq
          .next_element::<$payload>()?
          .ok_or_else(|| serde::de::Error::invalid_length(1, $expected))?
          .into(),
      )
    };

    // A payload-free variant: it is *written* as a one-field struct, so a
    // compact format leaves nothing after the tag and asking unconditionally
    // would read into whatever follows in the buffer. The derived reader always
    // consumed one element and accepted a unit there, so a sequence that does
    // carry the trailing unit keeps parsing. The element is optional; present,
    // it must be the unit.
    (@element $variant:ident, $seq:expr, $expected:expr) => {{
      let _ = $expected;
      $seq.next_element::<NoContent>()?;
      TailPolicy::$variant
    }};
  }

  wire_tail_policy! {
    tag: "kind", content: "value";
    KeepWithCoverage => "keep_with_coverage",
    DropBelowMin(crate::plan::Minimum) => "drop_below_min",
    PadFull => "pad_full",
  }

  /// Into the wire form. Exhaustive over [`TailPolicy`]: a variant added to the
  /// public enum stops the build here until the wire form can express it.
  impl From<TailPolicy> for wire::TailPolicy {
    fn from(policy: TailPolicy) -> Self {
      match policy {
        TailPolicy::KeepWithCoverage => Self::KeepWithCoverage,
        TailPolicy::DropBelowMin(min) => Self::DropBelowMin(Minimum(min)),
        TailPolicy::PadFull => Self::PadFull,
      }
    }
  }

  /// Out of it. Exhaustive over the wire form, which closes the other direction:
  /// between the two, neither enum can gain a variant the other cannot carry,
  /// nor change a payload's shape alone.
  impl From<wire::TailPolicy> for TailPolicy {
    fn from(policy: wire::TailPolicy) -> Self {
      match policy {
        wire::TailPolicy::KeepWithCoverage => Self::KeepWithCoverage,
        wire::TailPolicy::DropBelowMin(Minimum(min)) => Self::DropBelowMin(min),
        wire::TailPolicy::PadFull => Self::PadFull,
      }
    }
  }

  impl Serialize for TailPolicy {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
      // The derive writes the document, so it is `0.4.0`'s document — the same
      // code that wrote it then, reached through a conversion the compiler
      // checks. Nothing here decides a field count, a tag, or a name.
      wire::TailPolicy::from(*self).serialize(serializer)
    }
  }

  impl Serialize for Minimum {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
      self.0.serialize(serializer)
    }
  }

  impl<'de> Deserialize<'de> for Minimum {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
      deserializer
        .deserialize_u64(Count::of(wire::VARIANTS[1], wire::FIELDS[1]))
        .map(Minimum)
    }
  }

  impl From<Minimum> for usize {
    fn from(Minimum(min): Minimum) -> usize {
      min
    }
  }

  /// The content element a payload-free variant may carry in a sequence.
  ///
  /// Deserializing through `deserialize_unit` is what keeps it free on a compact
  /// format — a unit is zero bytes in postcard, so a present-but-empty slot
  /// costs no read — while still refusing a real value, and the visitor's
  /// `expecting` is what names the refusal.
  struct NoContent;

  struct NoContentVisitor;

  impl Visitor<'_> for NoContentVisitor {
    type Value = NoContent;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(
        f,
        "no `{}`, which `drop_below_min` alone carries",
        wire::FIELDS[1]
      )
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<NoContent, E> {
      Ok(NoContent)
    }

    fn visit_none<E: serde::de::Error>(self) -> Result<NoContent, E> {
      Ok(NoContent)
    }
  }

  impl<'de> Deserialize<'de> for NoContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
      deserializer.deserialize_unit(NoContentVisitor)
    }
  }

  /// The reader of either road. `human_readable` is the format's own answer,
  /// taken before the struct is asked for: a human-readable map is read by
  /// windit alone, and any other by the derive.
  struct TailPolicyVisitor {
    human_readable: bool,
  }

  impl<'de> Visitor<'de> for TailPolicyVisitor {
    type Value = TailPolicy;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(
        f,
        "a tail policy: `{}` naming one of {:?}, with `{}` carrying `drop_below_min`'s minimum",
        wire::FIELDS[0],
        wire::VARIANTS,
        wire::FIELDS[1]
      )
    }

    /// The shape a self-describing format hands over.
    ///
    /// A human-readable format's map (JSON, YAML, TOML) is read by windit
    /// alone, in either order: its keys by [`Keys`](crate::keys::Keys), its
    /// `kind` by [`KindValue`], a `value` after the `kind` by the variant's own
    /// reader, and a `value` before it by [`HeldValue`]. The policy is then
    /// built here, `drop_below_min`'s minimum by [`Count`]'s rule. No derived
    /// reader is asked, so every refusal on this road is windit's own, named,
    /// and made inside the reader the format asked, at the value or the key.
    ///
    /// Any other map (CBOR among them) is handed whole to the wire enum's
    /// derived reader, as `0.4.0` handed it, through [`KnownKeys`], which
    /// refuses an entry that is neither field, naming its key.
    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<TailPolicy, A::Error> {
      if self.human_readable {
        human_map(map)
      } else {
        wire::TailPolicy::deserialize(MapAccessDeserializer::new(KnownKeys { map, seen: 0 }))
          .map(Into::into)
      }
    }

    /// The shape a format that encodes a struct as a sequence hands over: the
    /// tag, then the content the tag calls for.
    ///
    /// The tag is read through [`TagSeed`]: by [`KindValue`] in a
    /// human-readable format, and as the derive's tag twin otherwise, whose
    /// names and ordinals are the derive's own. `from_tag` reads what the
    /// resolved variant declares.
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<TailPolicy, A::Error> {
      let tag = seq
        .next_element_seed(TagSeed)?
        .ok_or_else(|| A::Error::invalid_length(0, &self))?;
      let policy = from_tag(tag, &mut seq, &self)?;
      exhausted(&mut seq, wire::FIELDS.len(), wire::NAME)?;
      Ok(policy)
    }
  }

  /// A human-readable format's map, read by windit alone.
  ///
  /// A `value` read before the `kind` is held as what it is, nothing or a
  /// count, by a reader that refuses every other shape at the value; once the
  /// `kind` is known, the held value is checked against the variant here, so
  /// a refusal of it names the variant but has no location of its own.
  fn human_map<'de, A: MapAccess<'de>>(mut map: A) -> Result<TailPolicy, A::Error> {
    let mut keys = crate::keys::Keys::new("`TailPolicy`", wire::FIELDS);
    let mut kind = None;
    let mut min = None;
    let mut held = None;
    while let Some(at) = keys.next(&mut map)? {
      if at == 0 {
        kind = Some(map.next_value_seed(TagSeed)?);
        continue;
      }
      match kind {
        Some(wire::tag::TailPolicy::DropBelowMin) => min = Some(map.next_value::<Minimum>()?.0),
        Some(tag) => map.next_value_seed(NoValueOf(tag_name(tag)))?,
        None => held = Some(map.next_value_seed(HeldValue)?),
      }
    }
    let tag = kind.ok_or_else(|| A::Error::missing_field(wire::FIELDS[0]))?;
    let name = tag_name(tag);
    Ok(match tag {
      wire::tag::TailPolicy::DropBelowMin => {
        let count = Count::of(name, wire::FIELDS[1]);
        TailPolicy::DropBelowMin(match (min, held) {
          (Some(min), _) => min,
          (None, Some(Held::Count(n))) => usize::try_from(n)
            .map_err(|_| A::Error::invalid_value(Unexpected::Unsigned(n), &count))?,
          (None, Some(Held::Nothing)) => {
            return Err(A::Error::invalid_type(Unexpected::Unit, &count));
          }
          (None, None) => return Err(A::Error::missing_field(wire::FIELDS[1])),
        })
      }
      wire::tag::TailPolicy::KeepWithCoverage => {
        no_count::<A::Error>(held, name)?;
        TailPolicy::KeepWithCoverage
      }
      wire::tag::TailPolicy::PadFull => {
        no_count::<A::Error>(held, name)?;
        TailPolicy::PadFull
      }
    })
  }

  /// A held `value` checked against a variant that carries none.
  fn no_count<E: serde::de::Error>(held: Option<Held>, name: &'static str) -> Result<(), E> {
    match held {
      Some(Held::Count(n)) => Err(E::invalid_type(Unexpected::Unsigned(n), &NoValueOf(name))),
      Some(Held::Nothing) | None => Ok(()),
    }
  }

  /// A `value` read before its `kind`: nothing, or a count.
  #[derive(Clone, Copy)]
  enum Held {
    Nothing,
    Count(u64),
  }

  /// The reader of a `value` written before its `kind`: the shapes some
  /// variant's reader would take, nothing (unit or none) or an unsigned
  /// integer, held as what they are; any other shape refused here, at the
  /// value, its expected text naming the `value`.
  struct HeldValue;

  impl<'de> DeserializeSeed<'de> for HeldValue {
    type Value = Held;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Held, D::Error> {
      deserializer.deserialize_any(self)
    }
  }

  impl Visitor<'_> for HeldValue {
    type Value = Held;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(
        f,
        "`{}` written before its `{}`: nothing, or `drop_below_min`'s unsigned integer",
        wire::FIELDS[1],
        wire::FIELDS[0]
      )
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<Held, E> {
      Ok(Held::Nothing)
    }

    fn visit_none<E: serde::de::Error>(self) -> Result<Held, E> {
      Ok(Held::Nothing)
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Held, E> {
      Ok(Held::Count(v))
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Held, E> {
      u64::try_from(v)
        .map(Held::Count)
        .map_err(|_| E::invalid_value(Unexpected::Signed(v), &self))
    }
  }

  /// The `value` of a variant that carries none, read after its `kind`:
  /// nothing, unit or none, and every other shape refused at the value, the
  /// expected text naming the variant.
  struct NoValueOf(&'static str);

  impl<'de> DeserializeSeed<'de> for NoValueOf {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
      deserializer.deserialize_any(self)
    }
  }

  impl Visitor<'_> for NoValueOf {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "no `{}`: `{}` carries none", wire::FIELDS[1], self.0)
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<(), E> {
      Ok(())
    }

    fn visit_none<E: serde::de::Error>(self) -> Result<(), E> {
      Ok(())
    }
  }

  /// The derived reader's map, for a format that is not human-readable, each
  /// key checked before the derived reader sees it: `kind` or `value` (or `0`
  /// or `1`, from a format that numbers its fields) passes through, a repeated
  /// field is refused at the key, and any other key is refused by name, a text
  /// key as itself and a byte key as its text, or as hex where it is not UTF-8.
  struct KnownKeys<A> {
    map: A,
    /// The fields already read, one bit each.
    seen: u8,
  }

  impl<'de, A: MapAccess<'de>> MapAccess<'de> for KnownKeys<A> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
      &mut self,
      seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
      self.map.next_key_seed(KnownKey {
        seed,
        seen: &mut self.seen,
      })
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, A::Error> {
      self.map.next_value_seed(seed)
    }

    fn size_hint(&self) -> Option<usize> {
      self.map.size_hint()
    }
  }

  /// One key of the derived road: checked, refused at the key where it
  /// repeats a field, then handed to the derived reader's own seed in the form
  /// it arrived in.
  struct KnownKey<'a, K> {
    seed: K,
    seen: &'a mut u8,
  }

  impl<K> KnownKey<'_, K> {
    /// The field at `at` (`0` the `kind`, `1` the `value`), refused where it
    /// was read already.
    fn first<E: serde::de::Error>(&mut self, at: usize) -> Result<(), E> {
      let bit = 1u8 << at;
      if *self.seen & bit != 0 {
        return Err(E::duplicate_field(wire::FIELDS[at]));
      }
      *self.seen |= bit;
      Ok(())
    }
  }

  impl<'de, K: DeserializeSeed<'de>> DeserializeSeed<'de> for KnownKey<'_, K> {
    type Value = K::Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<K::Value, D::Error> {
      deserializer.deserialize_identifier(self)
    }
  }

  /// A byte key that is not UTF-8, as lowercase hex.
  struct Hex<'a>(&'a [u8]);

  impl fmt::Display for Hex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
  }

  impl<K> KnownKey<'_, K> {
    /// The refusal of a byte key that is neither field, by its name.
    fn unknown_bytes<E: serde::de::Error>(bytes: &[u8]) -> E {
      match core::str::from_utf8(bytes) {
        Ok(text) => E::unknown_field(text, wire::FIELDS),
        Err(_) => E::custom(format_args!(
          "unknown field 0x{}, expected `{}` or `{}`",
          Hex(bytes),
          wire::FIELDS[0],
          wire::FIELDS[1]
        )),
      }
    }
  }

  impl<'de, K: DeserializeSeed<'de>> Visitor<'de> for KnownKey<'_, K> {
    type Value = K::Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "`{}` or `{}`", wire::FIELDS[0], wire::FIELDS[1])
    }

    fn visit_u64<E: serde::de::Error>(mut self, v: u64) -> Result<K::Value, E> {
      match usize::try_from(v)
        .ok()
        .filter(|at| *at < wire::FIELDS.len())
      {
        Some(at) => {
          self.first(at)?;
          self.seed.deserialize(v.into_deserializer())
        }
        None => Err(E::invalid_value(Unexpected::Unsigned(v), &self)),
      }
    }

    fn visit_str<E: serde::de::Error>(mut self, v: &str) -> Result<K::Value, E> {
      match wire::FIELDS.iter().position(|field| *field == v) {
        Some(at) => {
          self.first(at)?;
          self.seed.deserialize(v.into_deserializer())
        }
        None => Err(E::unknown_field(v, wire::FIELDS)),
      }
    }

    fn visit_borrowed_str<E: serde::de::Error>(mut self, v: &'de str) -> Result<K::Value, E> {
      match wire::FIELDS.iter().position(|field| *field == v) {
        Some(at) => {
          self.first(at)?;
          self.seed.deserialize(BorrowedStrDeserializer::new(v))
        }
        None => Err(E::unknown_field(v, wire::FIELDS)),
      }
    }

    fn visit_bytes<E: serde::de::Error>(mut self, v: &[u8]) -> Result<K::Value, E> {
      match wire::FIELDS.iter().position(|field| field.as_bytes() == v) {
        Some(at) => {
          self.first(at)?;
          self.seed.deserialize(BytesDeserializer::new(v))
        }
        None => Err(Self::unknown_bytes(v)),
      }
    }

    fn visit_borrowed_bytes<E: serde::de::Error>(mut self, v: &'de [u8]) -> Result<K::Value, E> {
      match wire::FIELDS.iter().position(|field| field.as_bytes() == v) {
        Some(at) => {
          self.first(at)?;
          self.seed.deserialize(BorrowedBytesDeserializer::new(v))
        }
        None => Err(Self::unknown_bytes(v)),
      }
    }
  }

  /// A `kind`, on either road: read by [`KindValue`] where the format is
  /// human-readable, by the rule `AggregatePolicyKind` holds its unit
  /// variants to (a name, or a YAML tag on no value; a one-entry map refused
  /// by its variant's name), and as the derive's tag twin otherwise, the enum
  /// asked for as the derive asks.
  struct TagSeed;

  impl<'de> DeserializeSeed<'de> for TagSeed {
    type Value = wire::tag::TailPolicy;

    fn deserialize<D: Deserializer<'de>>(
      self,
      deserializer: D,
    ) -> Result<wire::tag::TailPolicy, D::Error> {
      if deserializer.is_human_readable() {
        deserializer.deserialize_any(KindValue)
      } else {
        wire::tag::TailPolicy::deserialize(deserializer)
      }
    }
  }

  /// A `kind` as a human-readable format writes it, read to its variant.
  struct KindValue;

  impl<'de> Visitor<'de> for KindValue {
    type Value = wire::tag::TailPolicy;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(
        f,
        "a `{}` naming one of {:?}",
        wire::FIELDS[0],
        wire::VARIANTS
      )
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<wire::tag::TailPolicy, E> {
      variant_named(v)
    }

    /// A map of one entry is how a variant carrying a value is written, and
    /// no `kind` carries one: a map whose key names a variant is refused by
    /// that name, whatever its value, before any of it is read. Any other map,
    /// a number `arbitrary_precision` hands over as one among them, is refused
    /// as a map, this reader's expected text naming the `kind`.
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<wire::tag::TailPolicy, A::Error> {
      match map.next_key_seed(MaybeVariant)? {
        Some(Some(tag)) => Err(crate::refusal::unit_given_a_value(tag_name(tag))),
        Some(None) | None => Err(A::Error::invalid_type(Unexpected::Map, &self)),
      }
    }

    /// A tagged value, as YAML writes `!pad_full`: the tag names the variant,
    /// and its value is read through `UnitTag`, which takes none and names
    /// the variant when it refuses one.
    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<wire::tag::TailPolicy, A::Error> {
      let (tag, variant) = data.variant_seed(VariantName)?;
      variant.newtype_variant_seed(crate::refusal::UnitTag(tag_name(tag)))?;
      Ok(tag)
    }
  }

  /// The variant `name` names, refused by name where there is none.
  fn variant_named<E: serde::de::Error>(name: &str) -> Result<wire::tag::TailPolicy, E> {
    tag_named(name).ok_or_else(|| E::unknown_variant(name, wire::VARIANTS))
  }

  /// The key of a map a `kind` is written as: the variant it names, if any.
  /// A key that names none is not refused here, so the map it opens is
  /// refused as a map, naming the `kind`.
  struct MaybeVariant;

  impl<'de> DeserializeSeed<'de> for MaybeVariant {
    type Value = Option<wire::tag::TailPolicy>;

    fn deserialize<D: Deserializer<'de>>(
      self,
      deserializer: D,
    ) -> Result<Option<wire::tag::TailPolicy>, D::Error> {
      deserializer.deserialize_identifier(self)
    }
  }

  impl Visitor<'_> for MaybeVariant {
    type Value = Option<wire::tag::TailPolicy>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "a key of a `{}`", wire::FIELDS[0])
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Option<wire::tag::TailPolicy>, E> {
      Ok(tag_named(v))
    }

    fn visit_bytes<E: serde::de::Error>(
      self,
      v: &[u8],
    ) -> Result<Option<wire::tag::TailPolicy>, E> {
      Ok(core::str::from_utf8(v).ok().and_then(tag_named))
    }

    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Option<wire::tag::TailPolicy>, E> {
      Ok(None)
    }

    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Option<wire::tag::TailPolicy>, E> {
      Ok(None)
    }
  }

  /// A variant's name where a YAML tag gives it, `!pad_full`: an unknown one
  /// is refused by name.
  struct VariantName;

  impl<'de> DeserializeSeed<'de> for VariantName {
    type Value = wire::tag::TailPolicy;

    fn deserialize<D: Deserializer<'de>>(
      self,
      deserializer: D,
    ) -> Result<wire::tag::TailPolicy, D::Error> {
      deserializer.deserialize_identifier(self)
    }
  }

  impl Visitor<'_> for VariantName {
    type Value = wire::tag::TailPolicy;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "one of {:?}", wire::VARIANTS)
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<wire::tag::TailPolicy, E> {
      variant_named(v)
    }
  }

  impl<'de> Deserialize<'de> for TailPolicy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
      // Asking for the adjacent shape itself, rather than for whatever the value
      // happens to be, is what keeps this readable by a format that cannot
      // answer `deserialize_any`, and serde routes the request to `visit_seq`
      // or `visit_map` by what the format encodes a struct as. Whether the
      // format is human-readable is asked first, so the map road knows whose
      // it is.
      let human_readable = deserializer.is_human_readable();
      deserializer.deserialize_struct(
        wire::NAME,
        wire::FIELDS,
        TailPolicyVisitor { human_readable },
      )
    }
  }
};

/// Fully configurable window geometry: window size, hop/overlap, tail handling,
/// and an optional cap on the number of windows.
///
/// Construct with [`WindowOptions::new`] (non-overlapping windows that keep a
/// ragged tail) and refine with the `with_*` builders. Nothing is validated
/// until `WindowPlan::spans` runs — or you call
/// [`validate`](WindowOptions::validate) directly — so construction is
/// infallible.
///
/// With the `serde` feature every field is serialized, so a self-persisted
/// geometry always round-trips exactly. On deserialization the `window`, `hop`,
/// and `tail` fields are required; `max_windows` may be omitted and then
/// defaults to no cap (`None`). Unknown keys are rejected by name, and a
/// sequence form with more than its four elements is refused. A refused count
/// is refused by a reader that names its field, `` expected `window`, an
/// unsigned integer ``, so the format locates the refusal at the value; its
/// keys are read by windit's own reader, so a key that is not a field's name,
/// whatever its shape, and a repeated field are refused at the key, named. In
/// CBOR, a count of the wrong type is refused by ciborium itself, before this
/// reader is asked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct WindowOptions {
  window: usize,
  hop: usize,
  tail: TailPolicy,
  max_windows: Option<usize>,
}

/// `drop_below_min`'s minimum on the wire: written as the `usize` it is, and
/// read as one through [`Count`], whose expected text names it, so that a
/// refusal of it is made by the format at the value and names
/// `` `drop_below_min`'s `value` ``. In CBOR, where the derived reader holds a
/// `value` written before its `kind`, it reads the held value through this
/// same type; on a human-readable road a held count is checked by [`Count`]'s
/// rule once the `kind` is known.
#[cfg(feature = "serde")]
#[derive(Clone, Copy)]
struct Minimum(usize);

/// A count, read as serde reads a `usize`, through a visitor whose expected
/// text names the field it fills: `` expected `window`, an unsigned integer ``.
///
/// The format makes the refusal, at the value, and the name rides in it; no
/// error is ever rebuilt around another.
#[cfg(feature = "serde")]
#[derive(Clone, Copy)]
struct Count {
  owner: Option<&'static str>,
  field: &'static str,
}

#[cfg(feature = "serde")]
impl Count {
  /// A count that is a field of its own.
  const fn field(field: &'static str) -> Self {
    Self { owner: None, field }
  }

  /// A count that is `owner`'s `field`.
  const fn of(owner: &'static str, field: &'static str) -> Self {
    Self {
      owner: Some(owner),
      field,
    }
  }
}

#[cfg(feature = "serde")]
impl serde::de::Visitor<'_> for Count {
  type Value = usize;

  fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self.owner {
      Some(owner) => write!(f, "`{owner}`'s `{}`, an unsigned integer", self.field),
      None => write!(f, "`{}`, an unsigned integer", self.field),
    }
  }

  fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<usize, E> {
    usize::try_from(v).map_err(|_| E::invalid_value(serde::de::Unexpected::Unsigned(v), &self))
  }

  fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<usize, E> {
    usize::try_from(v).map_err(|_| E::invalid_value(serde::de::Unexpected::Signed(v), &self))
  }
}

/// Refuses an element past the `declared` ones a sequence road has read.
///
/// A format that does not check an array's end (TOML) would otherwise drop
/// it. A compact format's sequence holds exactly the declared elements, so it
/// has none to show, and nothing more is read from its buffer.
#[cfg(feature = "serde")]
fn exhausted<'de, A: serde::de::SeqAccess<'de>>(
  seq: &mut A,
  declared: usize,
  name: &str,
) -> Result<(), A::Error> {
  match seq.next_element::<serde::de::IgnoredAny>()? {
    None => Ok(()),
    Some(_) => Err(serde::de::Error::custom(format_args!(
      "`{name}`: {declared} elements expected, found more"
    ))),
  }
}

/// `Deserialize` for [`WindowOptions`]: windit's own reader, in every format,
/// of a map or of a sequence held to its four elements. No derived reader is
/// asked. Its keys are read by [`Keys`](crate::keys::Keys), so a key that is
/// not a field's name, whatever its shape, and a repeated field are refused at
/// the key, named; its counts are read through [`Count`], so a refused one
/// names its field; a tail policy names what it refuses itself.
#[cfg(feature = "serde")]
const _: () = {
  use core::fmt;

  use serde::{
    de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer,
  };

  const FIELDS: &[&str] = &["window", "hop", "tail", "max_windows"];

  /// What a sequence that is too short is refused as expecting, in the
  /// derive's words.
  const ELEMENTS: &str = "struct WindowOptions with 4 elements";

  /// A count, read through [`Count`](super::Count).
  struct CountOf(&'static str);

  impl<'de> DeserializeSeed<'de> for CountOf {
    type Value = usize;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<usize, D::Error> {
      deserializer.deserialize_u64(Count::field(self.0))
    }
  }

  /// The cap, read through serde's own `Option`, so that `null` and an absent
  /// key are no cap, and a value is a count.
  struct MaxWindows(usize);

  impl<'de> Deserialize<'de> for MaxWindows {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
      CountOf(FIELDS[3]).deserialize(deserializer).map(MaxWindows)
    }
  }

  struct OptionsVisitor;

  impl<'de> Visitor<'de> for OptionsVisitor {
    type Value = WindowOptions;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str("struct WindowOptions")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<WindowOptions, A::Error> {
      let mut keys = crate::keys::Keys::new("`WindowOptions`", FIELDS);
      let (mut window, mut hop, mut tail, mut max_windows) = (None, None, None, None);
      while let Some(at) = keys.next(&mut map)? {
        match at {
          0 => window = Some(map.next_value_seed(CountOf(FIELDS[0]))?),
          1 => hop = Some(map.next_value_seed(CountOf(FIELDS[1]))?),
          2 => tail = Some(map.next_value::<TailPolicy>()?),
          _ => max_windows = map.next_value::<Option<MaxWindows>>()?,
        }
      }
      Ok(WindowOptions {
        window: window.ok_or_else(|| A::Error::missing_field(FIELDS[0]))?,
        hop: hop.ok_or_else(|| A::Error::missing_field(FIELDS[1]))?,
        tail: tail.ok_or_else(|| A::Error::missing_field(FIELDS[2]))?,
        max_windows: max_windows.map(|MaxWindows(cap)| cap),
      })
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<WindowOptions, A::Error> {
      let window = seq
        .next_element_seed(CountOf(FIELDS[0]))?
        .ok_or_else(|| A::Error::invalid_length(0, &ELEMENTS))?;
      let hop = seq
        .next_element_seed(CountOf(FIELDS[1]))?
        .ok_or_else(|| A::Error::invalid_length(1, &ELEMENTS))?;
      let tail = seq
        .next_element::<TailPolicy>()?
        .ok_or_else(|| A::Error::invalid_length(2, &ELEMENTS))?;
      let max_windows = seq
        .next_element::<Option<MaxWindows>>()?
        .ok_or_else(|| A::Error::invalid_length(3, &ELEMENTS))?;
      exhausted(&mut seq, FIELDS.len(), "WindowOptions")?;
      Ok(WindowOptions {
        window,
        hop,
        tail,
        max_windows: max_windows.map(|MaxWindows(cap)| cap),
      })
    }
  }

  impl<'de> Deserialize<'de> for WindowOptions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
      deserializer.deserialize_struct("WindowOptions", FIELDS, OptionsVisitor)
    }
  }
};

impl WindowOptions {
  /// A geometry with the given window size, no overlap (`hop == window`),
  /// [`TailPolicy::KeepWithCoverage`], and no window cap.
  ///
  /// The window size is not validated here; a zero window is rejected by
  /// [`validate`](WindowOptions::validate) and `WindowPlan::spans`.
  #[must_use]
  pub const fn new(window: usize) -> Self {
    Self {
      window,
      hop: window,
      tail: TailPolicy::KeepWithCoverage,
      max_windows: None,
    }
  }

  /// Set the hop (stride between consecutive window starts), in elements.
  ///
  /// A hop below the window overlaps consecutive windows; a hop equal to the
  /// window tiles them. A zero hop is rejected by
  /// [`validate`](WindowOptions::validate). For full input coverage the hop
  /// should not exceed the window size.
  #[must_use]
  pub const fn with_hop(mut self, hop: usize) -> Self {
    self.hop = hop;
    self
  }

  /// Set the overlap between consecutive windows, in elements
  /// (`hop == window - overlap`).
  ///
  /// An overlap of `0` tiles the windows. The subtraction saturates, so an
  /// overlap of at least the window size yields a zero hop that
  /// [`validate`](WindowOptions::validate) reports as
  /// [`WinditError::OverlapGeWindow`].
  #[must_use]
  pub const fn with_overlap(mut self, overlap: usize) -> Self {
    self.hop = self.window.saturating_sub(overlap);
    self
  }

  /// Set the tail policy for a final ragged window.
  #[must_use]
  pub const fn with_tail(mut self, tail: TailPolicy) -> Self {
    self.tail = tail;
    self
  }

  /// Cap the number of windows the plan may produce; exceeding it is an error.
  #[must_use]
  pub const fn with_max_windows(mut self, max_windows: usize) -> Self {
    self.max_windows = Some(max_windows);
    self
  }

  /// The window size, in elements.
  #[must_use]
  pub const fn window(&self) -> usize {
    self.window
  }

  /// The hop (stride between window starts), in elements.
  #[must_use]
  pub const fn hop(&self) -> usize {
    self.hop
  }

  /// The overlap between consecutive windows, in elements (`window - hop`,
  /// saturating at `0`).
  #[must_use]
  pub const fn overlap(&self) -> usize {
    self.window.saturating_sub(self.hop)
  }

  /// The tail policy.
  #[must_use]
  pub const fn tail(&self) -> &TailPolicy {
    &self.tail
  }

  /// The configured window cap, if any.
  #[must_use]
  pub const fn max_windows(&self) -> Option<usize> {
    self.max_windows
  }

  /// Check the geometry: the window must be non-zero and the hop must advance
  /// (equivalently, the overlap must be below the window size).
  ///
  /// # Errors
  ///
  /// - [`WinditError::ZeroWindow`] if the window size is zero.
  /// - [`WinditError::OverlapGeWindow`] if the hop is zero — the overlap is at
  ///   least the window size.
  pub const fn validate(&self) -> Result<(), WinditError> {
    if self.window == 0 {
      return Err(WinditError::ZeroWindow);
    }
    if self.hop == 0 {
      return Err(WinditError::OverlapGeWindow {
        overlap: self.overlap(),
        window: self.window,
      });
    }
    Ok(())
  }
}

/// **This spelling is persisted by downstream derivation fingerprints — change
/// it only with a breaking bump.**
///
/// `key=value` pairs in declaration order, joined by `,`, one key per document
/// field — the fingerprint is the document spelling, so a field with no serde
/// rename (every field here) prints under its own name: `window`, `hop`,
/// `tail` (nesting [`TailPolicy`]'s own `Display`), and `max_windows`. All four
/// always appear; `max_windows` prints `none` when absent rather than omitting
/// the segment (the identity-fold law: `Option` folds to `none`, never to a
/// gap).
///
/// For example, `WindowOptions::new(512).with_hop(256)
/// .with_tail(TailPolicy::DropBelowMin(3))` prints
/// `window=512,hop=256,tail=drop_below_min(3),max_windows=none`.
impl core::fmt::Display for WindowOptions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let Self {
      window,
      hop,
      tail,
      max_windows,
    } = *self;
    write!(f, "window={window},hop={hop},tail={tail},max_windows=")?;
    match max_windows {
      Some(max_windows) => write!(f, "{max_windows}"),
      None => f.write_str("none"),
    }
  }
}

/// The window planner: turns an input length and [`WindowOptions`] into the list
/// of [`Span`]s that drives pre- and post-processing.
#[derive(Clone, Copy, Debug)]
pub struct WindowPlan;

#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
const _: () = {
  use std::vec::Vec;

  impl WindowPlan {
    /// Plan the windows covering `input_len` elements under `opts`, in input
    /// order.
    ///
    /// An empty input yields an empty plan; a non-empty input can also yield an
    /// empty plan when [`TailPolicy::DropBelowMin`] drops the sole, too-short
    /// window. The final window may be ragged according to the [`TailPolicy`].
    /// When `opts.hop() <= opts.window()` every input element is covered by at
    /// least one span; a larger hop strides over the input, leaving gaps
    /// uncovered.
    ///
    /// # Errors
    ///
    /// Propagates [`WindowOptions::validate`], returns
    /// [`WinditError::TooManyWindows`] if the plan would exceed
    /// [`WindowOptions::max_windows`], and returns
    /// [`WinditError::AllocFailed`] if the plan itself cannot be allocated.
    ///
    /// `input_len` is a count, not a slice length: nothing ties it to memory
    /// that exists, so an untrusted one can name more windows than can be held.
    /// Setting [`WindowOptions::max_windows`] converts that into the sharper
    /// `TooManyWindows` at a bound of the caller's choosing.
    pub fn spans(opts: &WindowOptions, input_len: usize) -> Result<Vec<Span>, WinditError> {
      opts.validate()?;
      let (w, hop) = (opts.window(), opts.hop());
      let mut out = Vec::new();
      if input_len == 0 {
        return Ok(out);
      }
      // Reserving the plan up front is what keeps an unservable one from being
      // approached a `push` at a time: `usize::MAX` elements at hop 1 is a well
      // formed request no allocator can answer, and walking toward it is neither
      // faster to fail nor more informative than saying so.
      //
      // The loop visits `start` at 0, hop, 2*hop, ... and stops at the first one
      // that leaves at most a window of input (`input_len - start <= w`), so the
      // count is the hops strictly below `input_len - w`, plus that final tail
      // window. Deriving it from the hop alone would be wildly loose in exactly
      // the geometry that matters -- a window near `usize::MAX` with a small hop
      // places two spans, while `ceil(input_len / hop)` would ask for 1e17.
      // Capping the reservation at `max + 1` keeps `TooManyWindows` the answer
      // wherever the cap is what the plan actually runs into.
      let planned = if input_len <= w {
        1
      } else {
        ((input_len - w - 1) / hop).saturating_add(2)
      };
      let reserve = match opts.max_windows() {
        Some(max) => core::cmp::min(planned, max.saturating_add(1)),
        None => planned,
      };
      out
        .try_reserve_exact(reserve)
        .map_err(|_| WinditError::AllocFailed { elements: reserve })?;
      let mut start = 0usize;
      loop {
        // Reachable only when `hop > w` (a stride wider than the window); keeps
        // the function total instead of underflowing `input_len - start`.
        if start >= input_len {
          break;
        }
        let len = core::cmp::min(w, input_len - start);
        // `input_len - start` is safe under the loop guard (`start < input_len`);
        // computing the tail this way avoids overflowing `start + w` when
        // `input_len` is within a window of `usize::MAX`.
        let is_tail = input_len - start <= w;
        let keep = match opts.tail() {
          TailPolicy::DropBelowMin(m) => len >= *m || len == w,
          _ => true,
        };
        if keep {
          // `w` is non-zero (validated) and the loop guard gives `start <
          // input_len`, so `len` is in `1..=w`; `len` is also at most
          // `input_len - start`, so `start + len <= input_len` is representable.
          // The whole span invariant holds, and `new` cannot panic here.
          out.push(Span::new(start, len, w));
        }
        if let Some(max) = opts.max_windows() {
          if out.len() > max {
            return Err(WinditError::TooManyWindows {
              got: out.len(),
              max,
            });
          }
        }
        if is_tail {
          break;
        }
        // A hop that would carry `start` past `usize::MAX` cannot reach any further
        // in-bounds element, so stop rather than wrap.
        match start.checked_add(hop) {
          Some(s) => start = s,
          None => break,
        }
      }
      Ok(out)
    }
  }
};
