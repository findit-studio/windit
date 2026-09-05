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
/// and then the payload, `[01, 05]` under postcard. `Deserialize` asks for a
/// struct, which is what lets serde route each format to the road it belongs on:
/// a self-describing format's map goes straight to serde's own derived
/// adjacently tagged reader, so that grammar is `0.4.0`'s to the letter, and a
/// compact format's sequence goes to a small reader written for it.
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
    de::{value::MapAccessDeserializer, Error as _, Expected, MapAccess, SeqAccess, Visitor},
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

        /// The adjacently tagged document, derived.
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
    };

    // A variant with a payload: the element after the tag is that payload, and
    // it is required.
    (@element $variant:ident ( $payload:ty ), $seq:expr, $expected:expr) => {
      TailPolicy::$variant(
        $seq
          .next_element::<$payload>()?
          .ok_or_else(|| serde::de::Error::invalid_length(1, $expected))?,
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
    DropBelowMin(usize) => "drop_below_min",
    PadFull => "pad_full",
  }

  /// Into the wire form. Exhaustive over [`TailPolicy`]: a variant added to the
  /// public enum stops the build here until the wire form can express it.
  impl From<TailPolicy> for wire::TailPolicy {
    fn from(policy: TailPolicy) -> Self {
      match policy {
        TailPolicy::KeepWithCoverage => Self::KeepWithCoverage,
        TailPolicy::DropBelowMin(min) => Self::DropBelowMin(min),
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
        wire::TailPolicy::DropBelowMin(min) => Self::DropBelowMin(min),
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

  struct TailPolicyVisitor;

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

    /// The shape a self-describing format hands over — JSON and TOML, and the
    /// binary ones that report `is_human_readable() == false` while still naming
    /// their fields, CBOR and MessagePack among them.
    ///
    /// The map is handed whole to the wire enum's derived reader, so every
    /// question this road raises — whether the content precedes the tag and must
    /// be buffered, what type to ask the format for once the tag is known,
    /// whether a key arrived as text or as bytes, what to do with an entry that
    /// is neither field, and what each failure says — is answered exactly as
    /// `0.4.0` answered it, because it is the same code.
    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<TailPolicy, A::Error> {
      wire::TailPolicy::deserialize(MapAccessDeserializer::new(map)).map(Into::into)
    }

    /// The shape a format that encodes a struct as a sequence hands over: the
    /// tag, then the content the tag calls for.
    ///
    /// The derive cannot serve this road — it reads an adjacent tag by buffering
    /// the content, which asks for `deserialize_any`, and postcard answers
    /// `WontImplement` — so the tag is resolved by deserializing the tag twin,
    /// whose names and ordinals are the derive's own, and `from_tag` reads what
    /// the resolved variant declares.
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<TailPolicy, A::Error> {
      let tag = seq
        .next_element::<wire::tag::TailPolicy>()?
        .ok_or_else(|| A::Error::invalid_length(0, &self))?;
      from_tag(tag, &mut seq, &self)
    }
  }

  impl<'de> Deserialize<'de> for TailPolicy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
      // Asking for the adjacent shape itself, rather than for whatever the value
      // happens to be, is what keeps this readable by a format that cannot
      // answer `deserialize_any` — and serde routes the request to `visit_seq`
      // or `visit_map` by what the format encodes a struct as, which is the line
      // the wire actually splits on.
      deserializer.deserialize_struct(wire::NAME, wire::FIELDS, TailPolicyVisitor)
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
/// defaults to no cap (`None`). Unknown keys are rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct WindowOptions {
  window: usize,
  hop: usize,
  tail: TailPolicy,
  max_windows: Option<usize>,
}

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
