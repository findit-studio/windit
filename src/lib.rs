// The README is the crate's front page, and every worked example in it exercises
// the `Vec`-returning algorithms. Those live behind the heap tier, so the README
// is the crate documentation only where its examples compile; a featureless
// build gets a summary instead. Without this split the examples become doctests
// that reference items their own feature row does not export.
#![cfg_attr(any(feature = "std", feature = "alloc"), doc = include_str!("../README.md"))]
#![cfg_attr(
  not(any(feature = "std", feature = "alloc")),
  doc = "Generic windowed-sequence processing — chunk, pad/mask, aggregate, smooth, segment, split — for embeddings, VAD, and ASR.\n\nThis is the featureless tier: the type and trait surface only (`Span`, `WindowOptions`, `TailPolicy`, `Vector`, `Windowed`, `WinditError`). Enable `alloc` — the default feature — for the planner, the pre-processing helpers, the four strategy families, and the worked examples on the crate's front page."
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod aggregate;

pub mod plan;

#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod pre;
pub mod prelude;
pub mod scalar;
// The segment module spans both tiers: the `Gate`/`GatePolicy` traits, the gate
// configs, the `Segmenter` state machine, `SegmentTail`, `Range`, and
// `SegmentOptions` are featureless core (they allocate nothing), while the
// `Vec`-returning batch drivers gate on `alloc` inside the module.
pub mod segment;
// The smooth module spans both tiers likewise: the `Smoother`/`SmoothPolicy`
// traits and the `Identity`/`Ema` configs and states are featureless core, while
// the `Vec`-returning batch `smooth` driver gates on `alloc` inside the module.
pub mod smooth;
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod split;
pub mod windowed;

mod error;

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod test_support;

pub use error::WinditError;
