//! `windit` — generic windowed-sequence processing (pre + post) for embeddings, VAD, and ASR.
//!
//! `windit` is a sans-I/O, `no_std + alloc` core for the windowing, chunking,
//! aggregation, smoothing, and segmentation machinery shared across embedding,
//! VAD, and ASR model families. It owns no model code (no tokenizer, resampler,
//! mel front-end, or inference runtime); it is pure computation plus pluggable
//! policy traits.
//!
//! # The unifying idea
//!
//! A single [`WindowPlan`](plan::WindowPlan) geometry maps an input length plus a
//! fully configurable [`WindowOptions`](plan::WindowOptions) to a list of
//! [`Span`](plan::Span)s — plain `usize` element counts, so samples, tokens,
//! patches, and frames are treated identically. Those same spans drive both
//! pre-processing (slice / pad / mask) and post-processing (aggregate / smooth /
//! segment). The span is the thread that ties the two ends together.
//!
//! Values are generic over the type `V` (an embedding, a probability, a logit),
//! carried alongside their span by [`Windowed<V>`](windowed::Windowed).
//! Aggregation additionally works over any embedding type implementing
//! [`Vector`](windowed::Vector); embeddings live in f32 space.
//!
//! # Feature flags
//!
//! - **(no features)**: the type and trait surface — [`Span`](plan::Span),
//!   [`WindowOptions`](plan::WindowOptions), the [`Vector`](windowed::Vector)
//!   trait, and [`WinditError`] — is always available.
//! - **`alloc`** (default): the `Vec`-returning algorithms ([`WindowPlan::spans`],
//!   [`slice_pad_mask`], and the policies).
//! - **`std`**: implies `alloc` and adds the [`std::error::Error`] implementation
//!   for [`WinditError`].
//! - **`text`**: content-aware string chunking (adds `unicode-segmentation`).
//! - **`serde`**: `Serialize` / `Deserialize` for the configuration and policy
//!   enums.
//!
//! [`WindowPlan::spans`]: plan::WindowPlan::spans
//! [`slice_pad_mask`]: pre::slice_pad_mask
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

#[cfg(any(feature = "alloc", feature = "std"))]
extern crate alloc;

pub mod aggregate;
pub mod plan;
pub mod pre;
pub mod prelude;
pub mod segment;
pub mod smooth;
pub mod split;
pub mod windowed;

mod error;

#[cfg(all(test, feature = "alloc"))]
mod test_support;

pub use error::WinditError;
