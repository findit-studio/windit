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
//!   trait, and [`WinditError`] (including its [`core::error::Error`]
//!   implementation) — is always available.
//! - **`alloc`** (default): the `Vec`-returning algorithms ([`WindowPlan::spans`],
//!   [`slice_pad_mask`], and the policies).
//! - **`std`**: implies `alloc` and links the crate against `std` (that is, it
//!   turns off the crate's `#![no_std]`). It gates no API of its own:
//!   [`WinditError`] implements [`core::error::Error`] unconditionally, and in a
//!   `std` build that is the very same trait as `std::error::Error`. The feature
//!   is kept because downstream crates conventionally unify on a `std` feature
//!   and need one to enable here.
//! - **`text`**: content-aware string chunking (adds `unicode-segmentation`).
//! - **`serde`**: `Serialize` / `Deserialize` for the configuration and policy
//!   enums.
//!
//! [`WindowPlan::spans`]: plan::WindowPlan::spans
//! [`slice_pad_mask`]: pre::slice_pad_mask
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

// House strategy: alias `alloc` as `std` so heap-type paths read `std::`
// uniformly across the `std` and `no_std + alloc` tiers. This works only for
// names `alloc` provides (`Vec`, `Box`, `String`, `vec!`) — it cannot supply
// std-only APIs such as time, fs, net, thread, or process. Any future need for
// those must be gated on the `std` feature, not assumed available through this
// alias.
#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

pub mod aggregate;
pub mod plan;
pub mod pre;
pub mod prelude;
pub mod segment;
pub mod smooth;
pub mod split;
pub mod windowed;

mod error;

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod test_support;

pub use error::WinditError;
