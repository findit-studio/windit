//! Aggregation policies: combine a window sequence into a single embedding.
//!
//! `AggregatePolicy` is the object-safe seam: its one method works on plain
//! slices of a [`Real`] compute scalar (`aggregate_values`), so
//! `&dyn AggregatePolicy` is usable. The scalar is a trait type parameter
//! defaulting to `f64` — the compute domain of every shipped storage scalar —
//! which is what keeps that bare `dyn` spelling valid; another compute domain
//! names it (`dyn AggregatePolicy<C>`). The generic free
//! function `aggregate` extracts the compute slices and per-window coverages
//! from a `&[WindowEmbedding<E>]`, runs the policy, and reconstructs the
//! embedding type `E` through `Vector::from_unnormalized`. Keeping
//! reconstruction out of the trait is what lets the trait stay object-safe while
//! embedding reconstruction stays generic.
//!
//! Policy *configuration* is typed by **what it multiplies**, not by where the
//! number came from. A coverage used to be typed the other way — a
//! window-geometry fraction rather than an embedding value, therefore `f32` —
//! and that classification asked the wrong question. [`Span::coverage`] is a
//! *weight*: [`CoverageWeightedMean`] multiplies an embedding by it, inside an
//! `f64` fold. A weight resolved more coarsely than the arithmetic it drives
//! discards information that arithmetic would have used, whatever its
//! provenance. So a coverage is `f64` — the domain its own division runs in and
//! the widest this crate has — and widens into a policy's `C` through
//! [`Real::from_f64`]. A smoothing factor multiplies the accumulator too, and is
//! the compute scalar itself: [`EmaRenormalized`] carries a `C`, defaulted to
//! `f64` exactly as the trait is.
//!
//! Why a coverage is a concrete `f64` in the trait signature rather than the
//! policy's own `C`, when a coefficient is `C`: a coefficient is *configured*,
//! by a caller who already has an embedding in mind, so it can be asked for in
//! the domain that embedding computes in. A coverage is *derived* — by
//! [`Span::coverage`] from two `usize`s, in the featureless `plan` tier, before
//! any embedding or compute scalar is in sight — so there is no `C` to ask for
//! it in. `f64` is exact for the whole quotient that division can produce and
//! widens exactly into every [`Real`], which is the same shape the serde
//! selector `AggregatePolicyKind` has for the same reason: a wire value read
//! before any compute scalar exists. (Named rather than linked, here and below:
//! that selector is behind `serde` and this prose is not, so a link would be
//! unresolved on every feature row that leaves `serde` off.)
//!
//! Built-in strategies weight the windows by different signals:
//!
//! - `CoverageWeightedMean` (the default) weights by [`Span::coverage`], so
//!   fuller windows count more.
//! - `MeanRenormalized` weights uniformly (a renormalized arithmetic mean).
//! - `EmaRenormalized` weights by recency (an exponential moving average).
//! - `SaliencyWeighted` weights by each input's L2 norm, so higher-magnitude
//!   (more salient) inputs dominate. Because `aggregate` passes already-unit
//!   embeddings, saliency is meaningful only when a caller invokes
//!   `aggregate_values` directly with vectors that still carry magnitude.
//!
//! `keep_separate` is the multi-vector path: it returns every window unchanged
//! for callers that want per-window embeddings rather than one summary.
//!
//! # Scale
//!
//! Aggregation runs in [`ComputeOf<E>`](crate::windowed::ComputeOf), which is
//! `f64` for both shipped scalars: an `f32` embedding widens to `f64` before a
//! single value is folded. That is what disarms the whole class of magnitude
//! hazard for `f32` inputs — every `f32` is exact in `f64`, every `f32`
//! subnormal is a normal `f64`, and `f32::MAX` squared is ~1.2e77 against
//! `f64`'s ~1.8e308 ceiling — so a sum that would have overflowed, flushed a
//! subnormal, or lost a cancellation in `f32` does none of those here. No power
//! of-two prescaling of the fold is applied, and none is needed.
//!
//! Two hazards survive into `f64` itself, because `f64` is the widest domain
//! there is, and both are handled where they arise rather than by a blanket
//! shift:
//!
//! - **Cancellation across a wide exponent spread.** A weighted sum whose exact
//!   value is zero can fold to an order-dependent non-zero residue once a small
//!   term is absorbed into a large partial sum and the large term is later
//!   subtracted away. The accumulation is therefore a compensated
//!   (Neumaier's variant of Kahan-Babuška) sum, and a determinacy gate then
//!   rejects any result at or below the fold's own provable rounding floor — so
//!   [`WinditError::NonFinite`] keeps meaning "no direction determined at
//!   working precision" rather than "the fold happened to round to a residue".
//!   The [Input domain](self#input-domain) note states the bound this rests on.
//! - **A norm that is not representable although the vector is.**
//!   `[f64::MAX, f64::MAX]` is an ordinary diagonal whose norm, `sqrt(2) *
//!   f64::MAX`, overflows. The renormalization divides each component by its own
//!   `2^exponent` power-of-two scale and by the scaled norm separately, so it
//!   never forms that norm; dividing by a power of two is exact, so the quotient
//!   is the direct `v_i / norm` to the bit wherever the direct computation was
//!   valid. A vector's squares leaving the range (`[f64::MAX, 0.0]` squares to
//!   infinity, `[f64::MIN_POSITIVE, 0.0]` to zero) is the same mechanism and the
//!   same fix.
//!
//! There is no second attempt anywhere, which is what keeps
//! [`WinditError::NonFinite`] meaning what it says: an all-zero (or exactly
//! cancelling) vector, or a component that is itself not finite. A retry cannot
//! tell those from a norm that merely overflowed, and a vector whose components
//! cancel exactly is not a vector whose norm was unrepresentable — it is a
//! vector with no direction.
//!
//! Rather than let any policy reach for the edge of `f64`, aggregation enforces
//! an input domain that keeps every fold clear of it — see below.
//!
//! # Input domain
//!
//! Every input component must be finite and either zero or of a magnitude
//! between [`Real::MIN_AGG_MAGNITUDE`] and [`Real::MAX_AGG_MAGNITUDE`]
//! (`[2^-400, 2^400]` for `f64`, about `[3.9e-121, 2.6e120]`); every coverage
//! must be finite and in `[0, 1]`. Inputs outside this domain are rejected with
//! [`WinditError::MagnitudeOutOfRange`] or [`WinditError::CoverageOutOfRange`]
//! before any arithmetic. The bounds are sized so that within them every
//! intermediate of every built-in policy is finite, and — for the policies whose
//! weights the domain itself bounds below ([`MeanRenormalized`] at `1` and
//! [`SaliencyWeighted`] at a norm `>= 2^-400`) — every nonzero intermediate is a
//! normal `f64`, no overflow and no subnormal flush, including the squared term
//! [`SaliencyWeighted`] forms. Two policies have weights that reach below that,
//! and in both what is unbounded is the *ratio* between weights rather than their
//! scale:
//!
//! - [`EmaRenormalized`]'s recency weights `w_i = alpha * (1 - alpha)^(n - 1 - i)`
//!   sum to exactly `1`, so there is no scale to divide out, but they decay
//!   without limit against the newest window: at a large window count the oldest
//!   windows' products underflow toward a subnormal (or to zero) even for
//!   in-domain inputs.
//! - [`CoverageWeightedMean`] folds `c_i / max_j c_j`, so its largest weight is
//!   exactly `1` however the caller scaled the slice — see that type's *Weights
//!   up to scale* note — but the domain admits the rest anywhere in `[0, 1]`, so
//!   a window weighing `2^-1000` against the fullest one drives its own product
//!   subnormal.
//!
//! Both are regimes the determinacy gate's absolute floor handles (see below),
//! and the floor's soundness argument is about products rather than about which
//! policy formed them. What a pinned *largest* weight buys is the other half of
//! the picture: the fold's accumulated mass is at least the heaviest window's
//! own, so the floor can decide a verdict only when the heaviest windows carry
//! no mass at all. For [`CoverageWeightedMean`] that means the fullest windows
//! are themselves the zero vector — never that the caller's coverages were
//! small, which is a scale and cannot change a normalized weighted mean.
//!
//! Every value an `f32`-storage embedding can produce lies more than 250 binary
//! orders inside this window on both sides, so no realizable `f32` input ever
//! reaches a boundary.
//!
//! Within the domain, an aggregated result is the direction of a vector within
//! `4 * `[`Real::EPSILON`]` * ||M|| + K_abs` of the exact weighted sum, where `M`
//! is the componentwise sum of the folded term magnitudes and `K_abs` is the small
//! absolute term defined below; any result whose norm is at or below
//! `16 * `[`Real::EPSILON`]` * ||M|| + `[`Real::MIN_GATE_THRESHOLD`] is reported as
//! [`WinditError::NonFinite`] — no direction is determined at working precision.
//! This is the crate's one accuracy claim, and it is a theorem rather than an
//! observation. Each product `w_i * e_i` is rounded relatively when it is a normal
//! `f64` (by at most `u * |w_i * e_i|`, `u = EPSILON / 2`) and absolutely when it
//! has underflowed toward a subnormal (by at most `2^-1075`, half the subnormal
//! spacing). Per dimension the relative parts sum to at most `u * M_j`, the
//! Neumaier fold adds at most `2u * M_j` (plus an `O(n * u^2) * M_j` tail, and it
//! is exact for subnormal operands), together at most `4 * EPSILON * M_j`; the
//! absolute parts sum to at most `n * 2^-1075`. Over all dimensions,
//! `||R - exact|| <= 4 * EPSILON * ||M|| + K_abs` with
//! `K_abs <= sqrt(dim) * n * 2^-1075 <= 2^-1018` for any `n <= 2^40` and
//! `dim <= 2^32`. The threshold
//! `τ = 16 * EPSILON * ||M|| + `[`Real::MIN_GATE_THRESHOLD`] carries a matching
//! absolute floor (`2^-1000` for `f64`, above `K_abs` and — for any mass a
//! domain-bounded weight accumulates — far below `16 * EPSILON * ||M||`), so an
//! exactly cancelling sum has `||R|| <= 4 * EPSILON * ||M|| + K_abs < τ` and is
//! always gated, whatever the ordering, tier structure, or weight range — so no
//! fold can fabricate a direction from in-domain cancellation without violating the
//! bound. When EMA's unbounded-below weights drive the whole fold subnormal,
//! `||M||` is itself subnormal and `16 * EPSILON * ||M||` underflows, leaving the
//! floor to gate alone: the entire signal then sits below the precision the domain
//! guarantees, so `NonFinite` remains the honest verdict. The floor also engages
//! earlier, while every product is still normal: once the accumulated mass falls
//! below about `2^-948`, `16 * EPSILON * ||M||` itself drops beneath the `2^-1000`
//! floor and the floor decides the verdict alone, monotonically turning a
//! sub-floor direction into `NonFinite` rather than admitting it — an
//! over-rejection-only widening of the gate, pinned by a regression test.
//!
//! An absolute floor is only ever sound against a quantity carried in the
//! embedding's own units, which `||M||` and `||R||` are and a *weight* is not.
//! So reaching either regime takes an unbounded weight **ratio** —
//! [`EmaRenormalized`]'s decaying recency factors, or a [`CoverageWeightedMean`]
//! fold whose fullest windows are themselves all zero — and never a weight
//! **scale**, which the renormalization ending every policy divides back out.
//! Neither regime is one a realizable `f32` workload reaches through
//! [`aggregate`].
//!
//! [`Real`]: crate::scalar::Real
//! [`Real::from_f64`]: crate::scalar::Real::from_f64
//! [`Span`]: crate::plan::Span
//! [`Span::coverage`]: crate::plan::Span::coverage

use std::vec::Vec;

use crate::{
  error::WinditError,
  scalar::Real,
  windowed::{ComputeOf, Vector, WindowEmbedding},
};

#[cfg(test)]
mod tests;

/// A policy that combines a sequence of window embeddings into one embedding.
///
/// The single required method operates on plain slices so the trait is
/// object-safe (`&dyn AggregatePolicy` works). Embedding reconstruction lives in
/// the generic free function [`aggregate`], not here.
///
/// `C` is the compute scalar — the [`Real`] domain the math runs in — and
/// defaults to `f64`. Because both shipped scalars compute in `f64` (an `f32`
/// embedding widens to it), that default is the domain every built-in
/// aggregation actually uses, and it keeps `dyn AggregatePolicy` and
/// `Box<dyn AggregatePolicy>` spelling the object every ordinary embedding
/// needs. A custom compute scalar names it, as in `Box<dyn AggregatePolicy<C>>`.
/// Note that trait objects are per-scalar: two `AggregatePolicy` objects over
/// different `C` are unrelated types and cannot share one collection.
///
/// # Custom policies
///
/// Implement [`aggregate_values`](AggregatePolicy::aggregate_values) to add a
/// strategy. This one keeps the first window unchanged, and serves the default
/// `f64` compute domain by leaving the type parameter off:
///
/// ```
/// use windit::aggregate::AggregatePolicy;
/// use windit::WinditError;
///
/// struct FirstWindow;
///
/// impl AggregatePolicy for FirstWindow {
///   fn aggregate_values(
///     &self,
///     embeddings: &[&[f64]],
///     _coverages: &[f64],
///     dim: usize,
///   ) -> Result<Vec<f64>, WinditError> {
///     let first = embeddings.first().ok_or(WinditError::Empty)?;
///     if first.len() != dim {
///       return Err(WinditError::DimMismatch { got: first.len(), expected: dim });
///     }
///     Ok(first.to_vec())
///   }
/// }
/// ```
///
/// Writing `impl<C: Real> AggregatePolicy<C> for FirstWindow` instead — with
/// `&[&[C]]` and `Vec<C>` — makes the same policy serve every compute scalar.
pub trait AggregatePolicy<C: Real = f64> {
  /// Combine `embeddings` (each a `dim`-length slice of the compute scalar)
  /// with their matching `coverages` into a single `dim`-length vector.
  ///
  /// The built-in policies return an L2-normalized vector, and a custom policy
  /// should do the same; either way [`aggregate`] re-normalizes the result
  /// through [`Vector::from_unnormalized`]. `coverages` must have the same length
  /// as `embeddings` even for policies that do not weight by coverage. They are
  /// `f64` at every scalar, and deliberately not `C`: a coverage is *derived*
  /// rather than configured — [`Span::coverage`](crate::plan::Span::coverage)
  /// computes it from two `usize`s before any embedding or compute scalar
  /// exists — so it arrives in the domain that division runs in and widens
  /// through [`Real::from_f64`] where a policy uses it. It is still a weight on
  /// an `f64` fold, which is why it is not narrower than one.
  ///
  /// Every component must be finite and either zero or of magnitude within
  /// `[MIN_AGG_MAGNITUDE, MAX_AGG_MAGNITUDE]`, and every coverage finite and in
  /// `[0, 1]`; see the module [Input domain](self#input-domain) note.
  ///
  /// # Errors
  ///
  /// - [`WinditError::Empty`] if `embeddings` is empty.
  /// - [`WinditError::DimMismatch`] if `coverages.len() != embeddings.len()` or
  ///   any embedding's length differs from `dim`.
  /// - [`WinditError::MagnitudeOutOfRange`] if a nonzero component's magnitude is
  ///   outside `[MIN_AGG_MAGNITUDE, MAX_AGG_MAGNITUDE]`.
  /// - [`WinditError::CoverageOutOfRange`] if a coverage is not a finite fraction
  ///   in `[0, 1]`.
  /// - [`WinditError::NonFinite`] if the combined vector cannot be normalized to
  ///   a finite unit vector (zero norm or a non-finite component).
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f64],
    dim: usize,
  ) -> Result<Vec<C>, WinditError>;
}

/// Aggregate a sequence of window embeddings into one embedding of type `E`.
///
/// Projects each window into `E`'s compute domain through
/// [`compute_components`](Vector::compute_components), pairs it with its
/// [`Span::coverage`](crate::plan::Span::coverage), runs `policy` there, and
/// reconstructs `E` via [`Vector::from_unnormalized`]. Works with any policy,
/// including `&dyn AggregatePolicy`.
///
/// # Errors
///
/// [`WinditError::Empty`] if `windows` is empty; otherwise any error from the
/// per-window projection (for example [`WinditError::MissingDequantization`] when
/// quantized storage did not override its dequantization), from the policy, or
/// from [`Vector::from_unnormalized`].
pub fn aggregate<E, P>(policy: &P, windows: &[WindowEmbedding<E>]) -> Result<E, WinditError>
where
  E: Vector,
  P: AggregatePolicy<ComputeOf<E>> + ?Sized,
{
  if windows.is_empty() {
    return Err(WinditError::Empty);
  }
  let dim = windows[0].value.dim();
  let mut coverages = try_vec_with_capacity(windows.len())?;
  for w in windows {
    coverages.push(w.span.coverage());
  }

  // Project each window into its compute domain: a zero-copy borrow when the
  // storage already is the compute scalar (`f64`), an exact elementwise widening
  // otherwise (`f32`, `f16`, `bf16`), or the implementor's own dequantization
  // (quantized storage overrides `compute_components`). This runs before any
  // weighting, so every policy — including the magnitude-weighted one — sees
  // represented values, and it is these slices the input-domain check validates.
  let mut cows = try_vec_with_capacity(windows.len())?;
  for w in windows {
    cows.push(w.value.compute_components()?);
  }
  let mut embeddings: Vec<&[ComputeOf<E>]> = try_vec_with_capacity(cows.len())?;
  for c in &cows {
    embeddings.push(c.as_ref());
  }
  let raw = policy.aggregate_values(&embeddings, &coverages, dim)?;
  E::from_unnormalized(&raw)
}

/// A `Vec` that can hold `n` elements, or [`WinditError::AllocFailed`] when the
/// allocator cannot (or refuses to) provide the space.
///
/// The fallible counterpart to `Vec::with_capacity` for the growing buffers on
/// these `Result`-returning paths. Every buffer an aggregation grows is sized by
/// the caller's window count or embedding dimension — counts that need not
/// correspond to memory that exists — so a refused allocation must surface as a
/// typed error rather than abort the process. `try_reserve_exact` because each
/// buffer is then filled to exactly `n` and never grown again.
///
/// `pub(crate)` so [`Vector::compute_components`](crate::windowed::Vector::compute_components)'s
/// default projection can share the same typed-OOM discipline.
pub(crate) fn try_vec_with_capacity<T>(n: usize) -> Result<Vec<T>, WinditError> {
  let mut v = Vec::new();
  v.try_reserve_exact(n)
    .map_err(|_| WinditError::AllocFailed { elements: n })?;
  Ok(v)
}

/// A `dim`-length vector of [`Real::ZERO`], or [`WinditError::AllocFailed`].
///
/// The accumulator every weighted sum folds into; [`try_vec_with_capacity`]
/// reserves it and `resize` fills the reserved space without growing again.
fn try_zeroed<C: Real>(dim: usize) -> Result<Vec<C>, WinditError> {
  let mut v = try_vec_with_capacity(dim)?;
  v.resize(dim, C::ZERO);
  Ok(v)
}

/// The multi-vector path: return every window unchanged.
///
/// The counterpart to [`aggregate`], for callers that keep per-window embeddings
/// (for example, one speaker centroid per window) instead of collapsing them.
#[must_use]
pub fn keep_separate<E>(windows: Vec<WindowEmbedding<E>>) -> Vec<WindowEmbedding<E>> {
  windows
}

/// Coverage-weighted mean, then L2 renormalization (the default strategy).
///
/// Each window contributes in proportion to its [`Span::coverage`](crate::plan::Span::coverage),
/// so a padded ragged tail counts less than a full window.
///
/// # Weights up to scale
///
/// The weights of a *normalized* weighted mean are defined only up to a common
/// positive factor: `sum_i (s * c_i) * e_i` is `s * sum_i c_i * e_i`, and the
/// renormalization that ends this policy divides `s` back out. So the **scale**
/// of the coverage slice carries no information about the answer — only the
/// ratios between its entries do — and multiplying every coverage by a positive
/// factor must leave the result unchanged.
///
/// It is a property of the policy, so this policy establishes it rather than
/// hoping for it: the fold's weights are `c_i / max_j c_j`, and the largest of
/// them is exactly `1.0`. Scaling every coverage by an `s` that is itself exact
/// leaves each quotient's exact value untouched, and IEEE division is correctly
/// rounded, so the weights — and with them the whole fold, bit for bit — are the
/// same. An all-zero slice is not a scale of anything: every weight is zero, the
/// exact sum is the zero vector, and the [determinacy gate](self#input-domain)
/// reports [`WinditError::NonFinite`], no direction to report.
///
/// Two consequences worth naming. A slice that already contains a full window
/// (`1.0`, as every plan with one does) divides by exactly `1.0` and folds
/// bit-identically to an un-normalized fold. And the [input domain](self#input-domain)'s
/// `[0, 1]` is now the whole of the contract: a coverage anywhere in it, however
/// small, weighs against the others rather than against `f64`'s exponent range.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoverageWeightedMean;

/// Uniform (unweighted) mean, then L2 renormalization.
///
/// Every window contributes equally regardless of coverage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MeanRenormalized;

/// Exponential moving average across the window sequence, then L2 renormalization.
///
/// State advances `s_i = alpha * emb_i + (1 - alpha) * s_{i-1}` from `s_0 = emb_0`,
/// so later windows weigh more (recency). `coverages` are ignored beyond the
/// length check. `alpha` must be in `[0, 1]`; an out-of-range or non-finite
/// `alpha` is rejected with [`WinditError::AlphaOutOfRange`] rather than
/// silently producing a non-convex (sign-flipping) combination.
///
/// Like [`WindowOptions`](crate::plan::WindowOptions), construction is
/// infallible and the range is checked where the value is used — here in
/// [`aggregate_values`](AggregatePolicy::aggregate_values), which already returns a
/// `Result`.
///
/// # The coefficient is the compute scalar
///
/// `C` is the [`Real`] the fold runs in, defaulted to `f64` exactly as
/// [`AggregatePolicy`] is, so `EmaRenormalized::new(0.3)` needs no turbofish and
/// inference takes `C` from the embeddings the policy is about to run over. The
/// coefficient is that `C` rather than an `f32` widened into it, because it
/// multiplies the accumulator: an `f32` field cannot hold `1 - 2^-30` (its
/// nearest `f32` is exactly `1.0`, at which the weights collapse to
/// `[0, .., 0, 1]` and the fold returns its last window), and its grid is
/// `2^-24` apart relatively where the weights, the products and the compensated
/// sum all round at `2^-53`. The same argument decided the coverage channel:
/// [`Span::coverage`](crate::plan::Span::coverage) is a weight on this fold too,
/// so its `f32` grid inside an `f64` sum was this defect wearing a different
/// provenance, and it is `f64` now. What had kept it was price rather than
/// numerics — widening it changes the object-safe
/// [`AggregatePolicy::aggregate_values`] signature every custom policy
/// implements — and a price is a reason to schedule a break, not to leave one
/// standing.
///
/// Carrying the domain as a type parameter rather than hardcoding `f64` is what
/// keeps the policy honest if a second `Real` is ever sealed in: its
/// coefficient would follow its own domain with no further signature change.
/// The serde selector `AggregatePolicyKind` is the one place a *bare* `f64`
/// remains, being a wire type that is read before any compute scalar exists.
///
/// The bound is on the type and not only on its impls, matching
/// [`AggregatePolicy`] itself: `C` names a compute domain, and
/// `EmaRenormalized<String>` is not a type this crate wants to be nameable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmaRenormalized<C: Real = f64> {
  alpha: C,
}

impl<C: Real> EmaRenormalized<C> {
  /// An EMA aggregation with the given smoothing factor.
  ///
  /// `alpha` is not validated here: a value outside `[0, 1]` (or a NaN) is
  /// reported as [`AlphaOutOfRange`](WinditError::AlphaOutOfRange) by
  /// [`aggregate_values`](AggregatePolicy::aggregate_values). Deferring the check is
  /// what keeps this constructor usable from `AggregatePolicyKind::into_policy`,
  /// which builds a policy from deserialized configuration and has no error
  /// channel of its own — and, since no comparison runs here, what keeps this
  /// constructor `const` at a generic `C` where [`VectorEma::new`] cannot be.
  ///
  /// [`VectorEma::new`]: crate::smooth::VectorEma::new
  #[must_use]
  pub const fn new(alpha: C) -> Self {
    Self { alpha }
  }

  /// The smoothing factor: larger values track recent windows more.
  #[must_use]
  pub const fn alpha(&self) -> C {
    self.alpha
  }
}

/// L2-norm-weighted mean, then renormalization: higher-magnitude inputs dominate.
///
/// Each window is weighted by the L2 norm of its input slice, so more salient
/// (larger-magnitude) vectors pull the result toward them. `coverages` are
/// ignored beyond the length check. This differs from the other strategies only
/// when the inputs carry magnitude; [`aggregate`] feeds unit vectors, so use
/// [`aggregate_values`](AggregatePolicy::aggregate_values) directly to exploit it.
///
/// Because it squares magnitudes (weight times component, and the weight is
/// itself a norm), its intermediates reach higher than any linear policy's. The
/// crate-level [input domain](self#input-domain) is sized so even that square
/// stays a finite, normal `f64`, which is why this policy needs no window of its
/// own: a component outside the domain is rejected by the shared input check
/// before the square is ever formed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SaliencyWeighted;

impl<C: Real> AggregatePolicy<C> for CoverageWeightedMean {
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f64],
    dim: usize,
  ) -> Result<Vec<C>, WinditError> {
    // The weights are the coverages divided by the largest of them, which is the
    // policy's scale invariance made structural rather than argued (see the type's
    // *Weights up to scale* note). `max_magnitude` is the right fold and not
    // merely a convenient one: it is the same "largest, and NaN never wins"
    // reduction, and its `abs` is the identity on the `[0, 1]` the domain admits —
    // a coverage outside it reaches `check_inputs` below and is rejected before
    // any weight is read.
    let largest = max_magnitude(coverages);
    weighted_sum_renorm(embeddings, coverages, dim, move |i, _| {
      if largest > 0.0 {
        // `coverages[i] <= largest` and both are positive, so the quotient is in
        // `(0, 1]` — it cannot overflow, and it cannot underflow either, being at
        // least `coverages[i]` itself.
        C::from_f64(coverages[i] / largest)
      } else {
        // Every coverage is zero. The exact weighted sum is the zero vector, and
        // the determinacy gate is what reports it; the division is skipped rather
        // than allowed to produce the `0 / 0` NaN that no gate can see.
        C::ZERO
      }
    })
  }
}

impl<C: Real> AggregatePolicy<C> for MeanRenormalized {
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f64],
    dim: usize,
  ) -> Result<Vec<C>, WinditError> {
    weighted_sum_renorm(embeddings, coverages, dim, |_, _| C::ONE)
  }
}

impl<C: Real> AggregatePolicy<C> for SaliencyWeighted {
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f64],
    dim: usize,
  ) -> Result<Vec<C>, WinditError> {
    check_inputs(embeddings, coverages, dim)?;
    // Materialized rather than recomputed inside the fold: a norm is a full pass
    // over a window, and `weighted_sum_renorm` reads each weight once per
    // dimension. Each norm is taken against the window's own power-of-two scale
    // ([`l2_norm`]), so a window whose norm is not representable still weighs in;
    // the shared magnitude divides back out in the renormalization that ends the
    // policy, leaving only the ratios a weight means here.
    let mut weights = try_vec_with_capacity(embeddings.len())?;
    for emb in embeddings {
      weights.push(l2_norm(emb));
    }
    weighted_sum_renorm(embeddings, coverages, dim, |i, _| weights[i])
  }
}

impl<C: Real> AggregatePolicy<C> for EmaRenormalized<C> {
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f64],
    dim: usize,
  ) -> Result<Vec<C>, WinditError> {
    // A convex EMA needs alpha in [0, 1]; anything else (including NaN, which
    // fails both comparisons) is a configuration error, checked first. Spelled
    // as two comparisons rather than as a `RangeInclusive::contains`, which
    // needs `PartialOrd<C>` on a *literal*: the coefficient is now the compute
    // scalar itself, so the bounds are `C::ZERO` and `C::ONE`.
    if !(self.alpha >= C::ZERO && self.alpha <= C::ONE) {
      return Err(WinditError::AlphaOutOfRange);
    }
    // The recurrence `s_i = alpha*e_i + (1-alpha)*s_{i-1}` from `s_0 = e_0` is
    // the convex combination `sum_i w_i * e_i` with `w_0 = (1-alpha)^(n-1)` and
    // `w_i = alpha*(1-alpha)^(n-1-i)` for `i >= 1`. Building those weights and
    // folding through `weighted_sum_renorm` gives EMA the same input domain,
    // determinacy gate, and error bound as every other policy from one proof —
    // there is no separate recurrence fold left to fabricate a direction. Unlike
    // the other policies these weights are unbounded below, so at a large window
    // count the oldest windows' products underflow toward a subnormal; the gate's
    // `MIN_GATE_THRESHOLD` floor is what keeps that regime sound (module Input
    // domain note). The dyadic case (`alpha = 0.5` over basis vectors) reproduces
    // the old recurrence bit for bit. The coefficient arrives already in `C` —
    // it is configured in the compute domain rather than widened into it — so
    // `1 - alpha` and every weight below carry the precision the type promises.
    let alpha = self.alpha;
    let complement = C::ONE - alpha;
    let n = embeddings.len();
    let mut weights = try_zeroed::<C>(n)?;
    // One backward pass carrying `power = complement^(n-1-i)`; the oldest window
    // gets the bare `complement^(n-1)` with no `alpha` factor.
    let mut power = C::ONE;
    for i in (1..n).rev() {
      weights[i] = alpha * power;
      power = power * complement;
    }
    if n > 0 {
      weights[0] = power;
    }
    weighted_sum_renorm(embeddings, coverages, dim, |i, _| weights[i])
  }
}

/// Serde-serializable selector over the built-in aggregation policies.
///
/// Deserialize a configured choice, then [`into_policy`](AggregatePolicyKind::into_policy)
/// to obtain a boxed [`AggregatePolicy`]. Requires `alloc` (for the boxed policy)
/// in addition to `serde`.
#[cfg(all(feature = "serde", any(feature = "std", feature = "alloc")))]
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum AggregatePolicyKind {
  /// Selects [`CoverageWeightedMean`].
  CoverageWeightedMean,
  /// Selects [`MeanRenormalized`].
  MeanRenormalized,
  /// Selects [`EmaRenormalized`] with the given smoothing factor.
  Ema {
    /// The EMA smoothing factor, widened into the compute scalar and forwarded
    /// to [`EmaRenormalized::new`].
    ///
    /// `f64` rather than the compute scalar `C`: this enum is the *wire* type,
    /// deserialized before any embedding — and so before `C` — is in sight, and
    /// a decimal in a configuration file has no compute domain of its own.
    /// [`into_policy`](AggregatePolicyKind::into_policy) widens it through
    /// [`Real::from_f64`], which is exact for every implementor.
    alpha: f64,
  },
  /// Selects [`SaliencyWeighted`].
  SaliencyWeighted,
}

#[cfg(all(feature = "serde", any(feature = "std", feature = "alloc")))]
impl AggregatePolicyKind {
  /// Build the boxed built-in policy this kind selects, at the compute scalar
  /// `C`.
  ///
  /// `C` is normally inferred from the embeddings the policy is about to run
  /// over — `aggregate(kind.into_policy().as_ref(), &windows)` needs no
  /// annotation. A turbofish is required only when the boxed policy is bound to
  /// a `let` that nothing downstream pins, as in `into_policy::<f32>()`.
  #[must_use]
  pub fn into_policy<C: Real>(self) -> std::boxed::Box<dyn AggregatePolicy<C>> {
    use std::boxed::Box;
    match self {
      Self::CoverageWeightedMean => Box::new(CoverageWeightedMean),
      Self::MeanRenormalized => Box::new(MeanRenormalized),
      Self::Ema { alpha } => Box::new(EmaRenormalized::new(C::from_f64(alpha))),
      Self::SaliencyWeighted => Box::new(SaliencyWeighted),
    }
  }
}

/// Validate `embeddings`, `coverages`, and every component against the
/// aggregation input domain, before any arithmetic runs.
///
/// Beyond the structural checks — non-empty, `coverages` length matching
/// `embeddings`, every embedding of length `dim` — this rejects a non-finite
/// component ([`NonFinite`](WinditError::NonFinite)), a nonzero component whose
/// magnitude is outside `[MIN_AGG_MAGNITUDE, MAX_AGG_MAGNITUDE]`
/// ([`MagnitudeOutOfRange`](WinditError::MagnitudeOutOfRange)), and a coverage
/// that is not a finite fraction in `[0, 1]`
/// ([`CoverageOutOfRange`](WinditError::CoverageOutOfRange)). Enforcing the
/// domain here — the one choke point every built-in policy passes through — is
/// what lets each fold run without overflow or subnormal flush; see the module
/// [Scale](self#scale) note.
fn check_inputs<C: Real>(
  embeddings: &[&[C]],
  coverages: &[f64],
  dim: usize,
) -> Result<(), WinditError> {
  if embeddings.is_empty() {
    return Err(WinditError::Empty);
  }
  if coverages.len() != embeddings.len() {
    return Err(WinditError::DimMismatch {
      got: coverages.len(),
      expected: embeddings.len(),
    });
  }
  for (window, (emb, &coverage)) in embeddings.iter().zip(coverages).enumerate() {
    if emb.len() != dim {
      return Err(WinditError::DimMismatch {
        got: emb.len(),
        expected: dim,
      });
    }
    if !(coverage.is_finite() && (0.0..=1.0).contains(&coverage)) {
      return Err(WinditError::CoverageOutOfRange { window });
    }
    for (component, &x) in emb.iter().enumerate() {
      if !x.is_finite() {
        return Err(WinditError::NonFinite);
      }
      if x != C::ZERO && (x.abs() < C::MIN_AGG_MAGNITUDE || x.abs() > C::MAX_AGG_MAGNITUDE) {
        return Err(WinditError::MagnitudeOutOfRange { window, component });
      }
    }
  }
  Ok(())
}

/// Accumulate `sum_i weight(i, emb_i) * emb_i`, gate it against its own rounding
/// floor, and L2-renormalize it.
///
/// One pass, no retry, no prescaling: the compute scalar is `f64` (an `f32`
/// embedding widened before this ran), and [`check_inputs`] has confined every
/// component to the input domain, so every product and partial sum is finite —
/// and normal wherever the domain bounds the weight below, while an unbounded
/// weight *ratio* ([`EmaRenormalized`]'s decaying recency factors, or a coverage
/// far below the fullest window's) can drive a product subnormal. The
/// sum is *compensated* (Neumaier, exact for subnormal operands), and alongside it
/// the routine accumulates `M`, the componentwise sum of the term magnitudes.
/// Before normalizing, a determinacy gate rejects any result whose norm is at or
/// below `16 * EPSILON * ||M|| + `[`MIN_GATE_THRESHOLD`](Real::MIN_GATE_THRESHOLD):
/// within the fold's provable `4 * EPSILON * ||M|| + K_abs` error bound the exact
/// weighted sum is indistinguishable from zero there, so a smaller residue is
/// rounding noise with no direction — not a vector for [`l2_renorm`] to amplify
/// into a fabricated unit direction. The absolute floor keeps the gate sound where
/// subnormal products make `16 * EPSILON * ||M||` underflow. The bound is the
/// crate's one accuracy claim; see the module [Input domain](self#input-domain)
/// note for the proof.
fn weighted_sum_renorm<C: Real>(
  embeddings: &[&[C]],
  coverages: &[f64],
  dim: usize,
  weight: impl Fn(usize, &[C]) -> C,
) -> Result<Vec<C>, WinditError> {
  // `dim` is caller-supplied, but this has just proved it is the length of an
  // embedding that exists, so the accumulators below are bounded by real data.
  check_inputs(embeddings, coverages, dim)?;
  let mut acc = try_zeroed(dim)?;
  // The running Neumaier compensation, one term per dimension: the sum of the
  // low-order bits `acc` could not hold as it grew.
  let mut comp = try_zeroed::<C>(dim)?;
  // `M`: the running sum of term magnitudes per dimension. Plain monotone adds
  // (no cancellation), so it is a faithful measure of the mass the fold summed,
  // and its own accumulation error only tightens the gate.
  let mut mag = try_zeroed::<C>(dim)?;
  for (i, emb) in embeddings.iter().enumerate() {
    let w = weight(i, emb);
    for (((a, c), m), &e) in acc
      .iter_mut()
      .zip(comp.iter_mut())
      .zip(mag.iter_mut())
      .zip(emb.iter())
    {
      let term = w * e;
      neumaier_add(a, c, term);
      *m = *m + term.abs();
    }
  }
  for (a, &c) in acc.iter_mut().zip(comp.iter()) {
    *a = *a + c;
  }
  // Determinacy gate: reject a result at or below the fold's own rounding floor
  // rather than let `l2_renorm` amplify rounding noise into a direction. `K = 16`
  // against the proven `<= 4 * EPSILON * ||M||` relative bound, plus the absolute
  // `MIN_GATE_THRESHOLD` floor. The floor dominates the residue once
  // `EmaRenormalized`'s unbounded-below recency weights push the fold's products
  // subnormal — there `16 * EPSILON * ||M||` itself underflows to zero and per-term
  // rounding turns absolute, so without the floor the gate would degenerate into an
  // exact-zero check a nonzero subnormal residue slips past (module Input domain
  // note). With it, exact cancellation (`||exact|| = 0`) is always caught, at every
  // ordering, tier structure, and weight range. Wherever the fold's heaviest
  // window carries mass of its own the floor sits far under
  // `16 * EPSILON * ||M||` and changes no verdict; only a fold whose whole mass
  // rides on a far lighter weight reaches it.
  let tau = C::from_f32(16.0) * C::EPSILON * l2_norm(&mag) + C::MIN_GATE_THRESHOLD;
  if l2_norm(&acc) <= tau {
    return Err(WinditError::NonFinite);
  }
  l2_renorm(&mut acc)?;
  Ok(acc)
}

/// Add `term` into the running sum `acc` with Neumaier compensation `comp`.
///
/// The correction is `(larger - new_sum) + smaller`: the part of the smaller
/// magnitude that `new_sum` could not represent, which is exactly what a naive
/// `acc + term` discards. Accumulated into `comp` and folded back once at the
/// end, it holds the fold's error to a small multiple of the accumulated term
/// magnitude, which is what makes the determinacy gate in
/// [`weighted_sum_renorm`] sound.
fn neumaier_add<C: Real>(acc: &mut C, comp: &mut C, term: C) {
  let sum = *acc + term;
  *comp = *comp
    + if acc.abs() >= term.abs() {
      (*acc - sum) + term
    } else {
      (term - sum) + *acc
    };
  *acc = sum;
}

/// The largest absolute component of `v`, or `ZERO` for an empty one.
///
/// NaN compares false against everything, so it never becomes the maximum; it
/// reaches the caller's own sum instead and carries through to the non-finite
/// result that rejects the vector.
fn max_magnitude<C: Real>(v: &[C]) -> C {
  let mut max = C::ZERO;
  for &x in v {
    let m = x.abs();
    if m > max {
      max = m;
    }
  }
  max
}

/// The exponent of the power of two a reduction over `v` divides by: that of
/// `v`'s largest component.
///
/// `None` when `v` has no scale to speak of — it is empty or all zero — or when
/// a component is infinite. Both are conditions to reject rather than reduce.
fn scale_exponent<C: Real>(v: &[C]) -> Option<i32> {
  let m = max_magnitude(v);
  if m == C::ZERO || !m.is_finite() {
    return None;
  }
  Some(m.exponent())
}

/// `sum_i (v_i / 2^exp)^2`, as an explicit left fold from `ZERO`.
///
/// With `exp` from [`scale_exponent`] every ratio is under two, so the sum is
/// under `4 * v.len()` and cannot overflow for any slice that fits in memory;
/// nor can it be zero, since the largest component divides to at least one.
/// Both bounds hold however far the unscaled squares would have left the scalar.
/// Dividing by a power of two is exact, so this is not an approximation of the
/// unscaled sum of squares but that same sum with its exponent moved by
/// `-2 * exp`.
///
/// Spelled out rather than through `Iterator::sum`, which would cost a `Sum`
/// supertrait on [`Real`] for a syntax preference.
fn scaled_sum_of_squares<C: Real>(v: &[C], exp: i32) -> C {
  let scale = C::ONE.ldexp(exp);
  let mut sum = C::ZERO;
  for &x in v {
    let r = x / scale;
    sum = sum + r * r;
  }
  sum
}

/// The L2 norm of `v`, via [`Real::sqrt`] (core has no `f32::sqrt`).
///
/// Taken against `v`'s own power-of-two scale and shifted back afterwards, so
/// the sum of squares never leaves the compute scalar even when the norm itself
/// would: `[f64::MAX, f64::MAX]` has norm `sqrt(2) * f64::MAX`, which overflows,
/// yet this returns it (as an overflow to infinity) rather than by squaring into
/// one. The result is `sqrt(sum(v_i^2))` to the bit wherever that direct
/// computation was valid, and the vector's actual norm wherever it was not.
///
/// [`SaliencyWeighted`] weights each window by this, so a window whose norm is
/// unrepresentable still contributes its direction: the shared magnitude divides
/// back out in the final renormalization, and only the ratios between norms —
/// what the weighting means — survive.
///
/// A vector with no scale returns its own largest magnitude: zero when it is all
/// zero, and the non-finite component itself when one is infinite. Each is
/// already the weight — and then the accumulator — that the caller must reject.
///
/// `pub(crate)` so [`VectorEma`](crate::smooth::VectorEma)'s streaming
/// determinacy gate measures against the *same* scale-aware norm this module's
/// gate does, rather than a second spelling that could drift from it.
pub(crate) fn l2_norm<C: Real>(v: &[C]) -> C {
  let Some(exp) = scale_exponent(v) else {
    return max_magnitude(v);
  };
  scaled_sum_of_squares(v, exp).sqrt().ldexp(exp)
}

/// Normalize `v` to unit L2 length in place.
///
/// # Errors
///
/// [`WinditError::NonFinite`] if `v` cannot be normalized to a finite unit
/// vector: it is all zero, or some component is not finite.
///
/// `pub(crate)` so [`VectorEma`](crate::smooth::VectorEma) renormalizes each
/// emitted window through this exact routine — the streaming sibling's
/// "renormalized" is the same arithmetic as the fold's, not a re-derivation of
/// it.
pub(crate) fn l2_renorm<C: Real>(v: &mut [C]) -> Result<(), WinditError> {
  // The one rejection, and a property of the input rather than of some
  // intermediate leaving range: an all-zero vector has no direction to normalize
  // to, and neither has one with an infinite component.
  let Some(exp) = scale_exponent(v) else {
    return Err(WinditError::NonFinite);
  };
  let scale = C::ONE.ldexp(exp);
  // `unit` is the norm divided by `scale`, which puts it in [1, 2*sqrt(len)]:
  // always representable, and always at least one, even for a vector whose norm
  // is not representable at all (`[f64::MAX, f64::MAX]`). Dividing by `scale` and
  // by `unit` separately is what avoids ever forming that norm. Both divisors are
  // exact power-of-two relatives of the direct computation's, so the quotient is
  // `v_i / norm` to the bit wherever the direct computation was valid.
  let unit = scaled_sum_of_squares(v, exp).sqrt();
  // Only a NaN component reaches this: it never becomes the maximum, so it
  // passes the scale check above and surfaces in the sum instead. Nothing is
  // written until every rejection is past, so a rejected vector is left as it
  // was.
  if !unit.is_finite() {
    return Err(WinditError::NonFinite);
  }
  for x in v.iter_mut() {
    *x = (*x / scale) / unit;
  }
  Ok(())
}
