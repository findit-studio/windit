# Changelog

All notable changes to `windit` are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 0.3.0

One new smoother — the streaming sibling of an aggregation policy that already
existed — and **two breaking changes of one class**: a number the fold multiplies
by must not be resolved more coarsely than the fold. An EMA smoothing factor is
now the compute scalar itself rather than an `f32` widened into it, and a window
coverage is `f64` rather than `f32` from `Span::coverage` all the way to the
weight. The new smoother was written with the first defect, the aggregation
policy has carried it since `0.2.0`, and the coverage channel has carried the
second since `0.1.0`; all of it is fixed here rather than a piece now and a piece
at a later major.

**The two breaks have different audiences, and are listed separately below.** The
coefficient change breaks no downstream implementor — it reaches callers as a
silently different number. The coverage change alters the signature of an
*object-safe trait method*, so **every downstream `AggregatePolicy` stops
compiling**, and it moves `CoverageWeightedMean`'s output for every caller: a
second re-measure in one release.

`cargo semver-checks` reports **neither** break, and that is worth stating plainly
rather than leaning on: forced to evaluate the release as a *patch* against
`0.2.0` it returns `223 checks: 223 pass, 30 skip` and `no semver update
required`, exit `0`. It models items appearing and disappearing, and nothing here
disappears. A type parameter with a default keeps `EmaRenormalized` nameable; a
trait method that keeps its name and arity is still present whatever its
parameter types are; a public method that keeps its name is still present
whatever it returns; and a supertrait added to a sealed trait is invisible.
Changed *parameter* types, changed *return* types, changed public *field* types,
and added supertraits have no lint at all between them.

That last point settles a question this release opened. When only the coefficient
had moved it was reasonable to suppose the tool was silent because a changed
*field* type is an unusual thing to lint, and that a changed *method signature*
on a public trait would be a different story. It is not: the same run, with
`Span::coverage`'s return type and `AggregatePolicy::aggregate_values`'s parameter
type both changed, still passes every check. The tool's verdict is evidence about
its own coverage, not about this release.

Both breaks were verified the only way they can be, by compiling a `0.2.0`-shaped
consumer against this crate:

- `EmaRenormalized::new(cfg_alpha)` with `cfg_alpha: f32` fails with
  `the trait bound f32: Real is not satisfied`.
- The same three-window fold at a bare `0.3` literal returns
  `[0.9664376815532346, 0.2569011632398902]` at `0.2.0` and
  `[0.9664376833865183, 0.2569011563432517]` here, while
  `EmaRenormalized::new(f64::from(0.3f32))` reproduces the `0.2.0` vector bit
  for bit.
- A custom `impl AggregatePolicy for FirstWindow` taking `_coverages: &[f32]`
  fails with ``method `aggregate_values` has an incompatible type for trait``,
  quoting `expected signature fn(&FirstWindow, &[&[f64]], &[f64], _) -> _`
  against `found signature fn(&FirstWindow, &[&[f64]], &[f32], _) -> _`.
- A four-window plan (`WindowOptions::new(3)` over `10` elements: three full
  windows and a one-element ragged tail) aggregated with `CoverageWeightedMean`
  returns `[0.9938837343123988, 0.11043152932582692]` at `0.2.0` and
  `[0.993883734673619, 0.11043152607484655]` here. The coefficient work earlier
  in this release does not move that fixture: `Span::coverage` and
  `CoverageWeightedMean`'s weight are byte-identical between `0.2.0` and the
  point in this release where the coverage widening lands, so `0.2.0` and
  "before this change" are the same baseline for it.

### Breaking

- **`aggregate::EmaRenormalized` and `smooth::VectorEma` carry a `C: Real`
  coefficient, defaulted to `f64`, instead of an `f32` one.**

  ```text
  0.2.0:  pub struct EmaRenormalized       { alpha: f32 }
          pub const fn new(alpha: f32) -> Self
          pub const fn alpha(&self) -> f32

  0.3.0:  pub struct EmaRenormalized<C: Real = f64> { alpha: C }
          pub const fn new(alpha: C) -> Self
          pub const fn alpha(&self) -> C
  ```

  `VectorEma` is new in this release and takes the same shape. `smooth::Ema`,
  `smooth::CadenceEma`, and the `segment` thresholds are **unchanged**, because
  their value type really is `f32` and a coefficient should match what it
  multiplies. What was wrong was a coefficient *narrower* than the arithmetic it
  drives: `Real` has one implementor, `f64`, and every storage scalar computes in
  it, so the EMA's weights, products and compensated sum were all `f64` around
  one `f32` constant. No `f32` expresses `1 - 2^-30` — its nearest is exactly
  `1.0`, which is a pass-through and not a slow filter — and the `f32` grid is
  `2^-24` apart relatively where the fold rounds at `2^-53`.

  **What a `0.2.x` caller edits.** Usually nothing, and that is the hazard:

  - `EmaRenormalized::new(0.3)` still compiles — and now means a **different
    number**. `0.3f32` widened to `0.30000001192092896`; `0.3f64` is
    `0.29999999999999999`. The weights change in the eighth significant digit,
    so for anyone who has pinned outputs bit-for-bit or tuned against a
    threshold this is a **re-measure, not a recompile**. To keep the old
    behaviour exactly, write `EmaRenormalized::new(f64::from(0.3f32))`; to take
    the value you meant, change nothing.
  - An `f32` *variable* — `EmaRenormalized::new(cfg.alpha)` where
    `cfg.alpha: f32` — stops compiling, with `f32: Real` unsatisfied. That one
    the compiler catches: write `f64::from(cfg.alpha)`.
  - `AggregatePolicyKind::Ema { alpha }` keeps a concrete field, now `f64`. The
    serde wire form is unchanged (a JSON number carries no width), so
    configuration files are unaffected — but a configured `0.3` that used to
    round through `f32` now reaches the fold at full `f64` precision, which is
    the same re-measure as above.
  - A custom `impl AggregatePolicy for MyPolicy` is untouched **by this break** —
    it changes no trait and no signature. The coverage break below does touch it.

- **`plan::Span::coverage` returns `f64`, and
  `aggregate::AggregatePolicy::aggregate_values` takes `coverages: &[f64]`.**

  ```text
  0.2.0:  pub fn coverage(&self) -> f32
          fn aggregate_values(&self, embeddings: &[&[C]], coverages: &[f32], dim: usize)
              -> Result<Vec<C>, WinditError>

  0.3.0:  pub fn coverage(&self) -> f64
          fn aggregate_values(&self, embeddings: &[&[C]], coverages: &[f64], dim: usize)
              -> Result<Vec<C>, WinditError>
  ```

  This is the same defect as the coefficient above wearing a different
  provenance. A coverage was typed by where it came from — a window-geometry
  fraction rather than an embedding value, therefore `f32` — when what types a
  number is **what it multiplies**. `CoverageWeightedMean` multiplies an
  embedding by it, inside an `f64` fold, and computing it in `f32` from two
  `usize`s cost two distinct things:

  - **The operands rounded before the division.** `f32` holds every integer only
    to `2^24`. A window of `16_777_217` narrowed to `16_777_216`, so a tail one
    element short of it divided out to exactly `1.0` — a ragged tail
    indistinguishable from a full window, in a crate whose front page asserts
    `spans[2].coverage() < 1.0`.
  - **The quotient landed on the `f32` grid**, `2^-24` apart relatively where the
    fold rounds at `2^-53`. Spans `(len 8388607, window 16777213)` and
    `(len 8388608, window 16777215)` — every operand exactly representable in
    `f32`, so nothing rounded going in — have true coverages `3.6e-15` apart, 32
    `f64` ulps and about `6e-8` of an `f32` ulp. Both arrived at the fold as
    `0.50000006`, and two different windows weighed the same. Each defect has a
    falsifier in the suite.

  **What a custom policy's diff is.** One character, and the compiler points at
  it. Nothing else about an implementor changes — not the trait, not its name,
  not its arity, not the compute-scalar type parameter:

  ```diff
   impl AggregatePolicy for FirstWindow {
     fn aggregate_values(
       &self,
       embeddings: &[&[f64]],
  -    _coverages: &[f32],
  +    _coverages: &[f64],
       dim: usize,
     ) -> Result<Vec<f64>, WinditError> {
  ```

  A policy that only length-checks the slice, as most do, stops there. A policy
  that *reads* it drops whatever widening it was doing: `C::from_f32(coverages[i])`
  becomes `C::from_f64(coverages[i])`, and a policy already computing in `f64`
  drops the call entirely. A policy that *builds* a coverage slice to hand to
  `aggregate_values` retypes its literals; `[1.0_f32; n]` becomes `[1.0; n]`.

  **What an existing caller's numbers do.** They move — this is the release's
  second re-measure, and unlike the first there is no source edit that opts out
  of it, because the coverage is derived by the planner rather than configured by
  the caller. A one-element ragged tail in a window of three is the commonest
  ragged geometry there is, and its coverage goes from `0.3333333432674408`
  (`1/3` rounded through `f32`) to `0.3333333333333333`: a relative move of
  `3.0e-8`, the **eighth significant digit**, exactly the size of the
  coefficient change above. Through a whole aggregation the move survives at the
  same order. The four-window fold quoted earlier returns

  ```text
  0.2.0:  [0.9938837343123988,  0.11043152932582692]
  0.3.0:  [0.993883734673619,   0.11043152607484655]
  ```

  — a relative change of `3.6e-10` in the dominant component (10th significant
  digit) and `2.9e-8` in the one the tail weight actually steers (8th). Anyone
  who has pinned an aggregate bit-for-bit, or tuned a similarity threshold
  against one, should **re-measure rather than recompile**. Callers whose plans
  have no ragged tail are unaffected: a full window's coverage is exactly `1.0`
  in both widths.

  **One consequence in the input domain.** The accepted range is unchanged in
  words — a coverage must still be a finite fraction in `[0, 1]` — but `f64`
  reaches `2^-1074` where `f32` stopped at `2^-149`, and that bound was doing
  quiet work: it kept `CoverageWeightedMean`'s weight bounded below, so its
  products were guaranteed normal. It no longer is, and the policy joins
  `EmaRenormalized` in the regime the determinacy gate's `MIN_GATE_THRESHOLD`
  floor decides — a fold whose whole mass sits under the floor is `NonFinite`
  rather than a direction manufactured out of subnormals. The `aggregate` path
  cannot reach it (`Span::coverage` is `len / window` with `len >= 1`, so at
  least `1 / usize::MAX`); only a caller assembling its own coverage slice for
  `aggregate_values` can. The module's Input domain note now says which policies
  the domain bounds below and which it does not, and a test pins the boundary.

- **`scalar::Real` gains `from_f64`, a `'static` bound, and a `Debug`
  supertrait.** The trait is sealed, so no downstream impl can break and all
  three are additive for callers. `from_f64` widens an `f64` value the fold will
  multiply an embedding by — an EMA smoothing factor, a `Span::coverage`, or the
  wire alpha `AggregatePolicyKind::into_policy` reads before any compute scalar
  exists — into the compute domain (the identity for `f64`). `'static` is a fact
  about every implementor — an owned, borrow-free arithmetic type — spelled so
  that `Box<dyn AggregatePolicy<C>>` needs no re-declaration at each use site.
  `Debug` is a fact about every implementor too (they are core floats), and it is
  what lets `smooth::VectorEmaState`'s hand-written `Debug` report the
  coefficient it was configured with: without it a `ComputeOf<E>` could not be
  formatted at all, so the impl could only describe buffer shapes. It still
  describes the buffers by shape rather than dumping one component per embedding
  dimension — that part was a choice, not the missing bound.

- **`Real::from_f32` is down to one job.** It widens the determinacy gate's own
  dimensionless constant, and nothing else: a smoothing factor stopped arriving
  through it, and so did a `Span::coverage`. Nothing the fold multiplies an
  embedding by is resolved at `f32` any more, and its documentation says so.

### Source compatibility

The glob-collision note this release inherits still applies, since it also adds
public names:

- **Guaranteed.** The prelude is unchanged, so `use windit::prelude::*;`
  resolves exactly as it did at `0.2.0`, including alongside a glob of the
  dependent's own module that happens to export a `VectorEma`.
- **Not guaranteed, and not guaranteeable by any release that adds a public
  name.** A dependent that globs the *module* — `use windit::smooth::*;`
  together with `use their_own::*;` where the other glob also supplies
  `VectorEma` — compiles at `0.2.0` and fails here with `E0659`, reported at
  the use site. Adding a public item anywhere carries that hazard; it is not
  special to the prelude, which is why the paragraph below argues about
  *exposure* rather than about safety. If your build globs `windit::smooth`,
  check for the collision before taking this release.

### Added

- **`error::WinditError::EpochTooLong`**: a `VectorEma` epoch reached
  `VectorEma::MAX_EPOCH_STEPS` charging steps, past which the determinacy gate's
  error bound is no longer proven. `WinditError` is `#[non_exhaustive]`, so the
  variant is additive; `cargo semver-checks` agrees.

- **`smooth::VectorEma`** (and its state, `smooth::VectorEmaState`): a
  component-wise exponential moving average over an *embedding*, L2-renormalized
  at every window — the streaming, span-preserving sibling of
  `aggregate::EmaRenormalized`. Where the aggregate folds a finished slice to one
  point, this rewrites one window in / one window out with the input span intact,
  so a per-window embedding stream can be denoised without being collapsed. That
  was the one shape the crate could not express: `smooth` had only the `f32`
  scalar low-passes, and a downstream `impl Smoother<TheirEmbedding>` is an
  orphan impl.

  It is generic over `windowed::Vector` — the carrier trait the aggregation half
  already reads embeddings through — so a consumer's own embedding type flows
  through `Windowed<E>` with no conversion at any window, and the crate gains no
  new public trait. The accumulator is carried in the embedding's compute domain
  (`ComputeOf<E>`, `f64` for every shipped scalar) and read through
  `Vector::compute_components`, so quantized storage with no dequantization
  override fails closed with `MissingDequantization` exactly as aggregation does.

  The renormalization is applied to an emitted *copy*; the accumulator stays raw.
  That is what makes window `i` emit the direction `EmaRenormalized` folds over
  the prefix `[0..=i]`, and a differential test pins it at five smoothing factors
  over every prefix and both storage widths. Renormalizing the accumulator in
  place would be a different filter with different weights.

  The equivalence is exact in exact arithmetic and **not bit-exact in
  floating point**, and cannot be made so: the aggregate materializes each weight
  and folds the prefix with Neumaier compensation, this carries a two-term
  recurrence. What the tests establish across prefix lengths, smoothing factors
  and storage widths is that determinate prefixes agree to within the sum of the
  two error bounds and that neither side fabricates a direction out of
  cancellation. **Near either threshold the verdicts can differ in *either*
  direction**, and no ordering between the two thresholds is claimed: a
  three-window prefix at `alpha = 0.3` is emitted by the aggregate and refused
  here, and the exact-bit case `alpha = 0x3e99999a` over the one-dimensional
  windows `0x3f0ca8ca28200000`, `0xbf20b7cb3226ac2d`, `0xbc2767b60c530643` puts
  both accumulators on `0x3c0c160dbb1cff8d` with the two thresholds one ulp
  apart the other way, so this side emits where the aggregate refuses. Both are
  regression tests.

  **The gate's contract is self-contained: it refuses when the accumulator is
  within its own error bound of zero.** That is provable from this type's own
  recurrence and says nothing about how another code path rounds; whether the
  two siblings agree on a given prefix is measured, not promised. (An earlier
  draft of this entry claimed this smoother was never the less conservative of
  the two. The induction behind it compared against the mass an *ideal* fold
  would accumulate, and `EmaRenormalized` rematerializes its weights instead —
  so the ordering does not survive its actual roundings.) The gate keeps the
  aggregate's shape, constant and absolute floor
  (`16 * EPSILON * ||M|| + MIN_GATE_THRESHOLD`) and its scale-aware
  renormalization is the aggregation half's own routine, but the mass `M` is the
  recurrence's, which carries the damped rounding of every step rather than the
  term magnitudes of one — the error a recurrence propagates is not the error a
  fold commits.

  A step that rounds nothing charges nothing: `alpha = 0` (an exact hold) and
  `alpha = 1` (an exact pass-through) accumulate no mass at all. Without that
  rule a held seed still charged `|s_0|` a push, so the threshold grew linearly
  and reached the seed's own magnitude after `2^48` pushes, after which the gate
  refused the hold forever. The mass does still grow at every other coefficient,
  and the horizon where it would overtake a determinate accumulator
  (`alpha < 2^-48`, upward of `2^48` pushes) is documented as reachable in
  principle rather than argued away. The published error bound now carries the
  absolute term the subnormal range needs — `|e_t| <= 2u * M_t + t * 2^-1074` —
  since a purely relative bound is false there; the gate's `MIN_GATE_THRESHOLD`
  floor dominates that term for every epoch with `t * sqrt(dim) <= 2^74`, so no
  verdict ever turned on it. `alpha` clamps into `[0, 1]` at construction, NaN to `0.0`,
  exactly as `smooth::Ema` does — the smoother idiom, not the aggregate's
  deferred `AlphaOutOfRange`. The clamp is three comparisons rather than
  `f64::clamp` (which propagates NaN, where `Real` offers ordering but no
  `is_nan`), and that costs `VectorEma::new` its `const`: a trait method cannot
  run in a `const fn`. `EmaRenormalized::new` stores without comparing and stays
  `const`.

  The coefficient is `C: Real` defaulted to `f64`, and the `SmoothPolicy` impl is
  for `VectorEma<ComputeOf<E>>` — so the coefficient sits in the same domain as
  the accumulator it multiplies by construction rather than by convention, while
  `VectorEma::new(0.3)` still needs no turbofish. `MAX_EPOCH_STEPS` is stated on
  `VectorEma<f64>` rather than on the generic impl: the horizon is derived from
  the compute domain's own `EPSILON`, so a second `Real` would carry a different
  number, and a `const` on the generic impl could not be reached through the bare
  path `VectorEma::MAX_EPOCH_STEPS` (a type parameter's default does not apply to
  an associated-item path).

  **The epoch is bounded, because `M` is floating point too.**
  `VectorEma::MAX_EPOCH_STEPS` (`2^50` charging steps) is enforced: the step that
  would carry an epoch past it is refused with `WinditError::EpochTooLong`,
  before the accumulator is touched, and so is every push after it until a
  `reset`. `M` dominates the accumulator's error only while `M` itself is
  accumulated faithfully, and its three roundings a step are to nearest, so the
  computed mass can sit *below* the exact one: `M^_t >= (1 - u)^(2t + 1) * M^ex_t`,
  which the gate's factor of sixteen absorbs only while
  `(1 - u)^(2t + 1) >= 1/16` — about `2^53.4` charging steps. Past there the
  guarantee fails, and demonstrably: at `alpha = 2^-54` the complement is exactly
  `1`, so an accumulator of `2^-24` absorbs every `2^-78` injection while `M`,
  charged exactly `2^-24` a step, reaches `2^29` after `2^53` steps and then
  **stagnates** on the round-to-even tie. `2^60` such steps and then `2_129_920`
  pushes of `-2^15` leave the exact recurrence at zero and the accumulator at
  `-2^-18` against a threshold of `2^-19` — an emitted direction for a prefix
  that exactly cancels, with the published `2u * M` bound broken by a factor of
  32. Every input there is finite, in domain, and exactly representable, so the
  regime is refused rather than assumed away, and the enforced `2^50` sits three
  binary orders inside the proven range (and inside the subnormal term's `2^74`
  reach). Only **charging** steps count, so `alpha = 0` and `alpha = 1` still
  hold their seed for an unbounded epoch: the exact-step exemption above keeps
  its liveness, rather than being undone at a further horizon.

  Input domain: the aggregation one, unchanged (`aggregate`'s *Input domain*
  note) — every component finite and either zero or between
  `Real::MIN_AGG_MAGNITUDE` and `Real::MAX_AGG_MAGNITUDE`. The two-term
  recurrence would not need it, but the determinacy gate's mass is an `n`-term
  geometric fold over the epoch, which is exactly what that domain exists to keep
  inside `f64`. No `f32`-storage embedding can reach the boundary; only an
  `f64`-storage one can.

  Errors, all raised before the accumulator is written so a refused push is a
  no-op: `DimMismatch` for a width that changed mid-epoch, `NonFinite` for a
  non-finite component (the scalar `Ema` absorbs one and poisons; this one
  refuses it), `MagnitudeOutOfRange` for a component outside the domain above,
  `Empty` for a zero-width embedding, `MissingDequantization` for raw
  quantization codes, `EpochTooLong` for an epoch past `MAX_EPOCH_STEPS`, and
  `AllocFailed` for a refused buffer. The one deliberate
  exception is a window whose *output* fails the determinacy gate: it has still
  advanced the accumulator, because it was a real observation and the prefix the
  aggregate would fold includes it.

  **Not re-exported from the prelude.** Adding a name to a glob prelude and
  adding one to a module carry the same hazard — `E0659` at a downstream use
  site where two globs both supply the name — so keeping it out of the prelude
  buys *less* exposure, not safety, and the release note above no longer claims
  otherwise. The difference is real all the same: `use windit::prelude::*;` is
  the import this crate documents and asks every dependent to write, while
  `use windit::smooth::*;` is suggested nowhere. Removing the larger exposure
  costs a dependent one line, `use windit::smooth::VectorEma;`. Both breaks were
  reproduced directly rather than argued from the language reference;
  `cargo-semver-checks` passes either way, because it does not model downstream
  glob resolution. Whether the name joins the prelude is a decision separate from
  this release's coefficient change, and is not taken here. Unlike the three scalar smoothers it
  gates on `alloc` rather than living in the featureless core tier: its state is
  one accumulator component per embedding dimension, which is a heap buffer and
  not the O(1) that tier admits. The buffers are grown on the first push of an
  epoch and reused by every push after it; `reset` keeps their capacity, so a
  discontinuity costs no allocation.

### Documented

- `Smoother::push`'s error note no longer implies that reading no spans makes a
  stage infallible, and `SmoothPolicy::smooth`'s "none for the shipped built-ins"
  is corrected. Both statements predate a fallible smoother existing.

## 0.2.0 - 2026-07-25

The temporal half of the crate — smoothing, gating, segmentation — is rebuilt
around streaming state. In 0.1.x each of those was a batch policy that took a
whole `&[Windowed<V>]` and returned a `Vec`; state lived inside one call and
could not cross it, so there was no way to decode a live stream and no way to
know that a chunked decode matched a whole-sequence one. 0.2.0 splits every
policy into a *configuration* (the `Ema`/`Threshold`/… value you already build)
and a *state* (`Smoother`/`Gate`) the configuration constructs, drives the batch
conveniences through that same state, and adds the `Decoder` that composes the
three stages into one pipeline. Batch output is now equal to streaming output by
construction rather than by hope.

Every breaking change below carries its migration. The two least mechanical are
`smooth::Hysteresis` becoming a `bool`-typed gate in `segment` (**Changed**, with
the `0.0`/`1.0`-float recipe spelled out) and the batch drivers becoming
fallible.

### Added

- **Streaming state traits, in the featureless core tier.** `smooth::Smoother<V>`
  and `segment::Gate<V>` are one-window-in / one-out state machines — `push`,
  `reset`, `discontinuity` — that hold O(1) state and allocate nothing. The
  configuration traits `SmoothPolicy`/`GatePolicy` construct them through
  `smoother()` / `gate()` and drive them for the batch conveniences, so a policy
  is now described once and reachable both ways.
- **The span contract is stated on both traits.** Spans arrive in ascending
  `Span::start` order, equal starts admitted, **and that is the only ordering
  guaranteed**: ends are *not* monotone. Nested and overlapping spans are legal,
  so a later span may end before one already seen, and any stage keeping a
  temporal horizon must fold it by maximum (`horizon = max(horizon,
  span.end())`) rather than read the current span's end. A strictly backward
  start is a contract violation, reported as the new
  `WinditError::NonMonotonicSpan`.
- **`segment::Segmenter` and `segment::SegmentTail`**: the incremental
  segmentation core. `push(active, span)` returns any `Range` that push
  finalized, `finish` drains the at-most-two pending ranges as a fixed-size
  iterator, `discontinuity` drains them and re-arms for a new epoch, `reset`
  discards. Bounded and zero-allocation (pinned by an allocation-regression
  test), and it is what every batch driver in the crate now runs on — so batch
  equals streaming by construction rather than by a parallel implementation.
- **`decode::Decoder<S, G, V>` and `decode::Step`**: the `Smoother` → `Gate` →
  `Segmenter` composition as one object, with the same
  `push`/`finish`/`discontinuity`/`reset` lifecycle. `Step` carries both output
  planes for a window: `active` (the gate's immediate causal decision) and
  `finalized` (a `Range` no later input can change). Concatenating the finalized
  ranges with the `finish` tail reproduces the batch composition exactly; the
  causal plane deliberately promises no batch parity. Entirely featureless — the
  module allocates nothing.
- **`smooth::Identity`**: the explicit pass-through smoother, the semantic
  no-rewrite baseline, generic over any `V`.
- **`smooth::CadenceEma`**: an EMA whose time constant is denominated in input
  *elements* rather than in pushes. Each push derives its own coefficient from
  the actual span distance, `alpha = 1 - exp(-delta / tau)`, so one `tau` yields
  the same smoothing at any hop — regular or irregular — where a bare per-push
  `Ema` `alpha` does not. `tau` is accepted in `(0, CadenceEma::MAX_TAU]`, a
  bounded domain chosen so the type's accuracy figures hold across all of it:
  `new` panics outside that interval and `try_new` reports the new
  `WinditError::TimeConstantOutOfRange` instead. Over differences the emitted
  `f32` can express the invariance holds at every accepted configuration; below
  that resolution it stays contrast-dependent — see the *Fine cadences* bullet
  on the type.
- **`segment::Vote`**: an N-of-M gate, active once at least `need` of the last
  `of` comparisons `value >= thr` were true, over a one-machine-word ring of up
  to 64 votes. `new` panics unless `1 <= need <= of <= 64`; `try_new` reports the
  new `WinditError::InvalidVote`. It counts *windows*, not elements, so its
  physical meaning changes with the hop — stated plainly on the type, with
  `Dwell` named as the element-denominated alternative.
- **`segment::Dwell` and `segment::Hangover`**: gate combinators, generic over
  the inner gate *and* over `V`. `Dwell` is on-delay — it suppresses the inner
  gate's `true` until the inner gate has been continuously active for `confirm`
  input elements. `Hangover` is off-delay — once active it holds `true` while the
  gap since inner-active coverage stays strictly below `hold` elements. Both are
  element-denominated, so they are portable across hops. Nesting them over a gate
  (`Hangover::new(Dwell::new(Vote::new(3, 5, 0.5), 8), 16)`) is the canonical
  stack.
- **Boxed stages.** `impl<V, T: Smoother<V> + ?Sized> Smoother<V> for Box<T>` and
  the matching `Gate` impl, so a run-time-selected `Box<dyn Smoother<f32>>` or
  `Box<dyn Gate<f32>>` can be *held* as a pipeline stage rather than merely
  called through auto-deref. Behind `alloc`. Both forward `discontinuity`
  explicitly, so a concrete stage's override is not silently downgraded to
  `reset`.
- New `WinditError` variants: `NonMonotonicSpan { prev_start, start }`,
  `TimeConstantOutOfRange`, and `InvalidVote { need, of }`. The enum is
  `#[non_exhaustive]`, so the additions are not themselves breaking.

### Changed

- **The batch segmentation drivers are fallible.** `runs`, `longest_run`, and
  `runs_sorted` return `Result` instead of a bare value, because they now check
  the ascending-span contract (`NonMonotonicSpan`) and reserve their output
  through `try_reserve` (`AllocFailed`) instead of aborting the process.
  Output for in-contract input is unchanged, pinned by a differential test
  against the retained 0.1.2 implementation.

  ```rust
  // 0.1.x
  let speech: Option<Range> = longest_run(&frames, |&p| p >= 0.5, &opts);
  // 0.2.0
  let speech: Option<Range> = longest_run(&frames, |&p| p >= 0.5, &opts)?;
  ```

- **`SmoothPolicy` is now a factory, and its batch method is fallible.** The
  trait's one required item is `type Smoother: Smoother<V>` plus
  `fn smoother(&self) -> Self::Smoother`; `smooth` became a provided method with
  the signature `fn smooth(&self, seq: &[Windowed<V>]) -> Result<Vec<Windowed<V>>,
  WinditError> where V: Clone`, still behind `alloc`, still fresh state per call.

  ```rust
  // 0.1.x — callers
  let smoothed = Ema::new(0.2).smooth(&seq);
  // 0.2.0 — callers
  let smoothed = Ema::new(0.2).smooth(&seq)?;
  ```

  ```rust
  // 0.1.x — implementors wrote the whole-sequence loop
  impl SmoothPolicy<f32> for MyFilter {
      fn smooth(&self, seq: &[Windowed<f32>]) -> Vec<Windowed<f32>> { /* loop */ }
  }
  // 0.2.0 — implement the one-window step and the batch driver comes free
  impl Smoother<f32> for MyFilterState {
      fn push(&mut self, w: Windowed<f32>) -> Result<Windowed<f32>, WinditError> { /* step */ }
      fn reset(&mut self) { /* … */ }
  }
  impl SmoothPolicy<f32> for MyFilter {
      type Smoother = MyFilterState;
      fn smoother(&self) -> MyFilterState { /* … */ }
  }
  ```

- **`segment::SegmentPolicy` is replaced by `segment::GatePolicy`, and the
  morphology moved from the policy to the call.** A gate decides membership; the
  `SegmentOptions` that shape the accepted runs are now an argument, so one
  configured gate can be segmented under several morphologies.

  ```rust
  // 0.1.x
  let ranges = Threshold::new(0.5)
      .with_opts(SegmentOptions::new().with_min_len(2))
      .segment(&seq);
  // 0.2.0
  let ranges = Threshold::new(0.5)
      .segment(&SegmentOptions::new().with_min_len(2), &seq)?;
  ```

  Implementors migrate the same way `SmoothPolicy` implementors do: write
  `Gate::push` (one window in, one `bool` out) and `GatePolicy::gate`, and the
  `segment` batch driver is provided.

- **`Hysteresis` moved from `smooth` to `segment` and is now a `bool`-typed
  gate.** In 0.1.x it was a `SmoothPolicy<f32>` whose output was a `0.0`/`1.0`
  float sequence; in 0.2.0 it is a `GatePolicy<f32>` whose `Gate` yields `bool`.
  The latch transition is bit-for-bit the same — on at `>= on`, off strictly
  below `off`, hold in the half-open band `off <= value < on`, and the identical
  `NaN`/±inf table — only the output type and the module changed. Note that a
  `use windit::prelude::*` glob keeps compiling and now resolves `Hysteresis` to
  the gate, so the failure surfaces at the call, not the import.

  If you were segmenting the latched output, that is now one call:

  ```rust
  // 0.1.x
  use windit::smooth::Hysteresis;
  let latched = Hysteresis::new(0.6, 0.3).smooth(&seq);
  let ranges = runs(&latched, |&v| v >= 0.5, &opts);
  // 0.2.0
  use windit::segment::Hysteresis;
  let ranges = Hysteresis::new(0.6, 0.3).segment(&opts, &seq)?;
  ```

  If you genuinely wanted the `0.0`/`1.0` float sequence — to feed it somewhere
  that consumes scores — map the decision yourself. The crate deliberately no
  longer carries that role, so this is the whole recipe:

  ```rust
  use windit::prelude::*;

  let mut gate = Hysteresis::new(0.6, 0.3).gate();
  let mut latched: Vec<Windowed<f32>> = Vec::with_capacity(seq.len());
  for w in &seq {
      let active = gate.push(w)?;
      latched.push(Windowed::new(if active { 1.0 } else { 0.0 }, w.span()));
  }
  ```

  A one-shot `0`/`1` smoother wrapper is *not* equivalent to a gate in a
  `Decoder`: the gate feeds a `bool` plane straight to the segmenter, where the
  float detour would re-threshold it.

- **`Threshold` is slimmed to its cutoff.** `Threshold::with_opts` and
  `Threshold::opts` are gone; see the `GatePolicy` migration above. `thr()` is
  unchanged, and the raw-IEEE comparison semantics are unchanged.

- **The `smooth` and `segment` modules dropped their feature gate.** Both were
  behind `any(feature = "std", feature = "alloc")` in 0.1.x. Everything in them
  that allocates nothing — the traits, every gate and smoother configuration and
  state, `Segmenter`, `SegmentTail`, `Range`, `SegmentOptions` — now compiles
  under `--no-default-features`. The `Vec`-returning drivers (`runs`,
  `longest_run`, `runs_sorted`, `SmoothPolicy::smooth`, `GatePolicy::segment`)
  stay behind `alloc`, gated inside the module. Purely additive: no existing
  build loses anything.

- **The prelude is reshaped to match.** The featureless block now carries
  `Smoother`, `SmoothPolicy`, `Identity`, `Ema`, `CadenceEma`, `Gate`,
  `GatePolicy`, `Threshold`, `Hysteresis`, `Vote`, `Dwell`, `Hangover`,
  `Segmenter`, `SegmentTail`, `Range`, `SegmentOptions`, `Decoder`, and `Step`;
  the `alloc` block keeps `runs`, `longest_run`, and `runs_sorted`.
  `SegmentPolicy` is gone from it.

### Removed

- **`segment::HysteresisSegment`.** The fused two-pass hysteresis segmenter is
  replaced by the ordinary gate composition, which is now equally single-pass:

  ```rust
  // 0.1.x
  let ranges = HysteresisSegment::new(0.6, 0.3).with_opts(opts).segment(&seq);
  // 0.2.0
  let ranges = segment::Hysteresis::new(0.6, 0.3).segment(&opts, &seq)?;
  ```

  Output is unchanged for every in-contract input, pinned by a differential test
  against the retained 0.1.2 implementation over fixed and randomized geometries.

- **`segment::SegmentPolicy`** — replaced by `segment::Gate` + `segment::GatePolicy`.
- **`smooth::Hysteresis`** — moved to `segment::Hysteresis` and retyped.
- **`Threshold::with_opts` and `Threshold::opts`** — morphology is a call
  argument now.

### Fixed

No 0.1.x program's *output* changes here: the numerics of everything 0.1.x
shipped are untouched, and every behavioural correction below is to a type
introduced in this release. They are recorded because each states a contract the
crate is now held to, and the first describes a regime an existing `Ema` user can
be in today without knowing it. The last four entries correct *published claims*
and the domain they are quantified over rather than behaviour — the code they
describe was already right — and each is now pinned by a test that sweeps the
accepted domain to its edges and fails if the claim is false.

- **`smooth::Ema`'s behaviour at a sub-epsilon `alpha` is now documented, and it
  is not a hold.** At an `alpha` at or below `2^-25` (~3e-8), `1 - alpha` rounds
  to exactly `1.0` in `f32`. That deletes the decay term but leaves the
  `alpha * x` injection, so the recurrence degenerates from a weighted average
  into the biased accumulator `s <- s + alpha * x`: it moves only in the
  direction of `sign(x)`, never *toward* the input, climbs from a `0.0` seed in
  exact steps of `alpha * x`, and stalls at `alpha * x * 2^24` without ever
  reaching `x`. It does genuinely hold from that stalling magnitude upward, and
  because `s_0 = x_0` seeds a steady signal there, a constant stream still looks
  like a clean hold — which is what made "it holds" a plausible reading. The
  numerics are identical to 0.1.x; 0.1.x simply left the regime unstated. Reach
  for `CadenceEma`, whose `f64` accumulator pushes the same degeneracy 29 binary
  orders further out — its `1 - alpha` collapses at `2^-54`, not at `2^-25`
  — if you need to work down there.
- **`segment::Dwell` folds its confirmation horizon by maximum.** Confirming
  against the *current* span's end let an on-delay gate deactivate
  mid-activation: with `confirm = 10`, the spans `[0, 10)` then `[1, 2)` and the
  inner gate active throughout, the gate emitted `true` then `false` while the
  inner gate never released. Folding the run's coverage horizon by `max` makes a
  confirmed run stay confirmed — a nested or overlapping span can never retract
  an activation — and makes `Dwell` symmetric with the same correction
  `Hangover` carries for its own horizon.
- **`smooth::CadenceEma` keeps its coefficient at a fine cadence.** Two
  independent losses were closed. The coefficient is derived as
  `-expm1f(-delta / tau)` rather than the literal `1 - expf(-delta / tau)`, which
  loses every bit of the ratio once `expf` rounds to `1.0` — below `2^-25` for
  `f32` — so a then-accepted `tau = 1e8` at `delta = 1` used to derive
  `alpha == 0.0` and freeze the filter entirely, while the same signal sampled
  at `delta = 100` moved normally. (`1e8` is above the ceiling this release
  ships and no longer constructs; the loss it exposed was real at every `tau`.)
  And the recurrence is accumulated in `f64` with only the emitted value
  rounded to `f32`, because applying an exact small coefficient to an `f32`
  state made a state near `1.0` a fixed point: at `tau = 4e7` a unit cadence
  decayed not at all where one 40,000-element step (0.001 `tau`) covering that
  same distance reached ~0.9990005. Both were the type's defining property —
  that the result does not depend on how finely the signal is sampled — failing
  in a reachable regime.
  Invariance is documented as a bound on `alpha * rho` (`rho` the contrast
  `|x - s| / |s|`), not on `alpha` alone — specifically `alpha * rho > 2^-50`,
  equivalently `alpha * |x - s| > 4 * ulp(s)`. Its `alpha > 2^-26` corollary,
  over differences the emitted `f32` can express, is *unconditional* on the
  accepted domain, because that is exactly what `MAX_TAU` guarantees. The
  constant accounts for all three roundings the recurrence performs —
  `1 - alpha`, the product `(1 - alpha) * s`, and the final sum — rather than
  the single fused step earlier drafts assumed, and it is deliberately looser
  than both the derivation (`(1.5 + alpha) * ulp(s)`) and an adversarial search
  over the accepted domain (~1.4e8 probes above the bar; worst absorption under
  one `ulp(s)`).
- **`segment::Dwell` with `confirm == usize::MAX` no longer activates.** The
  configuration is documented as never confirming, but the test was
  `horizon - origin >= confirm`, and the widest run a `Span` pair can describe —
  `[0, 1)` folded with `[usize::MAX - 1, usize::MAX)` — reaches
  `horizon - origin == usize::MAX` and met it exactly. The sentinel is now
  suppressed outright. `Hangover`'s mirror-image `hold == usize::MAX` needed no
  change: its test has the opposite sense (`gap < hold`) and `Span`'s own
  invariants cap the gap at `usize::MAX - 2`; that slack is now pinned by a test
  rather than left implicit.
- **Two published accuracy figures for `smooth::CadenceEma` were measured and
  corrected.** The absorption bound above was `ulp(s) / 2` with an
  `alpha > 2^-29` corollary, both falsified by a non-dyadic retained state where
  the recurrence's two separately rounded products absorb a step of
  `0.77 * ulp(s)`. And the one-`tau` decay was claimed accurate "within one" ulp
  of `exp(-1)`; it is within two — `tau = 14` and `tau = 238` both land exactly
  two representable values below — so the published figure is now four, and the
  `1e-6` tolerance that accompanied the claim (over 33 ulps at that magnitude,
  far too slack to enforce it) is replaced by exact ULP-distance assertions
  swept across the `tau` range.
- **`smooth::CadenceEma` now accepts a bounded `tau` domain,
  `(0, CadenceEma::MAX_TAU]`.** `MAX_TAU` is `2^26 - 4` elements — the largest
  `f32` strictly below `2^26` — and `try_new` reports
  `TimeConstantOutOfRange` above it, where it used to accept every positive
  finite `f32`. The ceiling is derived from the guarantees rather than picked
  for roundness: half an `f32` ulp is `2^28` `f64` ulps at the same magnitude,
  so `alpha > 2^-26` is precisely the condition that lifts every difference the
  emitted `f32` can express above the `4 * ulp(s)` absorption bar, and `MAX_TAU`
  is the largest `f32` whose `delta = 1` coefficient still clears it (one `f32`
  step further out, at `tau = 2^26`, the coefficient is exactly `2^-26` and the
  product lands *on* the bar instead of above it).

  This is a contract restriction, not a bug fix, and it is what makes the
  accuracy figures on the type true of *everything it admits* rather than of the
  range they were measured over. Unbounded acceptance falsified them at the
  edges: at `tau = 2^55` the `f64` `1 - alpha` rounds to exactly `1.0`, so a unit
  cadence cannot move a state of order `1` at all, while the "within four ulps of
  `exp(-1)` over one `tau`" figure says otherwise by millions of ulps.

  The ceiling is drawn at the accuracy boundary rather than at that degenerate
  one, and the two lie 28 binary orders apart. A rejected `tau` is not an inert
  filter: the first one, `2^26`, applies exactly `2^-26` per unit step, moving a
  state seeded at `0.0` to exactly `2^-26` and decaying a state of `1.0` by the
  same amount. It is rejected because that coefficient lands *on* the
  `4 * ulp(s)` absorption bar instead of above it, so the unconditional
  statements stop being provable of it — not because it stops moving. Freezing
  needs both a far larger `tau` (from `2^54`, where the `f64` `1 - alpha` is
  exactly `1.0` and the recurrence keeps no decay term) and a state large enough
  to absorb `alpha * x`; a state of `0.0` still moves even there. Nothing usable
  is excluded: `2^26` elements is over a week of audio at a 10 ms hop.
  `CadenceEma` is new in this release, so no published configuration can be
  affected.
- **Three more `CadenceEma` figures were re-derived against the bounded domain,
  and one `Ema` figure was wrong.** The coefficient floor at `delta = 1` was
  published as `2^-128` (read at `f32::MAX`, which no longer constructs) and is
  now `2^-26`; the `|alpha * x|` floor was `2^-277` and is now `2^-175`; and the
  `cadence_alpha` note claimed the derived coefficient lands "within an ulp" of
  the exact one; it lands within two (measured 1.51), and holding even that
  figure took a change to the derivation rather than only to the note. Forming
  the ratio as `(delta as f32) / tau` rounded the element *count* before the
  division — no `f32` holds an integer above 2^24 exactly — which put the
  coefficient 2.25 ulps out at `tau = MAX_TAU, delta = 16_812_203`, with 29_711
  further breaches of two ulps among the `delta`s that one `tau` admits. The
  ratio is now formed in `f64`, where `delta` is exact to 2^53, and narrowed
  once; below 2^24 that is bit for bit what the cast gave, so no other figure on
  the type moves. Separately, `Ema` and `CadenceEma` both said the `f64`
  `1 - alpha` collapses to exactly `1.0` at `2^-53`. It does not: `1 - 2^-53` is
  exactly representable. The threshold is `2^-54`, the tie that rounds to even,
  which also makes the `f32`-to-`f64` gap 29 binary orders rather than 28. Every
  figure above is now pinned by a test that sweeps the accepted domain to its
  edges, `MAX_TAU` included.

## 0.1.2 - 2026-07-25

### Changed

- `segment::HysteresisSegment` now segments in a single fused pass over the
  source sequence, sharing `smooth::Hysteresis`'s latch transition
  (`Hysteresis::step`) rather than smoothing into a full intermediate
  `Vec<Windowed<f32>>` and segmenting that. The output is identical to the
  previous two-pass composition for every input — finite, `NaN`, and `+/-inf`
  scores alike — enforced by a differential test against the retained two-pass
  reference (fixed geometries plus ~200 randomized finite cases). The
  full-length intermediate gated vector is no longer allocated, which an
  allocation-regression test pins. No public API or finite-input behaviour
  change.

### Documented

- The non-finite score and threshold semantics of `smooth::Ema`,
  `smooth::Hysteresis`, `segment::Threshold`, and `segment::HysteresisSegment`
  are now documented contract with exact-value tests: EMA does not sanitize
  inputs and a non-finite value poisons the rest of the call (including the
  `0.0 * inf` and `inf - inf` degradations); Hysteresis holds on `NaN`,
  latches and releases on infinities, and fails closed on a `NaN` `on`;
  Threshold membership is raw IEEE `>=`. The contradictory "never leaks a NaN
  downstream" comment on the EMA path is corrected.
- Both `SmoothPolicy` and `SegmentPolicy` are documented as restarting policy
  state on every call — batch conveniences, not incremental decoders.
- `segment::runs`' ascending-span precondition is sharpened: non-monotonic
  input still returns deterministically without panicking and yields
  well-formed ranges, but which ranges it returns is unspecified.

## 0.1.1

### Fixed

- `smooth::Hysteresis` now turns off strictly below `off` (`value < off`)
  instead of at or below it (`value <= off`); `segment::HysteresisSegment`
  inherits the fix, since it composes `Hysteresis` rather than reimplementing
  the latch. A value exactly at `off` now holds the gate's previous state
  instead of unconditionally turning it off — the hold region is the
  half-open band `off <= value < on`. This matches the strict-below
  convention both real VAD systems this primitive generalizes use at their
  own off threshold; the prior inclusive boundary was faithful to neither.
  Output changes only for inputs exactly equal to `off`; every other input
  (including inputs equal to `on`, and every input strictly above or below
  either threshold) is unaffected.

## 0.1.0

Initial release: the generic windowed-sequence processing core (pre + post) for
embeddings, VAD, and ASR.

### Added

- Window geometry: `WindowOptions`, `WindowPlan`, `Span`, and `TailPolicy` for
  turning an input length into unit-agnostic window spans. `Span` and
  `segment::Range` each pair a panicking `new` with a checked `try_new`, and
  both enforce their invariants — `0 < len <= window` with a representable
  `Span::end`, and `start <= end` — identically in debug and release.
- Pre-processing: `slice_pad_mask` / `try_slice_pad_mask` to slice, right-pad,
  and mask a span into a fixed-width window. A window is a caller-supplied count
  that need not correspond to memory that exists, so `try_slice_pad_mask`
  reserves it fallibly and reports `WinditError::AllocFailed` where the
  infallible variant documents a panic. `WindowPlan::spans` reserves its plan the
  same way, so an untrusted `input_len` is answered rather than approached one
  `push` at a time.
- Scalars: the sealed `Scalar` and `Real` traits and the `ComputeOf` alias.
  `Scalar` is implemented for the storage types `f32`, `f64`, and `i8`, and,
  behind the `half` feature, `half::f16` and `half::bf16` (re-exported as
  `scalar::f16`/`scalar::bf16`); `Real` — the domain the aggregation math runs
  in — is implemented only for `f64`. Every non-`f64` scalar widens to `f64`
  through `Scalar::Compute` rather than computing in itself: `f32`, `f16`, and
  `bf16` widen exactly (`Scalar::TO_COMPUTE_IS_VALUE` is `true` — every finite
  value of each is exact in `f64`), while `i8` is a *code* scalar whose widened
  value is not the value it represents until an embedding applies a
  quantization scale this crate cannot know (`TO_COMPUTE_IS_VALUE` is `false`).
  `Vector` carries an associated `Scalar`, so embeddings are generic over what
  they store.
- Quantized storage: `Vector::compute_components` projects an embedding's
  stored scalars into represented values before aggregation weighs them. The
  default projection borrows `f64` storage zero-copy, widens a
  value-preserving narrower scalar (`f32`, `f16`, `bf16`) elementwise, and
  refuses an `i8` embedding with `WinditError::MissingDequantization` rather
  than fold raw quantization codes as if they were values. A quantized
  `Vector` overrides the method with its own dequantization — per-tensor,
  per-row, per-block, affine or not; this crate never sees the parameters.
  Gated behind the same `alloc` tier as the aggregation it feeds.
- Aggregation input domain and determinacy: every input component to
  aggregation must be finite and either zero or of magnitude in
  `[Real::MIN_AGG_MAGNITUDE, Real::MAX_AGG_MAGNITUDE]` (`[2^-400, 2^400]` for
  `f64`), and every coverage a finite fraction in `[0, 1]`; a violation is
  rejected before any arithmetic runs, as `WinditError::MagnitudeOutOfRange`
  or `WinditError::CoverageOutOfRange`. Within that domain, aggregation folds
  through a compensated (Neumaier) sum with a proven error bound, and a
  determinacy gate rejects any result at or below its own rounding floor —
  `16 * Real::EPSILON * ||M|| + Real::MIN_GATE_THRESHOLD`, `M` the accumulated
  term magnitudes — as `WinditError::NonFinite`, so an exactly (or
  near-exactly) cancelling fold reports no direction instead of amplifying
  rounding noise into one. The `MIN_GATE_THRESHOLD` absolute term keeps that
  gate sound once `EmaRenormalized` — the one built-in policy whose recency
  weights are unbounded below — drives a fold's products toward the subnormal
  range, where the relative term alone would underflow to zero.
- Aggregation policies: the object-safe `AggregatePolicy<C: Real = f64>` with
  the built-ins `CoverageWeightedMean`, `MeanRenormalized`, `EmaRenormalized`,
  and `SaliencyWeighted`, the multi-vector `keep_separate`, and the serde
  `AggregatePolicyKind` selector. `f64` is the sole `Real` implementor and the
  domain every storage scalar computes in — `f64` as itself, every other
  shipped scalar by widening — so the `f64` default keeps `dyn AggregatePolicy`
  and `Box<dyn AggregatePolicy>` spelling the object every embedding needs.
  Policy configuration (for example `EmaRenormalized`'s `alpha`) stays `f32`
  regardless of `C`, so `AggregatePolicyKind`'s wire format is
  scalar-independent.
- Normalization is scale-aware: a vector whose norm is not representable even
  though the vector itself is (`[f64::MAX, f64::MAX]`, whose norm `sqrt(2) *
  f64::MAX` overflows) is normalized against its largest component instead of
  being rejected as `NonFinite`; the same divide-by-own-scale technique
  handles a vector whose squares alone would leave `f64`'s range.
  `SaliencyWeighted` squares magnitudes (a weight times a component, the
  weight itself a norm) with no separate rescaling step: the input domain
  above is sized so that square always stays a finite, normal `f64`. This is a
  fallback only: a sum of squares that lands in range is still the answer, bit
  for bit, so no ordinary-magnitude result moves.
- Smoothing policies: the `SmoothPolicy` trait with `Ema` and `Hysteresis`.
- Segmentation: `runs`, `longest_run`, `runs_sorted`, and the `SegmentPolicy`
  built-ins `Threshold` and `HysteresisSegment`. A run is continuous in the
  input geometry as well as in the sequence, so a plan whose hop exceeds its
  window never fuses two accepted spans across the elements it strided past;
  only `merge_gap` bridges them.
- Split policies: `FixedWindow`, and the tokenizer-free `ContentAware` string
  chunker behind the `text` feature. `ContentAware::chunk` is fallible: it
  reports invalid geometry and honours `WindowOptions::max_windows` exactly as
  `WindowPlan::spans` does, so the one configured bound on how much work a
  chunking may cost reaches the chunker too. The cap gates that work rather than
  reporting on it after the fact: atoms are produced on demand and packed as they
  are produced, so a capped chunking stops at the first chunk past the cap and
  never splits or measures the text beyond it, and peak memory is one chunk's
  worth of atoms rather than the whole input's. Neither the chunk list nor that
  atom buffer can be sized before packing runs, so both grow through
  `try_reserve` and report `WinditError::AllocFailed` rather than aborting a call
  that returns `Result`. Packing queries the caller's `MeasureText` `O(a)` times
  for `a` atoms — it never re-measures a range whose measure it already knows,
  and it locates each overlap boundary by a linear scan over just the trailing
  atoms of the chunk it closes (not a bisection: a context-sensitive measurer's
  token count need not fall monotonically as a repeated suffix shortens, so only
  a walk from the longest candidate suffix inward finds the earliest one that
  fits without silently dropping configured overlap) — which keeps a
  near-window overlap over untrusted text off the cubic path. Boundaries are
  still decided by measuring the real contiguous text, never by summing
  per-atom measurements, so a non-additive (BPE, wordpiece) `MeasureText` keeps
  its exact chunk boundaries. `chunk` returns `Vec<Chunk>`, not raw `(usize,
  usize)` byte offsets: `Chunk` is a half-open UTF-8 byte range with an
  `as_str` accessor, pairing a panicking `new` with a checked `try_new` exactly
  as `Span` and `Range` do (`WinditError::InvalidChunk` when `start > end`,
  enforced identically in debug and release), and kept a distinct type from
  `segment::Range` (input-element units) so a byte offset and an element index
  cannot silently trade places at a call site.
- `no_std + alloc` support with optional `std`, `text`, `serde`, and `half`
  features; minimum supported Rust version 1.95. `libm` is an unconditional
  dependency: `Real::sqrt` lives on the ungated core tier, so even a
  `--no-default-features` build needs it. `half` does not imply `alloc`: the
  `f16`/`bf16` scalars are core-tier, so `--no-default-features --features
  half` is a valid trait-surface-plus-scalars build with no algorithms.
