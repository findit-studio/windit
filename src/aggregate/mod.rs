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
//! - [`EmaRenormalized`]'s recency weights are `w_0 = (1 - alpha)^(n - 1)` for
//!   the oldest window and `w_i = alpha * (1 - alpha)^(n - 1 - i)` for the rest —
//!   the split the recurrence `s_i = alpha * e_i + (1 - alpha) * s_{i-1}` from
//!   `s_0 = e_0` actually produces, the first window carrying no `alpha` factor
//!   because nothing preceded it to blend with. Those *ideal* weights sum to
//!   exactly `1` in exact arithmetic (the `alpha` terms telescope to
//!   `1 - (1 - alpha)^(n - 1)`), which is the sense in which EMA has a canonical
//!   unit-order scale and so no free scale to divide out; the `f64` weights the
//!   policy materializes from it do not generally sum to `1` (at `alpha = 0.3`
//!   and `n = 4` they sum to `0.9999999999999998`), and nothing here needs them
//!   to. What matters is that they decay without limit against the newest
//!   window: at a large window count the *weights themselves* leave the
//!   exponent range, `alpha * (1 - alpha)^k` reaching the subnormal grid and
//!   then zero even for in-domain inputs. That is a different regime from a
//!   subnormal *product*, and the difference is what
//!   [the note below](self#a-weight-the-fold-did-not-form) is about.
//! - [`CoverageWeightedMean`] folds `c_i / max_j c_j`, lifted by one shared
//!   power of two, so its largest weight is exactly `2^shift` however the caller
//!   scaled the slice — `shift` being zero for every slice whose weights are
//!   already normal, which is every slice a plan can produce. See that type's
//!   *Weights up to scale* note. The domain admits the rest anywhere in `[0, 1]`,
//!   so a window weighing `2^-1000` against the fullest one drives its own
//!   product subnormal.
//!
//! A subnormal *product* is a regime the determinacy gate's absolute floor
//! handles, and the floor's soundness argument there is about products rather
//! than about which policy formed them. What a pinned *largest* weight buys is
//! the other half of the picture: the fold's accumulated mass is at least the
//! heaviest window's own, so the floor can decide a verdict only when the
//! heaviest windows carry no mass at all. For [`CoverageWeightedMean`] that
//! means the fullest windows are themselves the zero vector — never that the
//! caller's coverages were small, which is a scale and cannot change a
//! normalized weighted mean. A subnormal *weight* is not that regime, and
//! [`EmaRenormalized`] is the only built-in policy that reaches it; see
//! [A weight the fold did not form](self#a-weight-the-fold-did-not-form).
//!
//! Every value an `f32`-storage embedding can produce lies more than 250 binary
//! orders inside this window on both sides, so no realizable `f32` input ever
//! reaches a boundary.
//!
//! Within the domain, an aggregated result is the direction of a vector within
//! `4 * `[`Real::EPSILON`]` * ||M|| + K_abs` of the exact weighted sum, where `M`
//! is the componentwise sum of the folded term magnitudes and `K_abs` is the small
//! absolute term defined below; any result whose norm is at or below
//! `16 * `[`Real::EPSILON`]` * ||M|| + `[`Real::MIN_GATE_THRESHOLD`]` + S` is reported
//! as [`WinditError::NonFinite`] — no direction is determined at working
//! precision. This is the crate's one accuracy claim, and it is a theorem rather
//! than an observation, and it is stated against the exact weighted sum of the
//! weights the policy *intends*, not of the ones it managed to represent — which
//! is why the weight's own formation error is the first term of the derivation
//! and not an omission from it.
//!
//! **That first term is not the fold's to carry.** `16 * EPSILON * ||M||` bounds
//! what the *fold* did to the terms it was given; the gap between a materialized
//! weight and the ideal one is a property of whatever formed the weight, and only
//! that thing can bound it. So the threshold carries a third term `S`, supplied by
//! the policy beside its weight function:
//!
//! ```text
//! S = sum_i E_i * ||e_i||,   E_i a proven bound on |w_i - W_i|
//! ```
//!
//! — the *unweighted* window norms, which is a different quantity from `||M||` and
//! cannot be recovered from it. Three of the four built-ins hand in a literal
//! `C::ZERO`, and their verdicts are therefore unchanged to the bit:
//! [`MeanRenormalized`]'s weight is the exact constant `1`; [`CoverageWeightedMean`]'s
//! is one correctly rounded division of a lifted coverage, so `E_i <= u * w_i` and
//! the gate's own term already covers it — the lift being what keeps that division
//! out of the subnormal range where its error would turn absolute; and
//! [`SaliencyWeighted`]'s weight is a norm the input domain bounds below, taken by
//! the same `l2_norm` the gate measures `||M||` with, which is the sense in which
//! that policy weights by *this crate's* norm rather than by an ideal it
//! approximates.
//!
//! [`EmaRenormalized`] is the one that owes a term, and the reason it owes one is
//! the correction this note carries. Its ladder is built by repeated
//! multiplication, so weight `k` carries the roundings of every step before it —
//! about `0.7 * n * u`, past the gate's own `32u` near `n = 32`. A previous
//! revision measured that growth, agreed it was real, and then argued it had **no
//! reach**. That argument is false, and there is a witness.
//!
//! What survives of it is the opening move. For any input whose ideal weighted sum
//! is exactly zero the ideal terms `t_i` sum to zero, so any constant may be
//! subtracted from the weights' relative errors `d_i`, and the residue the gate
//! sees obeys `|sum_i t_i d_i| / sum_i |t_i| <= (max_i d_i - min_i d_i) / 2`: what
//! a witness needs is the error's *spread over its own support*, above `64u`,
//! never its size. What does not survive is the claim that no support reaches one.
//! That rested on a **two-window lever cap** — a pair cancels exactly only when
//! `(1 - alpha)^d` is a ratio of two `f64` significands, which with the complement
//! written `B * 2^-q`, `B` odd, needs `B^d < 2^53` — plus a search over pairs and
//! adjacent triples. **A cancellation is not a pair.** At `alpha = 1639/8192`,
//! where `b = 6553/8192` is exact, the polynomial
//!
//! ```text
//! P(x) = (8192 x - 6553) * SUM_j f_j x^j,   f_j = fl(f_{j-1} / b),  f_0 = 1
//! ```
//!
//! vanishes at `b` because its first factor does — no lever, and a support as wide
//! as the second factor's degree. Its coefficients `c_k = 8192 f_{k-1} - 6553 f_k`
//! are exactly representable, and not by luck: `c_k` is `-6553` times the rounding
//! in `f_k`, about `2^-40 * f_k` on the grid of `f_k`'s own last bit — the same
//! thirteen bits `B = 6553` occupies. **The short mantissa that caps the lever is
//! what buys the coefficients.** Over `1278` windows at chain indices `1168..2446`
//! the spread reaches `71.9u` against the `10.0u` the old search reported as the
//! widest reachable, every materialized weight stays a normal `f64`, every
//! component stays inside the domain, and the fold returned `Ok([-1.0])` out of an
//! exactly cancelling input at `1.09x` the threshold.
//! `a_multi_window_polynomial_cancellation_reaches_the_ema_weight_error_bound`
//! drives it.
//!
//! So EMA carries the bound instead of arguing it away: `E_i` is
//! `(2k + 2) * EPSILON * w_i` for the relative part and the subnormal grid for the
//! absolute one, and `ema_formation_slack` derives both. Evaluating each weight
//! once (`alpha * (1 - alpha).powi(k)`) is not an alternative and never was: `powi`
//! is exponentiation by squaring, so `O(log k)` roundings and not correctly
//! rounded, and it raises the same `fl(1 - alpha)` the chain does — that single
//! complement rounding, multiplied by `k`, is the larger part of the error.
//! Measured at `alpha = 0.46, n = 64`: `58.75u` for the chain against `48.15u` for
//! `powi`, a fifth, not a factor of `k` — and the witness's complement is exact, so
//! `powi` would not have touched it at all.
//!
//! Each product `w_i * e_i` is then rounded relatively when it
//! is a normal `f64` (by at most `u * |w_i * e_i|`, `u = EPSILON / 2`) and
//! absolutely when it has underflowed toward a subnormal (by at most `2^-1075`,
//! half the subnormal spacing). Per dimension the weight and product relative
//! parts sum to at most `2u * M_j`, the Neumaier fold adds at most `2u * M_j`
//! (plus an `O(n * u^2) * M_j` tail, and it is exact for subnormal operands),
//! together at most `4 * EPSILON * M_j`; the absolute parts sum to at most
//! `n * 2^-1075`. Over all dimensions,
//! `||R - exact|| <= 4 * EPSILON * ||M|| + K_abs` with
//! `K_abs <= sqrt(dim) * n * 2^-1075 <= 2^-1018` for any `n <= 2^40` and
//! `dim <= 2^32`. The threshold
//! `τ = 16 * EPSILON * ||M|| + `[`Real::MIN_GATE_THRESHOLD`]` + S` carries a matching
//! absolute floor (`2^-1000` for `f64`, above `K_abs` and — for any mass a
//! domain-bounded weight accumulates — far below `16 * EPSILON * ||M||`), so an
//! exactly cancelling sum has `||R|| <= 4 * EPSILON * ||M|| + K_abs < τ` and is
//! always gated, whatever the ordering, tier structure, or weight range — so no
//! fold can fabricate a direction from in-domain cancellation without violating the
//! bound — **as long as every weight is the ideal one to within `u`**. Where it is
//! not, `S` is what carries it, and `S` is supplied by the *policy* rather than by
//! the fold, for the reason
//! [A weight the fold did not form](self#a-weight-the-fold-did-not-form)
//! gives; it is exactly `0` for three of the four built-ins, so their thresholds
//! are the same numbers to the bit. When a fold's *products* are driven
//! subnormal by an unbounded weight ratio while the weights themselves stay
//! normal, `||M||` is subnormal too and `16 * EPSILON * ||M||` underflows,
//! leaving the floor to gate alone: the entire signal then sits below the
//! precision the domain guarantees, so `NonFinite` remains the honest verdict.
//! The floor also engages earlier, while every product is still normal: once the
//! accumulated mass falls
//! below about `2^-948`, `16 * EPSILON * ||M||` itself drops beneath the `2^-1000`
//! floor and the floor decides the verdict alone, monotonically turning a
//! sub-floor direction into `NonFinite` rather than admitting it — an
//! over-rejection-only widening of the gate, pinned by a regression test.
//!
//! An absolute floor is only ever sound against a quantity carried in the
//! embedding's own units, which `||M||` and `||R||` are and a *weight* is not —
//! and that is also why `S` is not a second floor but a *mass*, each window's
//! unweighted `||e_i||` against a bound on its own weight's error.
//! So reaching either regime takes an unbounded weight **ratio** —
//! [`EmaRenormalized`]'s decaying recency factors, or a [`CoverageWeightedMean`]
//! fold whose fullest windows are themselves all zero — and never a weight
//! **scale**, which the renormalization ending every policy divides back out.
//! Neither regime is one a realizable `f32` workload reaches through
//! [`aggregate`].
//!
//! # A weight the fold did not form
//!
//! The paragraph above holds while every weight is the ideal one to within a flat
//! `u`. There is one built-in policy for which it is not — twice over, relatively
//! and absolutely — and this section is what the gate carries instead.
//!
//! ## Relatively, without limit
//!
//! [`EmaRenormalized`]'s ladder is a chain of multiplications, so weight `k` is
//! off its ideal by the `k` roundings before it and by the `k` copies of
//! `b = fl(1 - alpha)` standing in for the exact complement: `2k + 1` roundings,
//! `|w_i / W_i - 1| <= gamma_(2k+1)`. **No constant multiple of `EPSILON` bounds
//! that**, because it grows with the window count, and the fold's own
//! `16 * EPSILON * ||M||` is a constant. The revision that claimed otherwise
//! reasoned from a two-window lever cap and a search over pairs, and named its own
//! limit — "a counting argument plus a search, not a theorem". The witness the
//! [Input domain](self#input-domain) note records lives exactly there: a
//! polynomial that vanishes at the complement needs no lever at all, and reaches a
//! `71.9u` spread over `1278` windows where the search reported `10.0u` as the
//! ceiling.
//!
//! So the term is `E_i = (2k + 2) * EPSILON * w_i`, twice the derived
//! `gamma / (1 - gamma)` bound, the doubling covering the slack's own arithmetic.
//! It is charged against `||e_i||` like the absolute part below, one formula for
//! both: `E_i` bounds `|w_i - W_i|` and how the weight came to be wrong changes
//! only which half dominates. Two ladders are exempt, by certificate rather than
//! by measurement — `alpha == 1` and `alpha == 0`, whose weights are exact at
//! every index, and any `alpha >= 1/2` whose complement is a power of two, where
//! Sterbenz makes `1 - alpha` exact and a power-of-two multiply moves only the
//! exponent. That second one is what keeps `alpha = 0.5`'s published dyadic range
//! bit-identical at the *gate* as well as in the ladder.
//!
//! What this costs a fold that is not cancelling is a few `EPSILON` of its own
//! mass: `sum_k (2k + 2) * EPSILON * alpha * b^k` is about `2 * EPSILON / alpha`,
//! against an accumulator of order `||e||`. It bites only where the fold has
//! already cancelled to within about `n * u`, which is where the weights genuinely
//! stop resolving a direction.
//!
//! ## Absolutely, past the exponent range
//!
//! [`EmaRenormalized`]'s weight ladder is also the only one whose *range*, not
//! merely whose ratio, is unbounded: `alpha * (1 - alpha)^k` falls below
//! [`Real::MIN_NORMAL`] and then below half the subnormal spacing at a window
//! count of about `1074 / log2(1 / (1 - alpha))` — `326` at `alpha = 0.9`, `23`
//! at `alpha = 1 - 2^-53`. There the weight is rounded **absolutely**, to the
//! subnormal grid, exactly as a [`CoverageWeightedMean`] quotient was before its
//! lift; and no way of forming the weight repairs it, because the value itself is
//! not an `f64`. The ratio between two adjacent ideal weights is
//! `1 / (1 - alpha)` — a factor of ten at `alpha = 0.9` — and at the bottom of
//! the subnormal grid that factor cannot be represented at all, so the older of
//! the pair rounds to zero while the newer survives. `powi` reaches the same
//! zero.
//!
//! [`CoverageWeightedMean`]'s cure does not transfer. Its lift works because a
//! coverage ratio is bounded by `f64`'s own range, so one shared power of two
//! puts every quotient in the normal range at once; EMA's ladder can span more
//! binary orders than `f64` has, and lifting it far enough would overflow the
//! products against a domain that admits components up to `2^400`. Measured
//! rather than asserted: a shared lift needs `w_max / w_min <= 2^1646` to keep
//! every weight normal *and* every product finite under that ceiling, against an
//! underflow onset at `2^1074` — half as much reach again, bought by moving every
//! fold in the regime it does not fix; and a lifted accumulator is no longer in
//! the embedding's units, so it would have to un-scale before the gate besides.
//!
//! The residue of an exactly cancelling fold is
//! `R_j = sum_i (w_i - W_i) * e_ij` against the *ideal* weights `W_i`, so
//!
//! ```text
//! ||R||  <=  sum_i |w_i - W_i| * ||e_i||
//! ```
//!
//! — the **unweighted** window norms, which is a different quantity from `||M||`
//! and cannot be recovered from it. With in-domain components that reaches
//! `n * 2^-675`, far above the `2^-1000` floor, which is why an exactly
//! cancelling in-domain fold used to come back as a direction: at `alpha = 0.9`
//! over `326` windows, with two ordinary components near `10^24` whose ideal
//! weighted sum is exactly zero, [`aggregate`] returned a unit vector.
//!
//! So `E_i` gains an absolute part alongside its relative one, written in units of
//! `MIN_NORMAL * EPSILON`, charged only where the materialized weight is under
//! `MIN_NORMAL`, and carrying **two coefficients, one per position**:
//! `(1 + alpha * D)` for the general weight `fl(alpha * p_k)` and the bare `D` for
//! `w_0`, with `D <= 1 / (1 - fl(1 - alpha))` the geometric damping of the chain's
//! own roundings that both are written over.
//!
//! **The general term's coefficient keeps its `alpha`.** A previous revision
//! dropped it for `alpha <= 1` and charged `(1 + D)`, which is a valid inequality
//! and a bad bound: `D` is about `1 / alpha`, so the derived `(1 + alpha * D)` is
//! about `2` at every coefficient while `(1 + D)` grows without limit as the
//! coefficient shrinks. At `alpha = 0.05, n = 14471`, one `2^400` component on a
//! flushed weight and one `2^-400` on a normal one, the dropped-`alpha` term came
//! to `5.185x` the accumulator and decided [`WinditError::NonFinite`] by itself —
//! where the flushed window's ideal contribution is `0.12` of the live term, so
//! even reversing it leaves a direction.
//! `the_underflow_slack_does_not_charge_1_over_alpha` drives that.
//!
//! **The oldest window is the exception the recurrence's own convex form already
//! names.** [Input domain](self#input-domain) writes the weights as
//! `w_0 = (1 - alpha)^(n - 1)` and `w_i = alpha * (1 - alpha)^(n - 1 - i)`, and
//! the ladder builds them that way — `weights[0]` is the bare chain value with no
//! `alpha` factor at all. The `1` in `(1 + alpha * D)` is the final `alpha *`
//! multiplication's own rounding and the `alpha * D` is every chain rounding
//! damped by it, so neither half of that coefficient is window 0's: its chain
//! roundings arrive undamped and its unit is `D`. The gap is `1 / (2 * alpha)`,
//! and it is reachable because past the flush point the ladder does not decay to
//! zero but **stalls** — `fl(p * b) == p` while `(1 - b) * p <= 2^-1075`, so the
//! chain lands on a fixed point of the subnormal grid at `floor(D / 2)` ulps of
//! `2^-1074`, within one grid step *below* the derived `2^-1075 * D` rather than
//! on it. One step, not orders: nothing smaller than `D` is a bound.
//!
//! ```text
//! alpha = 0.05, n = 20000, dim = 1, one 2^400 component on window 0, zeros elsewhere
//!   the ladder stalls at   9 * 2^-1074     w[0] * b == w[0]
//!   the relative charge    underflows to zero
//!   the absolute charge    2 * 2^-1074     derived for a weight window 0 is not
//!   derived for w[0]      20 * 2^-1074     D = 1 / (1 - fl(0.95))
//!   acc                    0x1.2p-671      an ordinary normal f64
//!   tau                    0x1.0000000000048p-673
//!   before                 Ok([1.0])       against an ideal contribution of
//!                                          2^-1079.94, eighty binary orders under
//!                                          the 2^-1000 floor
//!   now                    Err(NonFinite)
//! ```
//!
//! The two coefficients coincide at `alpha = 1/2` (`D = 2`, and `1 + 0.5 * 2 = 2`),
//! so no dyadic verdict moves.
//! `the_oldest_weights_charge_is_not_damped_by_alpha` drives it, and
//! `every_windows_charge_bounds_that_windows_own_weight_error` pins the
//! per-position invariant the whole class of defect on this seam violates.
//!
//! Four consequences worth stating plainly:
//!
//! - **`S` belongs to the policy, not to the fold.** A weight's formation error
//!   is a property of whatever formed the weight, so the fold cannot derive it
//!   and a shared widening of `τ` would tax three policies for a fourth's
//!   arithmetic. [`Real::MIN_GATE_THRESHOLD`]'s own soundness argument is "about
//!   products rather than about which policy formed them", and **that framing
//!   does not survive a term about weights** — which is exactly why this one is
//!   passed in beside the weight function rather than added to the floor.
//!   [`MeanRenormalized`] (an exact constant `1`), [`CoverageWeightedMean`] (one
//!   correctly rounded division of a lifted coverage) and [`SaliencyWeighted`]
//!   (a norm the input domain bounds below) all supply `C::ZERO`, and their
//!   verdicts are unchanged bit for bit.
//! - **It is a norm, not `n * max_ij |e_ij|`.** The scalar form the issue
//!   prototyped has no `dim` in it, and the residue's does: fill every dimension
//!   equally and `||R||` grows as `sqrt(dim)` while `max |e|` does not move. The
//!   ratio between the two is capped at `sqrt(dim) / (2 * n)` by the flush
//!   condition, and `alpha = 1 - 2^-43` over `27` windows reaches it closely
//!   enough for an eight-thousand-wide embedding to clear the prototype by
//!   `1.68x`. `the_weight_underflow_slack_carries_the_dimension` drives that
//!   input.
//! - **It is charged only where a weight actually left the range.** A `400`-window
//!   EMA at `alpha = 0.9` has `92` weights at or under the boundary and is still
//!   an entirely ordinary fold; charging its whole slice would have made `S` the
//!   `n`-times-larger thing it is not. As it stands `S` sits some `10^-321` under
//!   such a fold's own mass, so it decides nothing there —
//!   `an_ordinary_long_ema_still_answers_past_the_underflow_point` pins both
//!   halves.
//!
//! - **It is a bound on a weight, not a budget for a policy.** The absolute unit
//!   is charged at twice its derived size and the derived size is itself up to
//!   twice what a given flushed weight is worth, so a window in this regime is
//!   charged about `4x` its ideal contribution — the figure the `alpha = 0.05`
//!   row above reads as `0.494` against `0.120`. That is the conservatism the term
//!   is allowed; `1 / alpha` was not.
//!
//! `ema_weights_below_the_exponent_range_cannot_fabricate_a_direction` is the
//! falsifier, over `alpha = 0.9`, `0.5`, `1 - 2^-30` and `1 - 2^-53`; every row
//! returned `Ok` before this term and returns [`WinditError::NonFinite`] now. The
//! `alpha = 0.5` row is the one that identifies the mechanism: that chain is
//! exact at every representable index — it is one of the exempt ladders above —
//! so it carries none of the relative error at all, and `0.5 * 2^-1074` is simply
//! not an `f64`.
//!
//! What this does **not** claim is that the regime is now accurate. The verdict
//! it produces is a refusal: past the point its ladder leaves the exponent range,
//! a fold whose mass rides on the underflowed windows has no direction at working
//! precision, and says so. A fold whose underflowed windows carry no mass is
//! untouched to the bit; one whose underflowed windows carry mass is charged about
//! four times what that mass is ideally worth, and is refused only where that
//! charge outruns the live answer.
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
/// them is exactly a power of two. An all-zero slice is not a scale of anything:
/// every weight is zero, the exact sum is the zero vector, and the
/// [determinacy gate](self#input-domain) reports [`WinditError::NonFinite`], no
/// direction to report.
///
/// # A ratio is not materialized where it cannot be represented
///
/// A weight is a *ratio*, and the ratio of two in-domain coverages can be
/// arbitrarily small — `f64::from_bits(1)` against `0.75` is `(4/3) * 2^-1074`.
/// Below `2^-1022` an `f64` quotient rounds *absolutely*, to the subnormal grid,
/// so that ratio and its double land on `2^-1074` and `3 * 2^-1074`: a relative
/// error of a quarter and an eighth, in a fold whose whole error argument is
/// relative. Nothing downstream recovers it — a compensated sum is exact for
/// subnormal operands but sums the terms it is handed — and an exactly
/// cancelling in-domain fold then reports a direction it does not have.
///
/// So the slice is first lifted by one shared, exact `2^shift` and the weights
/// are `ldexp(c_i, shift) / max_j c_j`. The lift is a common positive factor, so
/// it changes no ratio; it is exact, because a value at or under `1` scaled up by
/// a power of two keeps its significand, subnormals included; and it is sized so
/// the smallest nonzero quotient lands in the normal range. `shift` is **zero
/// unless the smallest nonzero weight would itself be subnormal**, so this is the
/// identity on every slice a plan can produce and on every slice whose weights
/// were already sound.
///
/// **Each weight is still a rounded quotient**, and this note says so rather than
/// claiming otherwise: what the lift establishes is not that no weight is rounded
/// but that every weight is rounded *relatively*, by at most the unit roundoff,
/// which is the property the fold's [error bound](self#input-domain) is stated
/// against and the one a subnormal quotient breaks. Making the weight exact
/// outright is possible — divide by `2^exponent(max_j c_j)` instead of by
/// `max_j c_j`, which is a shift and rounds nothing — but it forfeits a largest
/// weight of exactly `1` for one anywhere in `[1, 2)`, and so moves the answer for
/// every slice whose largest coverage is not a power of two: 71% of a synthetic
/// four-window sweep (arbitrary tuples pushed through the policy directly, not
/// through a `WindowPlan`), including the ragged single window whose answer this
/// release had just made exact. No real *four*-window plan slice is among the
/// swept 71%: three of its four windows are always full, so its largest coverage
/// is always exactly `1`, already a power of two. A relative `u` on the weight
/// costs the fold nothing the bound does not already carry; that trade is why
/// the division stays.
///
/// # Scaling, bit for bit and otherwise
///
/// Scaling every coverage by an `s` for which **every product `s * c_i` is
/// exactly representable**, stays in range, and leaves no positive entry rounded
/// to zero, leaves each quotient's exact value untouched — and IEEE division is
/// correctly rounded, so the weights and with them the whole fold are
/// bit-identical. That is the contract, and it is about the *products*, not
/// about the factor: `[1.0, 0.1]` scaled by `0.1` is `[0.1,
/// 0.010000000000000002]`, whose second product is not exactly representable, so
/// the two slices are no longer proportional and the fold moves by an ulp.
/// Ordinary floating scaling is *approximately* invariant — the answer moves only
/// by the rounding the caller's own multiplication introduced.
///
/// One consequence worth naming: the [input domain](self#input-domain)'s
/// `[0, 1]` is the whole of the contract. A coverage anywhere in it, however
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
    // The whole slice is first lifted by one common, exact power of two, so that
    // no quotient is ever *formed* in the subnormal range where its rounding
    // would stop being relative (see [`normalizing_shift`]). The lift is a shared
    // factor and so changes no ratio; it is zero for every slice whose weights
    // are already normal, which is every slice a plan can produce.
    let shift = normalizing_shift(coverages, largest);
    weighted_sum_renorm(
      embeddings,
      coverages,
      dim,
      move |i, _| {
        if largest > 0.0 {
          // `coverages[i] <= largest`, so the quotient is at most `2^shift` and
          // cannot overflow; `shift` is chosen so it cannot be subnormal either.
          // `ldexp` by a non-negative exponent onto a value of at most `1` is exact,
          // so the only rounding is the division's own, and that is relative.
          C::from_f64(coverages[i].ldexp(shift) / largest)
        } else {
          // Every coverage is zero. The exact weighted sum is the zero vector, and
          // the determinacy gate is what reports it; the division is skipped rather
          // than allowed to produce the `0 / 0` NaN that no gate can see.
          C::ZERO
        }
      },
      // Every weight is one correctly rounded division of a lifted coverage, so
      // its error is relative and the gate's own `16 * EPSILON * ||M||` already
      // carries it. The lift is what establishes that; see [`normalizing_shift`].
      |_| C::ZERO,
    )
  }
}

impl<C: Real> AggregatePolicy<C> for MeanRenormalized {
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f64],
    dim: usize,
  ) -> Result<Vec<C>, WinditError> {
    // The weight is the exact constant `1`: no formation error at all, relative
    // or absolute, so nothing is owed the gate.
    weighted_sum_renorm(embeddings, coverages, dim, |_, _| C::ONE, |_| C::ZERO)
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
    // A norm is taken against its own power-of-two scale, and the input domain
    // puts every window's norm at or above `MIN_AGG_MAGNITUDE` — so no weight is
    // ever formed in the subnormal range and every one of them is rounded
    // relatively.
    weighted_sum_renorm(embeddings, coverages, dim, |i, _| weights[i], |_| C::ZERO)
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
    // there is no separate recurrence fold left to fabricate a direction. What
    // the shared fold cannot carry is the *weight*'s own formation error. This
    // ladder is built by repeated multiplication, so weight `k` sits off its
    // ideal relatively by the `k` roundings before it, and — once the ladder
    // leaves `f64`'s exponent range — absolutely by the subnormal grid it landed
    // on. Neither is a property of the fold, so neither can live inside the
    // gate's own `16 * EPSILON * ||M||` constant; `ema_formation_slack` is the
    // term this policy owes the gate for both, and it is the only nonzero one any
    // built-in supplies. See the module's *A weight the fold did not form*.
    // The dyadic case (`alpha = 0.5` over basis vectors) reproduces
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
    weighted_sum_renorm(
      embeddings,
      coverages,
      dim,
      |i, _| weights[i],
      // The one policy that owes the gate a term of its own.
      |embs| ema_formation_slack(&weights, embs, alpha, complement),
    )
  }
}

/// The term [`EmaRenormalized`] owes the determinacy gate for the weights it
/// could not form exactly: `sum_i E_i * ||e_i||`, over every window, with `E_i`
/// a proven bound on how far the materialized weight sits from the ideal one.
///
/// The fold's whole error argument rests on every materialized weight being the
/// ideal one to within a bounded *relative* error the gate's own
/// `16 * EPSILON * ||M||` already covers. This is the one built-in policy for
/// which that fails, and it fails twice over:
///
/// - **Relatively.** The ladder is a chain of multiplications, so weight `k`
///   carries the rounding of every step before it, and of the complement itself.
///   That error grows with `k` without limit, so no constant multiple of
///   `EPSILON` bounds it — see the module's
///   [A weight the fold did not form](self#a-weight-the-fold-did-not-form).
/// - **Absolutely.** Below [`Real::MIN_NORMAL`] the grid is absolute, so a weight
///   there is rounded to it — and below half its spacing there is no
///   representable weight at all, so the older of two adjacent ideal weights
///   becomes zero while the newer survives. No way of *forming* the weight
///   repairs that one, because the value itself is not a `C`.
///
/// # The bound
///
/// Write `W_i` for the ideal weight and `w_i` for the materialized one. Where the
/// exact weighted sum is zero the residue the gate sees is
/// `R_j = sum_i (w_i - W_i) * e_ij`, so
///
/// ```text
/// ||R||  <=  sum_i |w_i - W_i| * ||e_i||
/// ```
///
/// — the *unweighted* window norms, which is why this cannot be folded into
/// `||M||` and why it is a `sqrt(dim)`-carrying quantity rather than the scalar
/// `n * max_ij |e_ij|` the issue prototyped. (A residue that fills every dimension
/// equally grows as `sqrt(dim)` while `max |e|` does not move at all;
/// `the_weight_underflow_slack_carries_the_dimension` drives the input where that
/// difference decides the verdict.) One formula covers both regimes, which is the
/// point: `E_i` is a bound on `|w_i - W_i|`, and how the weight came to be wrong
/// changes only which of its two parts dominates.
///
/// The chain is `p_0 = 1`, `p_{k+1} = fl(p_k * b)`, `w_i = fl(alpha * p_k)` at
/// `k = n - 1 - i`, and each rounding is *either* relative by at most
/// `EPSILON / 2` *or* absolute by at most half the subnormal spacing.
///
/// **The relative part.** `k` chain roundings, one for the final `alpha * p_k`,
/// and `k` more for `b = fl(1 - alpha)` being raised in place of the exact
/// `1 - alpha` — `2k + 1` roundings, so `|w_i / W_i - 1| <= gamma_(2k+1)` with
/// `gamma_m = (1 + u)^m - 1`, `u = EPSILON / 2`. Turning that into a bound on
/// `|w_i - W_i|` against the weight this actually *has* costs one more factor:
/// `W_i <= w_i / (1 - gamma)`, so
///
/// ```text
/// |w_i - W_i|_relative  <=  gamma / (1 - gamma) * w_i  <=  (2k + 2) * EPSILON * w_i
/// ```
///
/// wherever `(2k + 1) * u <= 1/4` — `n <= 2^50`, well past the `n <= 2^40` the
/// crate's own `K_abs` already assumes, so no realizable slice reaches the
/// condition and no guard here can bind. The right-hand form is about twice the
/// derived one, and that doubling is what covers the roundings in computing the
/// slack itself: the integer `2k + 2` is exact up to `2^41`, its product with
/// `EPSILON` is exact (a power of two), and what is left is one rounding per
/// multiply and one per accumulation, at most `(n + dim) * u` all told.
///
/// **The absolute part**, charged only where `w_i < MIN_NORMAL`, and **two
/// formulas rather than one**: the ladder builds `weights[0]` differently from
/// every other entry, so a unit derived at one position is not a bound at the
/// other.
///
/// ```text
/// eta = MIN_NORMAL * EPSILON = 2^-1074,   D = sum_{j<k} b^j <= 1 / (1 - b)
///
/// |w_i - W_i|_absolute  <=  (eta/2) * (1 + alpha * D)   i >= 1, w_i = fl(alpha * p_k)
/// |w_0 - W_0|_absolute  <=  (eta/2) * D                 w_0 = p_(n-1), no alpha factor
/// ```
///
/// Both come off the same chain. In the subnormal range every rounding is
/// absolute, at most `eta / 2`, and the one at step `j` is damped by the
/// `K - 1 - j` multiplications after it, so
/// `|p_K - b^K| <= (eta/2) * sum_{m<K} b^m <= (eta/2) * D` — and the leading `1`
/// of that geometric sum **is** the last chain step's own rounding, undamped.
///
/// - For `i >= 1` the weight is `fl(alpha * p_k)`. That final multiplication adds
///   one further undamped `eta/2` — the `1` — and damps every chain rounding by
///   `alpha` — the `alpha * D`.
/// - For `i == 0` the weight is the bare `p_(n-1)`. There is no final
///   multiplication, so it contributes no rounding of its own and damps none of
///   the chain's: the unit is `D`, which already carries the last chain step as
///   its own leading term.
///
/// The two **coincide exactly at `alpha = 1/2`** (`D = 2`, and `1 + 0.5 * 2 = 2`),
/// which is why splitting them does not move the published dyadic contract by a
/// bit.
///
/// **`alpha` stays in the general term.** Dropping it for `alpha <= 1` leaves
/// `(1 + D)`, which is about `1 / alpha` where the derived coefficient is
/// about `2`, so the term grows as the coefficient shrinks and refuses folds
/// whose direction is not in doubt: at `alpha = 0.05, n = 14471`, one `2^400`
/// component on a zero weight and one `2^-400` component on a normal one, the
/// dropped-`alpha` term is `5.185x` the accumulator and decides `NonFinite` by
/// itself, where the ideal contribution of that same window is `0.12` of the
/// legitimate term. `the_underflow_slack_does_not_charge_1_over_alpha` drives it.
/// The unit is `MIN_NORMAL * EPSILON` — `2^-1074`, one binade above the
/// half-spacing, and the smallest quantity of this shape that *is* representable
/// — which leaves the same factor of two for the roundings in `D` and the mass.
///
/// **Nothing smaller than `D` will do at window 0**, because the ladder does not
/// decay to zero: past the flush point it **stalls**. `fl(p * b) == p` while
/// `(1 - b) * p <= eta / 2`, that is while `p <= (eta/2) * D`, so the chain lands
/// on a fixed point of the subnormal grid at the largest grid multiple under that
/// condition — `floor(D / 2)` ulps of `eta`. That is *within one grid step below*
/// the derived bound, never on it, and one step is what makes the bound tight:
/// any coefficient smaller than `D` is one the stall already exceeds. Measured,
/// with `n` chosen just past the flush and `w[0]` in units of `eta`:
///
/// ```text
/// alpha      0.02  0.05   0.1  0.125  0.15   0.2   0.25   0.3   0.4  0.5  0.75   0.9
/// w[0]/eta     24     9     5      4     3     2      2     1     1    0     0     0
/// D          50.0  20.0  10.0    8.0  6.67   5.0    4.0  3.33   2.5  2.0  1.33  1.11
/// D/2        25.0  10.0   5.0    4.0  3.33   2.5    2.0  1.67  1.25  1.0  0.67  0.56
/// ```
///
/// `w[0]/eta` is `floor(D/2)` in every column but `alpha = 0.5`, and that one is
/// the tie-break rather than an exception to the bound: where `b` is a power of
/// two the last representable step is exactly `b` ulps, at `b = 1/2` exactly the
/// half-ulp rounding point, so round-half-to-even sends it to an exact zero
/// instead of holding it at a fixed point. `0.75`'s quarter-ulp step falls short
/// of the point outright, and `floor(D/2)` is `0` there anyway.
///
/// Against all of that, a `(1 + alpha * D)` that is a flat `2` at every one of
/// those coefficients. `alpha = 0.05` stalls at `9 * eta` where the general unit charges
/// `2 * eta`, so the oldest weight sits at `4.5x` its own supposed bound and one
/// `2^400` component on it clears the gate;
/// `the_oldest_weights_charge_is_not_damped_by_alpha` drives that.
///
/// **The unit quantizes the coefficient, not the product.** `MIN_NORMAL *
/// EPSILON` is `2^-1074`, the smallest positive `C` there is, so
/// `coefficient * (MIN_NORMAL * EPSILON)` rounds the *coefficient* to the nearest
/// integer. Soundness therefore needs `round(D) >= D / 2`, which holds for every
/// `D >= 1` — and `D = 1 / (1 - b) >= 1` always. It is also why the general
/// term's `(1 + alpha * D)` is a flat `2 * 2^-1074` at **every** coefficient
/// (`alpha * D` is about `1` by construction) and so could never have tracked a
/// stall that grows as `D / 2`. `b < 1` implies `1 - b >= 2^-53`, so `D <= 2^53` and
/// `D * (MIN_NORMAL * EPSILON) <= 2^-1021`: always finite, never overflowing, and
/// normal for any coefficient small enough to matter.
///
/// A `w_i` that flushed to zero is covered by its own position's line rather than
/// by a case of its own: `fl(alpha * p_k) == 0` means `alpha * p_k` was under half
/// the spacing, so `W_i` is under `(eta/2) * (1 + alpha * D)` up to the relative
/// factor already charged; and `p_(n-1) == 0` means `b^(n-1)` was under
/// `(eta/2) * D` by the chain bound above, which is `W_0` up to the same factor.
///
/// # Zero wherever the ladder is exact
///
/// Two certificates make `E_i` exactly `C::ZERO`, and both are structural rather
/// than measured:
///
/// - **`alpha == 1`** (so `b == 0`) **and `alpha == 0`**: the two degenerate
///   ladders. The first keeps the newest window at an exact `1` and every other
///   weight at an exact zero; the second keeps the oldest at an exact `1` and does
///   the same to the rest. Both are answered before anything is read — charging
///   either absolute unit against a `2^400` component sitting on one of
///   `alpha == 1`'s exact zeros would refuse an entirely determined fold.
/// - **`alpha >= 1/2` with a power-of-two `b`**: Sterbenz makes `1 - alpha` exact
///   for `alpha >= 1/2`, so `b` *is* the ideal complement rather than a rounding
///   of it, and multiplying by a power of two moves only the exponent — so every
///   `p_k`, and every `alpha * p_k`, is exact for as long as it is representable
///   and the whole *relative* part is zero. Both halves of that test are needed:
///   `alpha = 1/2 - 2^-54` also has `fl(1 - alpha) = 1/2`, and there the exact
///   complement is `1/2 + 2^-54` and the chain does drift. This is what keeps
///   `alpha = 0.5`'s published dyadic range bit-identical at the *gate* as well as
///   in the ladder, and it is why `alpha = 1 - 2^-43`'s subnormal witness is
///   unchanged to the bit.
///
/// The absolute part survives a power-of-two `b` at both positions, and must: the
/// ladder is exact only while its value is representable, and `alpha = 1 - 2^-43`
/// walks straight onto the flush boundary. It is the two *coefficients* that
/// differ there, never the certificate — the relative part is zero at every index
/// and the absolute one is charged at every index that flushed, window 0 by its
/// own line.
///
/// `b == 1` — the chain that never decays, reached by every `alpha <= 2^-54` — is
/// **not** a third certificate, and the note that once called it one was wrong.
/// Nothing is ever *formed* in the subnormal range there, so both absolute units
/// are zero by their own condition; but the *ideal* ladder does still decay, by
/// about `alpha` per step, while every materialized weight stays an identical
/// `alpha`.
/// That is the complement rounding at its largest, and the relative part charges
/// it `k` times over, as it must.
///
/// `D` — the damping both units are written over — is bounded by the geometric
/// sum alone and not also by the window count, because the count can never be
/// the smaller of the two *here*: it binds only when
/// `n < 1 / (1 - b) ~ 1 / alpha`, while reaching this regime at all needs
/// `n * log2(1 / b) > 1022`, and `log2(1 / b) ~ 1.4427 * alpha` turns that into
/// `n > 708 / alpha > 708 * n`. A cap that cannot bind would read as
/// load-bearing while being unreachable, which is the same reason
/// [`normalizing_shift`] carries no second guard.
fn ema_formation_slack<C: Real>(weights: &[C], embeddings: &[&[C]], alpha: C, b: C) -> C {
  // The two degenerate ladders, each exact at every index: `alpha == 1` keeps the
  // newest window at an exact `1` and makes every other weight an exact zero,
  // and `alpha == 0` keeps the oldest at an exact `1` and does the same to the
  // rest. Charging either would be pure over-rejection, and for `alpha == 1` the
  // absolute unit below would land on `n - 1` weights that are exactly their own
  // ideals.
  if b == C::ZERO || alpha == C::ZERO {
    return C::ZERO;
  }
  let n = weights.len();
  // Sterbenz plus a power-of-two complement: the chain is exact at every
  // representable index, so only the flush below the exponent range is left.
  let half = C::from_f32(0.5);
  let exact_chain = alpha >= half && C::ONE.ldexp(b.exponent()) == b;
  // Two absolute units over one damping, because the ladder builds `weights[0]`
  // differently from every other entry and a unit derived for one position is not
  // a bound at the other. `w_i = fl(alpha * p_k)` for `i >= 1`: that last
  // multiplication rounds once undamped (the `1`) and damps every chain rounding
  // by `alpha` (the `alpha * D`). `w_0 = p_(n-1)` has no such multiplication, so
  // its chain roundings arrive undamped and its unit is the bare `D`, whose own
  // leading term is the last chain step. Both are `(EPSILON/2) * MIN_NORMAL`
  // times their coefficient charged at twice that size, in the one grouping that
  // keeps them representable: `MIN_NORMAL * EPSILON` is `2^-1074` exactly, and
  // every coefficient here is at least `1`. They coincide at `alpha = 1/2`
  // (`D = 2`), which is why splitting them moves no dyadic verdict.
  //
  // `b == C::ONE` is the chain that never decays, so no weight is ever formed in
  // the subnormal range — and it is the one value at which the damping would
  // divide by zero.
  let (absolute, absolute_oldest) = if b < C::ONE {
    let damping = C::ONE / (C::ONE - b);
    (
      // Grouped exactly as it was before the oldest window got a unit of its
      // own, so the general term stays bit-identical.
      (C::ONE + alpha * damping) * (C::MIN_NORMAL * C::EPSILON),
      damping * (C::MIN_NORMAL * C::EPSILON),
    )
  } else {
    (C::ZERO, C::ZERO)
  };
  let mut slack = C::ZERO;
  for (i, (&w, emb)) in weights.iter().zip(embeddings).enumerate() {
    // `(2k + 2) * EPSILON * w`: the integer is exact to `2^41` and its product
    // with `EPSILON` is a power-of-two scaling, so the only rounding is the last
    // multiply. Formed in this order so a subnormal `w` is reached last — and
    // where that product does underflow, the weight is subnormal and the absolute
    // unit below is the larger of the two by orders anyway.
    let mut error = if exact_chain {
      C::ZERO
    } else {
      C::from_f64(2.0 * ((n - 1 - i) as f64) + 2.0) * C::EPSILON * w
    };
    if w < C::MIN_NORMAL {
      // The oldest window is the one the backward pass leaves without an `alpha`
      // factor, so it is the one the general unit does not bound.
      error = error + if i == 0 { absolute_oldest } else { absolute };
    }
    // `l2_norm` rather than a componentwise sum: it is the norm the residue is
    // bounded by, and it is the same scale-aware spelling the gate measures
    // `||M||` with. Skipped where the error is an exact zero, which is every
    // window of an exact ladder that has not flushed — so a dyadic `alpha` still
    // pays for no pass at all.
    if error > C::ZERO {
      slack = slack + error * l2_norm(emb);
    }
  }
  slack
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
/// subnormal *products* make `16 * EPSILON * ||M||` underflow; what neither term
/// can cover is a *weight* that is not the ideal one to within `u` — a chain of
/// multiplications, or one rounded onto the subnormal grid, both of which only
/// [`EmaRenormalized`] forms — so that policy hands in `weight_slack`, the third
/// term of the threshold, `C::ZERO` for every other built-in and therefore
/// invisible in their verdicts. See the module's
/// [A weight the fold did not form](self#a-weight-the-fold-did-not-form).
/// The bound is the crate's one accuracy claim; see the module
/// [Input domain](self#input-domain) note for the proof.
///
/// `weight_slack` is a closure rather than a value because it reads the
/// components: it runs once, after [`check_inputs`] has admitted them and before
/// a single product is formed.
fn weighted_sum_renorm<C: Real>(
  embeddings: &[&[C]],
  coverages: &[f64],
  dim: usize,
  weight: impl Fn(usize, &[C]) -> C,
  weight_slack: impl FnOnce(&[&[C]]) -> C,
) -> Result<Vec<C>, WinditError> {
  // `dim` is caller-supplied, but this has just proved it is the length of an
  // embedding that exists, so the accumulators below are bounded by real data.
  check_inputs(embeddings, coverages, dim)?;
  // Asked only once, and only after the domain check, because it reads the
  // components: the term a policy owes the gate for the gap between the weights
  // it materialized and the ones it intends (see [`ema_formation_slack`]).
  // `C::ZERO` for the three policies whose weight is one rounding of an ideal, and
  // adding an exact zero to `tau` below leaves their verdicts bit-for-bit where
  // they were.
  let slack = weight_slack(embeddings);
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
  // note). With it, exact cancellation (`||exact|| = 0`) is caught at every
  // ordering, tier structure, and weight range for which each weight is its own
  // ideal to within `u`; `slack` is the third term, and covers the one policy
  // whose weights are not (module *A weight the fold did not form*). It is an
  // exact `C::ZERO` for the other three, so their thresholds are unchanged to the
  // bit. Wherever the fold's heaviest
  // window carries mass of its own the floor sits far under
  // `16 * EPSILON * ||M||` and changes no verdict; only a fold whose whole mass
  // rides on a far lighter weight reaches it.
  let tau = C::from_f32(16.0) * C::EPSILON * l2_norm(&mag) + C::MIN_GATE_THRESHOLD + slack;
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

/// The smallest nonzero absolute component of `v`, or `ZERO` when every
/// component is zero (or `v` is empty).
///
/// The lower companion to [`max_magnitude`], and NaN loses here for the same
/// reason: `m > ZERO` is false for it, so it never becomes the minimum and
/// reaches the caller's own arithmetic instead.
fn min_positive_magnitude<C: Real>(v: &[C]) -> C {
  let mut min = C::ZERO;
  for &x in v {
    let m = x.abs();
    if m > C::ZERO && (min == C::ZERO || m < min) {
      min = m;
    }
  }
  min
}

/// The binary exponent of the smallest positive normal `f64`, `2^-1022`.
///
/// Named rather than written into [`normalizing_shift`] as a literal, and pinned
/// by a test against `f64::MIN_POSITIVE`: it is the boundary below which a
/// division's error stops being relative, which is the only reason that function
/// exists.
const MIN_NORMAL_EXPONENT: i32 = -1022;

/// The common power of two [`CoverageWeightedMean`] lifts its coverage slice by
/// before dividing, so that no weight is ever *formed* in the subnormal range.
///
/// A weight is a ratio, and the ratio of two in-domain coverages can be
/// arbitrarily small — `f64::from_bits(1)` against `0.75` is `(4/3) * 2^-1074`.
/// Materializing such a ratio rounds it *absolutely*, to the subnormal grid: the
/// intended `(4/3) * 2^-1074` and `(8/3) * 2^-1074` become `2^-1074` and
/// `3 * 2^-1074`, a relative error of a quarter and an eighth. Nothing
/// downstream can recover that — a compensated sum is exact for subnormal
/// operands but sums the terms it is given, and the determinacy gate measures a
/// residue against the mass that produced it, not against the weights that were
/// meant. An exactly cancelling in-domain fold then leaves a residue far above
/// the gate and is reported as a direction.
///
/// Lifting the whole slice by one shared `2^shift` first is what stops that. The
/// lift is a common positive factor, so it changes no ratio and no answer; it is
/// exact, because `coverages[i] <= 1` and `shift <= 53` put every lifted value at
/// or under `2^53` with its significand untouched (a subnormal scaled up by a
/// power of two is exact); and it is chosen so the smallest nonzero quotient
/// lands at or above `2^-1022`, where correct rounding is relative to at most the
/// unit roundoff. The largest weight is then exactly `2^shift` — `ldexp(m, s) / m`
/// is `2^s` to the bit — so the fold still reads only the ratios, whatever scale
/// the caller's slice arrived in.
///
/// The lift does not make the weight *exact*; the division after it still rounds.
/// It makes that rounding **relative**, which is the property the fold's error
/// bound rests on and the one a subnormal quotient destroys. See
/// [`CoverageWeightedMean`]'s own note for why the exact alternative — dividing by
/// `2^exponent(max)` and dropping the division entirely — was not taken.
///
/// **`shift` is zero unless the smallest nonzero weight would itself be
/// subnormal**, so this is the identity on every slice a [`WindowPlan`] can
/// produce (whose coverages are at worst `1 / usize::MAX` apart) and on every
/// slice whose weights were already sound. Reaching a nonzero `shift` takes a
/// weight *ratio* past `2^1022`, hand-built by a direct
/// [`aggregate_values`](AggregatePolicy::aggregate_values) caller.
///
/// Zero when `coverages` has no scale to speak of — all zero, or a largest that
/// is not finite. Both are rejected by [`check_inputs`] before any weight is read,
/// so the guard below is about keeping this function total on its own rather than
/// about a reachable verdict.
///
/// There is deliberately **no** second guard on the smallest: past that first
/// one some component has a magnitude above zero, and [`min_positive_magnitude`]
/// only ever assigns such a magnitude, so it cannot return zero here. A guard
/// there would read as load-bearing while being unreachable.
///
/// [`WindowPlan`]: crate::plan::WindowPlan
fn normalizing_shift(coverages: &[f64], largest: f64) -> i32 {
  if !(largest > 0.0 && largest.is_finite()) {
    return 0;
  }
  let smallest = min_positive_magnitude(coverages);
  // The smallest weight the fold would form, at the resolution it would form it
  // in. Taking the *quotient's* exponent rather than the difference of the two
  // operands' is what makes the lift itself scale-invariant: scaling every
  // coverage by a factor that leaves the slice exactly proportional leaves this
  // ratio's exact value — and so its correctly rounded value, and so `shift` —
  // untouched. An out-of-domain `largest` can drive this quotient to zero, whose
  // exponent is not below the boundary, so no lift is attempted before
  // `check_inputs` rejects the slice.
  let smallest_weight = (smallest / largest).exponent();
  if smallest_weight < MIN_NORMAL_EXPONENT {
    // One bit of headroom above the boundary: the quotient whose exponent this
    // is was itself rounded, so its exact value can sit just under `2^exponent`.
    // `smallest_weight >= -1074`, so `shift <= 53` and no lifted coverage or
    // product can leave the range.
    MIN_NORMAL_EXPONENT + 1 - smallest_weight
  } else {
    0
  }
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
