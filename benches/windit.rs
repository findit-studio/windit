//! Temporal smoothing, gating, segmentation, and decoding benchmarks.
//!
//! Covers the smoothing policies (`Identity` copy baseline, `Ema`, `CadenceEma`,
//! and the embedding-wide `VectorEma`), the gates and their combinators
//! (`Threshold`, `Hysteresis`, `Vote`, `Dwell`, `Hangover`), the segmentation
//! drivers (the batch `GatePolicy::segment` composition, a materialized two-pass
//! drive, and the incremental `Segmenter` push/finish drive), the two
//! `longest_run` definitions, and the `Decoder` pipeline end to end, over
//! representative lengths and run patterns, reporting element throughput.
//!
//! # The comparable pairs, and the one variable each changes
//!
//! A pair whose arms differ in two things cannot support a claim about either,
//! so every declared pair below holds everything fixed but one variable. Where a
//! pair claims its arms compute the *same* output, the second arm
//! **asserts that equality against the first, outside its timed loop** — so
//! `cargo bench --bench windit -- --test`, which runs every benchmark once
//! without measuring, is an executable gate on the equivalences these comments
//! claim rather than a promise in a comment. Nothing here claims a universal
//! speedup; each pair prices one change on this input set and this machine.
//!
//! | pair | held fixed | the one variable |
//! |---|---|---|
//! | `segment/hysteresis_batch` / `_two_pass` / `_streaming` | `Hysteresis(0.6, 0.3)`, `SegmentOptions`, input, `Vec<Range>` output | driver shape: fused batch, materialized two-pass, manual push/finish |
//! | `smooth/cadence_ema` / `_streaming` | `CadenceEma(TAU)`, input, `Vec<Windowed<f32>>` output | driver shape: batch convenience vs manual `Smoother::push` |
//! | `smooth/vector_ema` / `_streaming` | `VectorEma(VECTOR_ALPHA)`, input, one output vector | the batch method's per-window `Windowed<V>` clone |
//! | `segment/longest_run_fold` / `_materialized` | `>= 0.5`, `SegmentOptions`, input, `Option<Range>` answer | keeping one range vs collecting every range first |
//! | `decode/identity_threshold` / `decode/cadence_threshold` | `Threshold(0.5)`, `SegmentOptions`, input, sink | the smoother: `Identity` vs `CadenceEma` |
//! | `decode/cadence_threshold` / `decode/hangover_dwell_vote` | `CadenceEma(TAU)`, `SegmentOptions`, input, sink | the gate: bare `Threshold` vs `Hangover(Dwell(Vote))` |
//!
//! The last pair is the one whose arms **do not** compute the same thing, and it
//! is labelled that way rather than called equivalent: a vote/dwell/hangover
//! stack carries history a bare threshold does not, so it decides differently as
//! well as costing more. `decode_gate_stack_arms_decide_differently` pins that
//! divergence on the smallest witness — one window scoring `0.6`, which the
//! threshold arm accepts and the stack arm does not — so the delta is read as
//! "what the headline stack costs", never as overhead over an identical answer.
//!
//! # Allocation is counted elsewhere, on purpose
//!
//! These benchmarks report throughput only. The allocation calls and bytes of
//! the same paths are pinned by counting/refusing global allocators in
//! `tests/segment_alloc.rs`, `tests/segment_longest_run_alloc.rs`,
//! `tests/smooth_alloc.rs`, and `tests/decode_alloc.rs`, which assert exact
//! counts and run in the ordinary test job. That split is deliberate: an
//! allocation count is an exact integer a test can assert, where a benchmark
//! mean is a machine-dependent estimate — so the streaming arms here collect the
//! same output their batch partners do (keeping the pair one-variable), and
//! their *zero-allocation* property is proven by the allocator that refuses
//! rather than inferred from a timing gap.
//!
//! Gated on the heap tier: the batch drivers do not exist without it, so the
//! featureless build compiles to an empty `main`.

#[cfg(any(feature = "std", feature = "alloc"))]
mod temporal {
  use std::hint::black_box;

  use criterion::{BenchmarkId, Criterion, Throughput};
  use windit::{
    decode::Decoder,
    plan::Span,
    segment::{
      longest_run, runs, Dwell, Gate, GatePolicy, Hangover, Hysteresis, Range, SegmentOptions,
      Segmenter, Threshold, Vote,
    },
    smooth::{CadenceEma, Ema, Identity, SmoothPolicy, Smoother, VectorEma},
    windowed::{Vector, Windowed},
    WinditError,
  };

  /// Representative sequence lengths, in windows.
  const LENGTHS: [usize; 3] = [1_024, 16_384, 262_144];

  /// `CadenceEma`'s element-denominated time constant, shared by the batch and
  /// streaming benchmarks so that pair measures the driver and not the
  /// configuration. Eight elements is a few windows at the unit cadence these
  /// inputs carry, which keeps the derived coefficient off both degenerate ends
  /// — neither saturated at `1.0` nor small enough to be absorbed by the state.
  const TAU: f32 = 8.0;

  /// The vector smoother's coefficient, held strictly inside `(0, 1)` so every
  /// window charges the recurrence — an `alpha` at either end takes `ema_step`'s
  /// exact-copy branch and would price a filter nobody runs.
  const VECTOR_ALPHA: f64 = 0.3;

  /// Embedding widths. 512 is the audio-embedding case this group exists for;
  /// 64 is the small-model contrast that separates the fixed per-window cost
  /// from the per-component arithmetic.
  const DIMS: [usize; 2] = [64, 512];

  /// Sequence lengths for the vector group, deliberately shorter than the scalar
  /// `LENGTHS`: one 512-wide window is 2 KiB of storage, so 4096 of them is
  /// already 8 MiB of input before the smoother allocates anything.
  const VECTOR_LENGTHS: [usize; 2] = [256, 4_096];

  /// xorshift64 — deterministic and dependency-free; the seed must be nonzero.
  fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
  }

  fn unit_span(i: usize) -> Span {
    Span::new(i, 1, 1)
  }

  /// Period-8 blocks (four active `0.9` then four inactive `0.1`) — many short
  /// runs, stressing run construction.
  fn dense(len: usize) -> Vec<Windowed<f32>> {
    (0..len)
      .map(|i| Windowed::new(if i % 8 < 4 { 0.9 } else { 0.1 }, unit_span(i)))
      .collect()
  }

  /// Four short active blocks totalling about 2% of the sequence — long quiet
  /// scans between them.
  fn sparse(len: usize) -> Vec<Windowed<f32>> {
    let block = (len / 200).max(1);
    let quarter = (len / 4).max(1);
    (0..len)
      .map(|i| Windowed::new(if i % quarter < block { 0.9 } else { 0.1 }, unit_span(i)))
      .collect()
  }

  /// xorshift-generated scores in `[0, 1)` — realistic branch behaviour around
  /// the thresholds.
  fn noisy(len: usize) -> Vec<Windowed<f32>> {
    let mut state: u64 = 0x1234_5678_9ABC_DEF1;
    (0..len)
      .map(|i| {
        let v = (xorshift(&mut state) >> 40) as f32 / (1u32 << 24) as f32;
        Windowed::new(v, unit_span(i))
      })
      .collect()
  }

  /// The three patterns at each length, labelled for the benchmark id.
  fn inputs() -> Vec<(&'static str, usize, Vec<Windowed<f32>>)> {
    let mut out = Vec::new();
    for &len in &LENGTHS {
      out.push(("dense", len, dense(len)));
      out.push(("sparse", len, sparse(len)));
      out.push(("noisy", len, noisy(len)));
    }
    out
  }

  /// Run `f` over every (pattern, length) input in its own throughput-reporting
  /// benchmark, building each input outside the timed loop.
  fn each_input<R>(c: &mut Criterion, group: &str, f: impl Fn(&[Windowed<f32>]) -> R) {
    let mut g = c.benchmark_group(group);
    for (pattern, len, input) in inputs() {
      g.throughput(Throughput::Elements(len as u64));
      g.bench_with_input(BenchmarkId::new(pattern, len), &input, |b, input| {
        b.iter(|| black_box(f(black_box(input.as_slice()))));
      });
    }
    g.finish();
  }

  /// [`each_input`], plus the parity that makes the arm comparable: on every
  /// input, `f`'s output must equal `reference`'s — the partner arm's — and the
  /// check runs *outside* `b.iter`, so it costs no measured time and cannot be
  /// optimized into the timed region.
  ///
  /// This is what keeps the pair table in the module documentation honest. It is
  /// executable: `cargo bench --bench windit -- --test` runs every benchmark
  /// once, and a divergence fails the run rather than quietly producing two
  /// numbers that measure different work.
  fn each_input_vs<R: PartialEq + core::fmt::Debug>(
    c: &mut Criterion,
    group: &str,
    f: impl Fn(&[Windowed<f32>]) -> R,
    reference: impl Fn(&[Windowed<f32>]) -> R,
  ) {
    let mut g = c.benchmark_group(group);
    for (pattern, len, input) in inputs() {
      assert_eq!(
        f(input.as_slice()),
        reference(input.as_slice()),
        "{group} is not comparable with its partner arm on {pattern}/{len}"
      );
      g.throughput(Throughput::Elements(len as u64));
      g.bench_with_input(BenchmarkId::new(pattern, len), &input, |b, input| {
        b.iter(|| black_box(f(black_box(input.as_slice()))));
      });
    }
    g.finish();
  }

  /// The hysteresis gate every arm of the segmentation pair uses. Named once so
  /// the three drivers cannot drift onto different gate semantics — the defect
  /// this pair previously had, where the batch arm latched on `(0.6, 0.3)` and
  /// the streaming arm applied a bare `>= 0.5`.
  fn bench_hysteresis() -> Hysteresis {
    Hysteresis::new(0.6, 0.3)
  }

  /// The shipped pass-through smoother: it copies scores through unchanged, so the
  /// benchmark measures the cost of producing the output vector with no smoothing.
  /// Not a quality claim — `Identity` is the least-assumptive score filter, never
  /// universally the most accurate one.
  fn smooth_identity(c: &mut Criterion) {
    each_input(c, "smooth/identity", |s| Identity::new().smooth(s).unwrap());
  }

  fn smooth_ema(c: &mut Criterion) {
    each_input(c, "smooth/ema", |s| Ema::new(0.2).smooth(s).unwrap());
  }

  /// The cadence-portable EMA through the same batch driver as `smooth/ema`. Its
  /// coefficient is derived per window from the span distance — an `expm1f` and
  /// an `f64` accumulate — where `Ema`'s is a constant, so the pair prices what
  /// cadence portability costs.
  fn smooth_cadence_ema(c: &mut Criterion) {
    each_input(c, "smooth/cadence_ema", |s| {
      CadenceEma::new(TAU).smooth(s).unwrap()
    });
  }

  /// The same filter driven one window at a time through `Smoother::push`, into
  /// the same `Vec<Windowed<f32>>` the batch convenience returns.
  ///
  /// Both arms run one recurrence per window and build one output vector, so the
  /// pair prices the driver and nothing else. It used to fold to a count, which
  /// made the delta batch-driver *plus* an output vector — two variables. The
  /// streaming path's zero-allocation property is asserted under a refusing
  /// global allocator in `tests/smooth_alloc.rs`, which is where an exact count
  /// belongs.
  fn smooth_cadence_ema_streaming(c: &mut Criterion) {
    each_input_vs(
      c,
      "smooth/cadence_ema_streaming",
      |s| {
        let mut sm = CadenceEma::new(TAU).smoother();
        let mut out: Vec<Windowed<f32>> = Vec::with_capacity(s.len());
        for w in s {
          out.push(sm.push(*w).unwrap());
        }
        out
      },
      |s| CadenceEma::new(TAU).smooth(s).unwrap(),
    );
  }

  /// A minimal `f32`-storage embedding double, L2-normalized on construction —
  /// the shape a real 512-dimension audio or text embedding has. Benchmarks see
  /// only the public API, so this is its own [`Vector`] implementor rather than a
  /// reach into the crate's test doubles.
  ///
  /// `Clone` because the streaming arm's untimed setup hands the loop a fresh
  /// owned copy of the input on every iteration; `PartialEq`/`Debug` so the two
  /// arms' outputs can be asserted equal outside the timed region, which is what
  /// makes the pair a comparison rather than two unrelated measurements.
  #[derive(Clone, Debug, PartialEq)]
  struct BenchEmbedding(Vec<f32>);

  impl Vector for BenchEmbedding {
    type Scalar = f32;

    fn as_slice(&self) -> &[f32] {
      &self.0
    }

    fn from_unnormalized(v: &[f64]) -> Result<Self, WinditError> {
      if v.is_empty() {
        return Err(WinditError::Empty);
      }
      let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
      if !norm.is_finite() || norm == 0.0 {
        return Err(WinditError::NonFinite);
      }
      Ok(Self(v.iter().map(|x| (x / norm) as f32).collect()))
    }
  }

  /// `len` unit-norm embeddings of width `dim` over unit spans, components drawn
  /// from the same xorshift the scalar patterns use so the input is deterministic
  /// and dependency-free.
  fn embeddings(len: usize, dim: usize) -> Vec<Windowed<BenchEmbedding>> {
    let mut state: u64 = 0x0F1E_2D3C_4B5A_6978;
    (0..len)
      .map(|i| {
        let raw: Vec<f64> = (0..dim)
          .map(|_| (xorshift(&mut state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
          .collect();
        let e = BenchEmbedding::from_unnormalized(&raw).expect("nonzero random direction");
        Windowed::new(e, unit_span(i))
      })
      .collect()
  }

  /// Every (width, length) pair, built once and shared by both arms of the vector
  /// smoothing comparison.
  fn vector_inputs() -> Vec<(usize, usize, Vec<Windowed<BenchEmbedding>>)> {
    let mut out = Vec::new();
    for &dim in &DIMS {
      for &len in &VECTOR_LENGTHS {
        out.push((dim, len, embeddings(len, dim)));
      }
    }
    out
  }

  /// The vector EMA through the batch `SmoothPolicy::smooth` convenience.
  ///
  /// Paired with `smooth/vector_ema_streaming` below, which drives the same
  /// filter over the same windows *owned*. Both arms run one recurrence per
  /// window and allocate one output vector, so the gap between them is the
  /// per-window `Windowed<E>` clone that the batch method's `V: Clone` bound
  /// exists to perform — the whole of what the convenience costs an embedding
  /// consumer.
  fn smooth_vector_ema(c: &mut Criterion) {
    let mut g = c.benchmark_group("smooth/vector_ema");
    for (dim, len, input) in vector_inputs() {
      g.throughput(Throughput::Elements(len as u64));
      g.bench_with_input(
        BenchmarkId::new(format!("dim{dim}"), len),
        &input,
        |b, input| {
          b.iter(|| {
            black_box(
              VectorEma::new(VECTOR_ALPHA)
                .smooth(black_box(input.as_slice()))
                .unwrap(),
            )
          });
        },
      );
    }
    g.finish();
  }

  /// The same filter driven one owned window at a time through `Smoother::push`.
  ///
  /// The input copy is made in `iter_batched`'s untimed setup, so the measured
  /// region is the recurrence and the output vector and nothing else.
  /// `PerIteration` because a 4096-window, 512-wide batch is 8 MiB and criterion
  /// would otherwise hold many of them at once.
  ///
  /// Read the pair as a bound, not as a sharp figure. That untimed setup frees
  /// and re-takes several thousand blocks between timed regions, where the batch
  /// arm allocates and frees each clone in one steady loop — so the allocator is
  /// warmer for the arm that does *more* work, and the bias runs against this
  /// one. The difference being priced is also under two percent of either arm,
  /// which is at or below what criterion's means separate from run-to-run noise
  /// on a machine with anything else on it. The per-window figures quoted on
  /// `SmoothPolicy::smooth` come from interleaved minima and a counting
  /// allocator, which is what a difference this small needs; this pair is the
  /// version that lives in the repository and can be re-run.
  fn smooth_vector_ema_streaming(c: &mut Criterion) {
    let mut g = c.benchmark_group("smooth/vector_ema_streaming");
    for (dim, len, input) in vector_inputs() {
      // Same filter, same windows: the two arms must return the identical
      // stream, or the gap between them is not the clone.
      let streamed = {
        let mut sm = SmoothPolicy::<BenchEmbedding>::smoother(&VectorEma::new(VECTOR_ALPHA));
        let mut out = Vec::with_capacity(input.len());
        for w in input.clone() {
          out.push(sm.push(w).unwrap());
        }
        out
      };
      assert_eq!(
        streamed,
        VectorEma::new(VECTOR_ALPHA)
          .smooth(input.as_slice())
          .unwrap(),
        "smooth/vector_ema_streaming is not comparable with its partner arm at dim{dim}/{len}"
      );

      g.throughput(Throughput::Elements(len as u64));
      g.bench_with_input(
        BenchmarkId::new(format!("dim{dim}"), len),
        &input,
        |b, input| {
          b.iter_batched(
            || input.clone(),
            |owned| {
              let mut sm = SmoothPolicy::<BenchEmbedding>::smoother(&VectorEma::new(VECTOR_ALPHA));
              let mut out = Vec::with_capacity(owned.len());
              for w in owned {
                out.push(sm.push(w).unwrap());
              }
              black_box(out)
            },
            criterion::BatchSize::PerIteration,
          );
        },
      );
    }
    g.finish();
  }

  /// The fused batch segmentation: `GatePolicy::segment` runs the hysteresis gate
  /// and the `Segmenter` in one pass over the input.
  ///
  /// The reference arm of the segmentation trio — `_two_pass` and `_streaming`
  /// below assert their output against it, and all three share the gate, the
  /// `SegmentOptions`, the input, and the `Vec<Range>` sink, so the only variable
  /// across them is the shape of the drive.
  fn segment_hysteresis_batch(c: &mut Criterion) {
    each_input(c, "segment/hysteresis_batch", segment_batch_arm);
  }

  /// The fused batch arm as a plain function, so the other two arms can assert
  /// against it.
  fn segment_batch_arm(s: &[Windowed<f32>]) -> Vec<Range> {
    bench_hysteresis()
      .segment(&SegmentOptions::new(), s)
      .unwrap()
  }

  /// The materialized two-pass drive: gate every window into a `Vec<bool>`
  /// first, then segment that. Same gate, same options, same output — this arm
  /// exists to price the O(n) intermediate decision vector the fused driver
  /// avoids, which is the concrete resource difference between the two shapes.
  fn segment_hysteresis_two_pass(c: &mut Criterion) {
    each_input_vs(
      c,
      "segment/hysteresis_two_pass",
      |s| {
        let mut gate = bench_hysteresis().gate();
        let decisions: Vec<bool> = s.iter().map(|w| gate.push(w).unwrap()).collect();
        let mut seg = Segmenter::new(SegmentOptions::new());
        let mut out: Vec<Range> = Vec::new();
        for (w, &active) in s.iter().zip(decisions.iter()) {
          if let Some(r) = seg.push(active, w.span()).unwrap() {
            out.push(r);
          }
        }
        out.extend(seg.finish());
        out
      },
      segment_batch_arm,
    );
  }

  /// The incremental `Segmenter` driven one window at a time through the same
  /// hysteresis gate, collecting the same `Vec<Range>` — the manual push/finish
  /// path a streaming consumer writes.
  ///
  /// It collects rather than counting so the pair changes one variable. That the
  /// `Segmenter` and the gate themselves allocate nothing is not measured here at
  /// all; it is asserted exactly, under a refusing global allocator, in
  /// `tests/segment_alloc.rs`.
  fn segment_hysteresis_streaming(c: &mut Criterion) {
    each_input_vs(
      c,
      "segment/hysteresis_streaming",
      |s| {
        let mut gate = bench_hysteresis().gate();
        let mut seg = Segmenter::new(SegmentOptions::new());
        let mut out: Vec<Range> = Vec::new();
        for w in s {
          let active = gate.push(w).unwrap();
          if let Some(r) = seg.push(active, w.span()).unwrap() {
            out.push(r);
          }
        }
        out.extend(seg.finish());
        out
      },
      segment_batch_arm,
    );
  }

  /// `longest_run` as shipped: one incumbent `Range` folded from the `Segmenter`
  /// emissions, keeping nothing else.
  ///
  /// Paired with `segment/longest_run_materialized`, which collects every
  /// finalized range and only then scans — the definition this replaced. Same
  /// predicate, same options, same input, same `Option<Range>` answer; the one
  /// variable is whether the output runs are stored. `dense` at 262,144 windows
  /// is the high-run-count case the pair exists for: 32,768 finalized ranges for
  /// the materialized arm to hold and one for the fold.
  fn segment_longest_run_fold(c: &mut Criterion) {
    each_input_vs(
      c,
      "segment/longest_run_fold",
      |s| longest_run(s, |&v| v >= 0.5, &SegmentOptions::new()).unwrap(),
      longest_run_materialized_arm,
    );
  }

  /// The materializing definition of `longest_run`, kept here as the fold's
  /// partner arm and written only against the public API.
  fn longest_run_materialized_arm(s: &[Windowed<f32>]) -> Option<Range> {
    let mut best: Option<Range> = None;
    for r in runs(s, |&v| v >= 0.5, &SegmentOptions::new()).unwrap() {
      match best {
        Some(b) if b.len() >= r.len() => {}
        _ => best = Some(r),
      }
    }
    best
  }

  fn segment_longest_run_materialized(c: &mut Criterion) {
    each_input(
      c,
      "segment/longest_run_materialized",
      longest_run_materialized_arm,
    );
  }

  fn segment_threshold(c: &mut Criterion) {
    each_input(c, "segment/threshold", |s| {
      Threshold::new(0.5)
        .segment(&SegmentOptions::new(), s)
        .unwrap()
    });
  }

  /// The N-of-M vote gate: a shift, a mask, and a popcount per window against
  /// `segment/threshold`'s single compare, over the same batch driver.
  fn segment_vote(c: &mut Criterion) {
    each_input(c, "segment/vote", |s| {
      Vote::new(3, 5, 0.5)
        .segment(&SegmentOptions::new(), s)
        .unwrap()
    });
  }

  /// Onset confirmation wrapped around a threshold gate. `Dwell` reads every span
  /// to fold its coverage horizon, so this prices a span-reading combinator
  /// against the bare `segment/threshold` it wraps.
  fn segment_dwell(c: &mut Criterion) {
    each_input(c, "segment/dwell", |s| {
      Dwell::new(Threshold::new(0.5), 3)
        .segment(&SegmentOptions::new(), s)
        .unwrap()
    });
  }

  /// Release hold wrapped around a threshold gate — `Dwell`'s mirror, and the
  /// other span-reading combinator.
  fn segment_hangover(c: &mut Criterion) {
    each_input(c, "segment/hangover", |s| {
      Hangover::new(Threshold::new(0.5), 3)
        .segment(&SegmentOptions::new(), s)
        .unwrap()
    });
  }

  /// Drive a decoder to exhaustion and count what it finalized. The shared sink
  /// of all three `decode/*` arms, so the two decode pairs vary only their named
  /// stage; it counts rather than collecting because no arm's partner collects
  /// either.
  fn drive<S: Smoother<f32>, G: Gate<f32>>(
    mut dec: Decoder<S, G, f32>,
    s: &[Windowed<f32>],
  ) -> usize {
    let mut finalized = 0usize;
    for w in s {
      if dec.push(*w).unwrap().finalized().is_some() {
        finalized += 1;
      }
    }
    finalized + dec.finish().count()
  }

  /// The `Decoder` at its floor: a pass-through smoother and a single-compare
  /// gate — the pipeline's own per-window cost with neither stage doing work.
  ///
  /// The first arm of the smoother pair: `decode/cadence_threshold` holds the
  /// gate, the options, the input, and the sink fixed and changes only the
  /// smoother, so the gap between them is what `CadenceEma` costs inside the
  /// pipeline.
  fn decode_identity_threshold(c: &mut Criterion) {
    each_input(c, "decode/identity_threshold", |s| {
      drive(
        Decoder::new(
          // `Identity` smooths every `V`, so the value type is named on the
          // factory call rather than inferred from the stage.
          SmoothPolicy::<f32>::smoother(&Identity::new()),
          Threshold::new(0.5).gate(),
          SegmentOptions::new(),
        ),
        s,
      )
    });
  }

  /// The pipeline's hinge arm: `CadenceEma` into a bare `Threshold`.
  ///
  /// It is the second half of the smoother pair (against
  /// `decode/identity_threshold`, gate fixed) and the first half of the gate pair
  /// (against `decode/hangover_dwell_vote`, smoother fixed). Adding it is what
  /// splits a pair that used to change both stages at once into two pairs that
  /// each change one.
  fn decode_cadence_threshold(c: &mut Criterion) {
    each_input(c, "decode/cadence_threshold", |s| {
      drive(
        Decoder::new(
          CadenceEma::new(TAU).smoother(),
          Threshold::new(0.5).gate(),
          SegmentOptions::new(),
        ),
        s,
      )
    });
  }

  /// The headline composition end to end: the same cadence-portable EMA into the
  /// canonical `Hangover(Dwell(Vote))` gate stack into the segmenter, one window
  /// at a time. Both combinators read spans, so this is also the deepest
  /// span-checking path the crate ships.
  ///
  /// Read against `decode/cadence_threshold`, whose smoother, options, input and
  /// sink are identical: the delta is the gate stack. That stack **decides
  /// differently**, not merely more slowly — it carries vote, dwell and hangover
  /// history a bare threshold has none of — so this is the cost of the headline
  /// behaviour, never overhead over an identical answer. The assertion below
  /// pins the divergence on its smallest witness so the distinction cannot decay
  /// into an equivalence claim.
  fn decode_hangover_dwell_vote(c: &mut Criterion) {
    decode_gate_stack_arms_decide_differently();
    each_input(c, "decode/hangover_dwell_vote", |s| {
      let gate = Hangover::new(Dwell::new(Vote::new(3, 5, 0.5), 3), 4);
      drive(
        Decoder::new(
          CadenceEma::new(TAU).smoother(),
          gate.gate(),
          SegmentOptions::new(),
        ),
        s,
      )
    });
  }

  /// Push one window through `dec` and collect every range it finalizes, from
  /// the push *and* from the tail — a `Step` is `#[must_use]` precisely because
  /// discarding it can drop one. Generic, because the two arms it serves nest
  /// different gate states.
  fn decide_one<S: Smoother<f32>, G: Gate<f32>>(
    mut dec: Decoder<S, G, f32>,
    w: Windowed<f32>,
  ) -> Vec<Range> {
    let mut out: Vec<Range> = dec.push(w).unwrap().finalized().into_iter().collect();
    out.extend(dec.finish());
    out
  }

  /// One window scoring `0.6`: the threshold arm finalizes `[0, 1)`, the gate
  /// stack finalizes nothing — 3-of-5 voting has seen one window, and dwell has
  /// no confirmed onset. Runs outside every timed loop.
  fn decode_gate_stack_arms_decide_differently() {
    let one = Windowed::new(0.6f32, unit_span(0));

    let threshold_out = decide_one(
      Decoder::new(
        CadenceEma::new(TAU).smoother(),
        Threshold::new(0.5).gate(),
        SegmentOptions::new(),
      ),
      one,
    );
    let stack_out = decide_one(
      Decoder::new(
        CadenceEma::new(TAU).smoother(),
        Hangover::new(Dwell::new(Vote::new(3, 5, 0.5), 3), 4).gate(),
        SegmentOptions::new(),
      ),
      one,
    );

    assert_eq!(threshold_out, [Range::new(0, 1)]);
    assert!(
      stack_out.is_empty(),
      "the gate pair is a cost measurement, not an equivalence: if the stack now \
       agrees with a bare threshold on a single 0.6 window, the comment above is stale"
    );
  }

  criterion::criterion_group!(
    smooth_benches,
    smooth_identity,
    smooth_ema,
    smooth_cadence_ema,
    smooth_cadence_ema_streaming,
    smooth_vector_ema,
    smooth_vector_ema_streaming
  );
  criterion::criterion_group!(
    segment_benches,
    segment_hysteresis_batch,
    segment_hysteresis_two_pass,
    segment_hysteresis_streaming,
    segment_longest_run_fold,
    segment_longest_run_materialized,
    segment_threshold,
    segment_vote,
    segment_dwell,
    segment_hangover
  );
  criterion::criterion_group!(
    decode_benches,
    decode_identity_threshold,
    decode_cadence_threshold,
    decode_hangover_dwell_vote
  );
}

#[cfg(any(feature = "std", feature = "alloc"))]
criterion::criterion_main!(
  temporal::smooth_benches,
  temporal::segment_benches,
  temporal::decode_benches
);

#[cfg(not(any(feature = "std", feature = "alloc")))]
fn main() {}
