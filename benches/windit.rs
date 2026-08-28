//! Temporal smoothing, gating, segmentation, and decoding benchmarks.
//!
//! Covers the smoothing policies (`Identity` copy baseline, `Ema`, `CadenceEma`,
//! and the embedding-wide `VectorEma`), the gates and their combinators
//! (`Threshold`, `Hysteresis`, `Vote`, `Dwell`, `Hangover`), the two segmentation
//! drivers (the batch `GatePolicy::segment` composition and the incremental
//! `Segmenter` streaming drive), and the `Decoder` pipeline end to end, over
//! representative lengths and run patterns, reporting element throughput. Four
//! pairs are deliberately comparable: `segment/hysteresis_batch` versus
//! `segment/streaming` contrasts the batch and incremental drivers of the one
//! shared state machine; `smooth/cadence_ema` versus
//! `smooth/cadence_ema_streaming` does the same for the smoothing side;
//! `decode/identity_threshold` versus `decode/hangover_dwell_vote` separates the
//! pipeline's own per-window cost from what the headline gate stack adds; and
//! `smooth/vector_ema` versus `smooth/vector_ema_streaming` prices the batch
//! method's per-window `Windowed<V>` clone, which is the whole of what its
//! `V: Clone` bound costs an embedding consumer.
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
      Dwell, Gate, GatePolicy, Hangover, Hysteresis, SegmentOptions, Segmenter, Threshold, Vote,
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

  /// The same filter driven one window at a time through `Smoother::push` — the
  /// zero-allocation streaming path. Returns a count so no output vector is
  /// built inside the timed loop.
  fn smooth_cadence_ema_streaming(c: &mut Criterion) {
    each_input(c, "smooth/cadence_ema_streaming", |s| {
      let mut sm = CadenceEma::new(TAU).smoother();
      let mut above = 0usize;
      for w in s {
        if *sm.push(*w).unwrap().value() >= 0.5 {
          above += 1;
        }
      }
      above
    });
  }

  /// A minimal `f32`-storage embedding double, L2-normalized on construction —
  /// the shape a real 512-dimension audio or text embedding has. Benchmarks see
  /// only the public API, so this is its own [`Vector`] implementor rather than a
  /// reach into the crate's test doubles.
  ///
  /// `Clone` because the streaming arm's untimed setup hands the loop a fresh
  /// owned copy of the input on every iteration.
  #[derive(Clone)]
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

  /// The gate-driven batch segmentation (`GatePolicy::segment` over a hysteresis
  /// gate), contrasted against the incremental drive of the same state machine
  /// below.
  fn segment_hysteresis_batch(c: &mut Criterion) {
    each_input(c, "segment/hysteresis_batch", |s| {
      Hysteresis::new(0.6, 0.3)
        .segment(&SegmentOptions::new(), s)
        .unwrap()
    });
  }

  /// The incremental `Segmenter` driven one window at a time through a threshold
  /// gate — the zero-allocation streaming push/finish path. Returns a count so
  /// no output vector is built inside the timed loop.
  fn segment_streaming(c: &mut Criterion) {
    each_input(c, "segment/streaming", |s| {
      let mut seg = Segmenter::new(SegmentOptions::new());
      let mut finalized = 0usize;
      for w in s {
        if seg.push(*w.value() >= 0.5, w.span()).unwrap().is_some() {
          finalized += 1;
        }
      }
      finalized + seg.finish().count()
    });
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

  /// Drive a decoder to exhaustion and count what it finalized, so no output
  /// vector is built inside the timed loop — `segment/streaming`'s shape, with
  /// the smoothing and gating stages folded in.
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
  /// gate, so the difference from `segment/streaming` is the pipeline's own
  /// per-window cost over a bare drive of the same segmenter.
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

  /// The headline composition end to end: a cadence-portable EMA into the
  /// canonical `Hangover(Dwell(Vote))` gate stack into the segmenter, one window
  /// at a time. Both combinators read spans, so this is also the deepest
  /// span-checking path the crate ships.
  fn decode_hangover_dwell_vote(c: &mut Criterion) {
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
    segment_streaming,
    segment_threshold,
    segment_vote,
    segment_dwell,
    segment_hangover
  );
  criterion::criterion_group!(
    decode_benches,
    decode_identity_threshold,
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
