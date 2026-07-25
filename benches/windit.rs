//! Temporal smoothing and segmentation benchmarks.
//!
//! Covers the smoothing policies (`Identity` copy baseline, `Ema`) and the
//! segmentation paths (the gate-driven batch `GatePolicy::segment` composition,
//! the incremental `Segmenter` streaming drive, and `Threshold` for context) over
//! representative lengths and run patterns, reporting element throughput. The
//! `segment/hysteresis_batch` versus `segment/streaming` pair contrasts the batch
//! and incremental drivers of the one shared state machine.
//!
//! Gated on the heap tier: the batch drivers do not exist without it, so the
//! featureless build compiles to an empty `main`.

#[cfg(any(feature = "std", feature = "alloc"))]
mod temporal {
  use std::hint::black_box;

  use criterion::{BenchmarkId, Criterion, Throughput};
  use windit::{
    plan::Span,
    segment::{GatePolicy, Hysteresis, SegmentOptions, Segmenter, Threshold},
    smooth::{Ema, Identity, SmoothPolicy},
    windowed::Windowed,
  };

  /// Representative sequence lengths, in windows.
  const LENGTHS: [usize; 3] = [1_024, 16_384, 262_144];

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

  criterion::criterion_group!(smooth_benches, smooth_identity, smooth_ema);
  criterion::criterion_group!(
    segment_benches,
    segment_hysteresis_batch,
    segment_streaming,
    segment_threshold
  );
}

#[cfg(any(feature = "std", feature = "alloc"))]
criterion::criterion_main!(temporal::smooth_benches, temporal::segment_benches);

#[cfg(not(any(feature = "std", feature = "alloc")))]
fn main() {}
