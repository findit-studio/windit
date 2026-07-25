use std::{boxed::Box, cell::Cell, rc::Rc, vec, vec::Vec};

use super::Decoder;
use crate::{
  error::WinditError,
  plan::Span,
  segment::{
    Dwell, Gate, GatePolicy, Hangover, Hysteresis, Range, SegmentOptions, Threshold,
    ThresholdState, Vote,
  },
  smooth::{CadenceEma, Ema, EmaState, Identity, SmoothPolicy, Smoother},
  windowed::Windowed,
};

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// Build a `Windowed<f32>` sequence with a fixed geometry: window `i` starts at
/// `i * hop` and covers `len` real elements of a `window`-wide window.
fn windows(scores: &[f32], hop: usize, len: usize, window: usize) -> Vec<Windowed<f32>> {
  scores
    .iter()
    .enumerate()
    .map(|(i, &s)| Windowed::new(s, Span::new(i * hop, len, window)))
    .collect()
}

/// The P2 geometry grid as `(hop, len, window)`: unit spans, wider adjacent
/// windows, a gapped plan (hop strides past the covered length), and an
/// overlapping plan (hop below the window). Parity must hold on every one.
const GEOMETRIES: [(usize, usize, usize); 4] = [(1, 1, 1), (4, 4, 4), (4, 2, 4), (2, 4, 4)];

/// A spread of segment morphologies: default, minimum-length only, merge-gap
/// only, and both together.
fn options_grid() -> Vec<SegmentOptions> {
  vec![
    SegmentOptions::new(),
    SegmentOptions::new().with_min_len(2),
    SegmentOptions::new().with_min_len(5),
    SegmentOptions::new().with_merge_gap(2),
    SegmentOptions::new().with_merge_gap(6),
    SegmentOptions::new().with_min_len(3).with_merge_gap(4),
  ]
}

/// xorshift64 — deterministic and dependency-free; the seed must be nonzero.
fn xorshift(state: &mut u64) -> u64 {
  let mut x = *state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  *state = x;
  x
}

/// A pseudo-random `f32` in `[0, 1)` from the generator's next 24 bits.
fn next_unit(state: &mut u64) -> f32 {
  (xorshift(state) >> 40) as f32 / (1u32 << 24) as f32
}

/// A score in `[0, 1)`, but one in eight a non-finite value, so the differential
/// exercises the exact NaN/±inf propagation both the batch and streaming paths
/// carry through the smoother and the gate comparison.
fn random_score(state: &mut u64) -> f32 {
  match xorshift(state) % 16 {
    0 => f32::NAN,
    1 => f32::INFINITY,
    2 => f32::NEG_INFINITY,
    _ => next_unit(state),
  }
}

/// Hash a value with a tiny, dependency-free FNV-1a hasher. Core and alloc ship
/// no concrete `Hasher`, and this suite builds on the `alloc`-only tier too, so
/// the `Step: Hash` derive is witnessed through this rather than a std hasher.
fn hash_of<T: core::hash::Hash>(v: &T) -> u64 {
  use core::hash::Hasher;
  struct Fnv1a(u64);
  impl Hasher for Fnv1a {
    fn finish(&self) -> u64 {
      self.0
    }
    fn write(&mut self, bytes: &[u8]) {
      for &b in bytes {
        self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
      }
    }
  }
  let mut hasher = Fnv1a(0xcbf2_9ce4_8422_2325);
  v.hash(&mut hasher);
  hasher.finish()
}

/// The property the whole type exists for: for a span-monotone sequence, the
/// streaming decode (per-push `finalized` concatenated with the `finish` tail,
/// and the per-push `active` stream) equals the batch composition of the same
/// policies on **both** planes — the finalized ranges and the causal flags.
///
/// Both sides drive the same three state types, in the same order, from fresh
/// state, so the equality is by construction; this pins it against any future
/// reordering of the pipeline.
fn assert_parity<CS, CG>(cs: &CS, cg: &CG, opts: SegmentOptions, seq: &[Windowed<f32>])
where
  CS: SmoothPolicy<f32>,
  CG: GatePolicy<f32>,
{
  // Batch composition: smooth, then segment the smoothed sequence; and the flag
  // drive of a fresh gate over the same smoothed sequence.
  let smoothed = cs.smooth(seq).expect("batch smooth");
  let batch_ranges = cg.segment(&opts, &smoothed).expect("batch segment");
  let mut batch_gate = cg.gate();
  let batch_flags: Vec<bool> = smoothed
    .iter()
    .map(|w| batch_gate.push(w).expect("batch gate"))
    .collect();

  // Streaming decode over the raw sequence.
  let mut dec: Decoder<CS::Smoother, CG::Gate, f32> = Decoder::new(cs.smoother(), cg.gate(), opts);
  let mut stream_flags = Vec::new();
  let mut stream_ranges = Vec::new();
  for w in seq {
    let step = dec.push(*w).expect("streaming push");
    stream_flags.push(step.active());
    stream_ranges.extend(step.finalized());
  }
  stream_ranges.extend(dec.finish());

  assert_eq!(
    stream_ranges, batch_ranges,
    "finalized plane diverged: seq={seq:?} opts={opts:?}"
  );
  assert_eq!(
    stream_flags, batch_flags,
    "causal plane diverged: seq={seq:?} opts={opts:?}"
  );
}

/// Drive a fresh decoder over `seq` to exhaustion, returning its `(flags,
/// ranges)` — the whole-epoch streaming output used by the epoch and dyn tests.
fn fresh_run<CS, CG>(
  cs: &CS,
  cg: &CG,
  opts: SegmentOptions,
  seq: &[Windowed<f32>],
) -> (Vec<bool>, Vec<Range>)
where
  CS: SmoothPolicy<f32>,
  CG: GatePolicy<f32>,
{
  let mut dec: Decoder<CS::Smoother, CG::Gate, f32> = Decoder::new(cs.smoother(), cg.gate(), opts);
  let mut flags = Vec::new();
  let mut ranges = Vec::new();
  for w in seq {
    let step = dec.push(*w).expect("push");
    flags.push(step.active());
    ranges.extend(step.finalized());
  }
  ranges.extend(dec.finish());
  (flags, ranges)
}

// ── End-to-end parity (the oracle) ───────────────────────────────────────────

#[test]
fn parity_grid_p2_policies() {
  let patterns: [&[f32]; 6] = [
    &[],
    &[0.9],
    &[0.1, 0.9, 0.9, 0.1, 0.9],
    &[0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    &[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1],
    &[0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
  ];
  for pattern in patterns {
    for &(hop, len, window) in &GEOMETRIES {
      let seq = windows(pattern, hop, len, window);
      for opts in options_grid() {
        // The P2 policy grid: {Identity, Ema} x {Threshold, Hysteresis}. The P3
        // policies (CadenceEma, Vote, Dwell, Hangover, and the canonical nesting)
        // get the parallel `parity_grid_p3_policies`.
        assert_parity(&Identity::new(), &Threshold::new(0.5), opts, &seq);
        assert_parity(&Ema::new(0.5), &Threshold::new(0.5), opts, &seq);
        assert_parity(&Identity::new(), &Hysteresis::new(0.6, 0.3), opts, &seq);
        assert_parity(&Ema::new(0.3), &Hysteresis::new(0.7, 0.4), opts, &seq);
      }
    }
  }
}

#[test]
fn parity_randomized_with_non_finite() {
  let mut state = 0x0bad_c0de_1234_5678u64;
  for _ in 0..200 {
    let len = (xorshift(&mut state) % 32) as usize;
    let scores: Vec<f32> = (0..len).map(|_| random_score(&mut state)).collect();
    let (hop, span_len, window) = GEOMETRIES[(xorshift(&mut state) % 4) as usize];
    let seq = windows(&scores, hop, span_len, window);
    let opts = SegmentOptions::new()
      .with_min_len((xorshift(&mut state) % 6) as usize)
      .with_merge_gap((xorshift(&mut state) % 8) as usize);
    let thr = next_unit(&mut state);
    let on = next_unit(&mut state);
    let off = next_unit(&mut state);
    let alpha = next_unit(&mut state);
    match xorshift(&mut state) % 4 {
      0 => assert_parity(&Identity::new(), &Threshold::new(thr), opts, &seq),
      1 => assert_parity(&Ema::new(alpha), &Threshold::new(thr), opts, &seq),
      2 => assert_parity(&Identity::new(), &Hysteresis::new(on, off), opts, &seq),
      _ => assert_parity(&Ema::new(alpha), &Hysteresis::new(on, off), opts, &seq),
    }
  }
}

/// A valid N-of-M vote configuration drawn from the generator: `1 <= need <= of
/// <= 64`, the exact admissible range `Vote::try_new` enforces, so construction
/// never panics on a generated pair.
fn random_vote(state: &mut u64) -> (usize, usize) {
  let of = 1 + (xorshift(state) % 64) as usize; // 1..=64
  let need = 1 + (xorshift(state) % of as u64) as usize; // 1..=of
  (need, of)
}

/// Assert end-to-end parity for one gate policy against a randomly chosen
/// smoother — the smoother axis of the P3 grid. Including `CadenceEma` here
/// threads its span-derived coefficient through both the streaming and the batch
/// path, so the differential covers a span-reading smoother, not just the
/// value-only `Identity`/`Ema`.
fn parity_random_smoother<CG: GatePolicy<f32>>(
  state: &mut u64,
  cg: &CG,
  opts: SegmentOptions,
  seq: &[Windowed<f32>],
) {
  match xorshift(state) % 3 {
    0 => assert_parity(&Identity::new(), cg, opts, seq),
    1 => assert_parity(&Ema::new(next_unit(state)), cg, opts, seq),
    // tau in [1, 21): finite and positive, so `CadenceEma::new` never panics.
    _ => assert_parity(
      &CadenceEma::new(1.0 + next_unit(state) * 20.0),
      cg,
      opts,
      seq,
    ),
  }
}

#[test]
fn parity_grid_p3_policies() {
  // The full P3 policy grid: {Identity, Ema, CadenceEma} smoothers x {Threshold,
  // Hysteresis, Vote, Dwell(.), Hangover(.), Hangover(Dwell(Vote))} gates — the
  // last the canonical onset-confirm-then-release nesting.
  // Streaming must equal the batch composition on BOTH planes for every cell,
  // over the same pattern/geometry/options grid the P2 oracle uses. The equality
  // is by construction (one core, two drivers); this pins it against any future
  // reordering of the pipeline or drift in a P3 state machine.
  let patterns: [&[f32]; 6] = [
    &[],
    &[0.9],
    &[0.1, 0.9, 0.9, 0.1, 0.9],
    &[0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    &[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1],
    &[0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
  ];

  // Every gate config is exercised against all three smoothers at the current
  // `opts`/`seq`. Each gate config is `Copy`, so binding it once and referencing
  // it three times re-runs no construction.
  macro_rules! all_smoothers {
    ($gate:expr, $opts:expr, $seq:expr) => {{
      let g = $gate;
      assert_parity(&Identity::new(), &g, $opts, $seq);
      assert_parity(&Ema::new(0.5), &g, $opts, $seq);
      assert_parity(&CadenceEma::new(6.0), &g, $opts, $seq);
    }};
  }

  for pattern in patterns {
    for &(hop, len, window) in &GEOMETRIES {
      let seq = windows(pattern, hop, len, window);
      for opts in options_grid() {
        all_smoothers!(Threshold::new(0.5), opts, &seq);
        all_smoothers!(Hysteresis::new(0.6, 0.3), opts, &seq);
        all_smoothers!(Vote::new(2, 3, 0.5), opts, &seq);
        all_smoothers!(Dwell::new(Threshold::new(0.5), 2), opts, &seq);
        all_smoothers!(Hangover::new(Threshold::new(0.5), 3), opts, &seq);
        all_smoothers!(
          Hangover::new(Dwell::new(Vote::new(2, 3, 0.5), 2), 3),
          opts,
          &seq
        );
      }
    }
  }
}

#[test]
fn parity_randomized_p3_policies() {
  // 200 randomized cases over the P3 gate family (random valid parameters) x a
  // random smoother, with non-finite score injection. A distinct seed from the P2
  // randomized run so the two cover disjoint sequences.
  let mut state = 0xdece_a5ed_600d_c0deu64;
  for _ in 0..200 {
    let len = (xorshift(&mut state) % 32) as usize;
    let scores: Vec<f32> = (0..len).map(|_| random_score(&mut state)).collect();
    let (hop, span_len, window) = GEOMETRIES[(xorshift(&mut state) % 4) as usize];
    let seq = windows(&scores, hop, span_len, window);
    let opts = SegmentOptions::new()
      .with_min_len((xorshift(&mut state) % 6) as usize)
      .with_merge_gap((xorshift(&mut state) % 8) as usize);
    let thr = next_unit(&mut state);
    let on = next_unit(&mut state);
    let off = next_unit(&mut state);
    // Small element-denominated delays, so activation is reachable on the short
    // (<= 31-window) sequences rather than always suppressed.
    let confirm = (xorshift(&mut state) % 10) as usize;
    let hold = (xorshift(&mut state) % 10) as usize;
    match xorshift(&mut state) % 6 {
      0 => parity_random_smoother(&mut state, &Threshold::new(thr), opts, &seq),
      1 => parity_random_smoother(&mut state, &Hysteresis::new(on, off), opts, &seq),
      2 => {
        let (need, of) = random_vote(&mut state);
        parity_random_smoother(&mut state, &Vote::new(need, of, thr), opts, &seq);
      }
      3 => parity_random_smoother(
        &mut state,
        &Dwell::new(Threshold::new(thr), confirm),
        opts,
        &seq,
      ),
      4 => parity_random_smoother(
        &mut state,
        &Hangover::new(Threshold::new(thr), hold),
        opts,
        &seq,
      ),
      _ => {
        let (need, of) = random_vote(&mut state);
        parity_random_smoother(
          &mut state,
          &Hangover::new(Dwell::new(Vote::new(need, of, thr), confirm), hold),
          opts,
          &seq,
        );
      }
    }
  }
}

#[test]
fn chunk_partition_invariance() {
  // One decoder driven in a single uninterrupted pass is invariant to where a
  // chunked consumer pauses between pushes: the concatenated output at every
  // split point equals the batch composition. This guards against a `push` that
  // ever grows a hidden per-call dependency.
  let scores = [0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9];
  let cs = Ema::new(0.4);
  let cg = Hysteresis::new(0.6, 0.35);
  let opts = SegmentOptions::new().with_min_len(2).with_merge_gap(3);
  for &(hop, len, window) in &GEOMETRIES {
    let seq = windows(&scores, hop, len, window);
    let (want_flags, want_ranges) = fresh_run(&cs, &cg, opts, &seq);
    for split in 0..=seq.len() {
      let mut dec: Decoder<_, _, f32> = Decoder::new(cs.smoother(), cg.gate(), opts);
      let mut flags = Vec::new();
      let mut ranges = Vec::new();
      for w in &seq[..split] {
        let step = dec.push(*w).unwrap();
        flags.push(step.active());
        ranges.extend(step.finalized());
      }
      for w in &seq[split..] {
        let step = dec.push(*w).unwrap();
        flags.push(step.active());
        ranges.extend(step.finalized());
      }
      ranges.extend(dec.finish());
      assert_eq!(
        flags, want_flags,
        "flags at split {split}, geom {hop}/{len}/{window}"
      );
      assert_eq!(
        ranges, want_ranges,
        "ranges at split {split}, geom {hop}/{len}/{window}"
      );
    }
  }
}

#[test]
fn discontinuity_splits_into_independent_epochs() {
  let cs = Ema::new(0.5);
  let cg = Hysteresis::new(0.6, 0.3);
  let opts = SegmentOptions::new().with_min_len(1).with_merge_gap(2);
  // Two epochs; the second restarts its span positions after the break.
  let a = windows(&[0.9, 0.9, 0.1, 0.9], 1, 1, 1);
  let b = windows(&[0.9, 0.1, 0.9, 0.9, 0.9], 1, 1, 1);

  // Combined: epoch A, a declared discontinuity, epoch B, then finish.
  let mut dec: Decoder<_, _, f32> = Decoder::new(cs.smoother(), cg.gate(), opts);
  let mut flags = Vec::new();
  let mut ranges = Vec::new();
  for w in &a {
    let step = dec.push(*w).unwrap();
    flags.push(step.active());
    ranges.extend(step.finalized());
  }
  ranges.extend(dec.discontinuity());
  for w in &b {
    let step = dec.push(*w).unwrap();
    flags.push(step.active());
    ranges.extend(step.finalized());
  }
  ranges.extend(dec.finish());

  // Two fresh whole-epoch runs, concatenated: the property holds only if
  // `discontinuity` fully re-armed every stage (the segmenter timeline, the EMA
  // seed, and the hysteresis latch).
  let (fa_flags, fa_ranges) = fresh_run(&cs, &cg, opts, &a);
  let (fb_flags, fb_ranges) = fresh_run(&cs, &cg, opts, &b);
  let want_flags: Vec<bool> = fa_flags.into_iter().chain(fb_flags).collect();
  let want_ranges: Vec<Range> = fa_ranges.into_iter().chain(fb_ranges).collect();

  assert_eq!(flags, want_flags, "causal plane across the epoch break");
  assert_eq!(
    ranges, want_ranges,
    "finalized plane across the epoch break"
  );
}

// ── F3: explicit discontinuity forwarding ────────────────────────────────────

/// A test-local pass-through [`Smoother<f32>`] recording whether
/// [`reset`](Smoother::reset) or [`discontinuity`](Smoother::discontinuity)
/// fired — the smoother half of the F3 forwarding probe. The counters live
/// behind shared `Rc<Cell<_>>` handles the test keeps after the probe is moved
/// into the decoder (or into a `Box<dyn Smoother<f32>>`).
struct ProbeSmoother {
  reset_calls: Rc<Cell<usize>>,
  discontinuity_calls: Rc<Cell<usize>>,
}

impl ProbeSmoother {
  fn new() -> (Self, Rc<Cell<usize>>, Rc<Cell<usize>>) {
    let reset_calls = Rc::new(Cell::new(0));
    let discontinuity_calls = Rc::new(Cell::new(0));
    (
      Self {
        reset_calls: reset_calls.clone(),
        discontinuity_calls: discontinuity_calls.clone(),
      },
      reset_calls,
      discontinuity_calls,
    )
  }
}

impl Smoother<f32> for ProbeSmoother {
  fn push(&mut self, w: Windowed<f32>) -> Result<Windowed<f32>, WinditError> {
    Ok(w)
  }

  fn reset(&mut self) {
    self.reset_calls.set(self.reset_calls.get() + 1);
  }

  fn discontinuity(&mut self) {
    self
      .discontinuity_calls
      .set(self.discontinuity_calls.get() + 1);
  }
}

/// The gate half of the F3 probe: a [`Gate<f32>`] emitting a fixed decision and
/// recording `reset`/`discontinuity` the same way.
struct ProbeGate {
  active: bool,
  reset_calls: Rc<Cell<usize>>,
  discontinuity_calls: Rc<Cell<usize>>,
}

impl ProbeGate {
  fn new(active: bool) -> (Self, Rc<Cell<usize>>, Rc<Cell<usize>>) {
    let reset_calls = Rc::new(Cell::new(0));
    let discontinuity_calls = Rc::new(Cell::new(0));
    (
      Self {
        active,
        reset_calls: reset_calls.clone(),
        discontinuity_calls: discontinuity_calls.clone(),
      },
      reset_calls,
      discontinuity_calls,
    )
  }
}

impl Gate<f32> for ProbeGate {
  fn push(&mut self, _w: &Windowed<f32>) -> Result<bool, WinditError> {
    Ok(self.active)
  }

  fn reset(&mut self) {
    self.reset_calls.set(self.reset_calls.get() + 1);
  }

  fn discontinuity(&mut self) {
    self
      .discontinuity_calls
      .set(self.discontinuity_calls.get() + 1);
  }
}

#[test]
fn discontinuity_forwards_to_every_stage() {
  let (sm, sm_reset, sm_disc) = ProbeSmoother::new();
  let (ga, ga_reset, ga_disc) = ProbeGate::new(true); // always active
  let mut dec = Decoder::new(sm, ga, SegmentOptions::new());

  // Two active unit windows open a run in the segmenter.
  let _ = dec.push(Windowed::new(0.0, Span::new(0, 1, 1))).unwrap();
  let _ = dec.push(Windowed::new(0.0, Span::new(1, 1, 1))).unwrap();

  // `discontinuity` drains the open run AND forwards `discontinuity` (not
  // `reset`) to both stages — the F3 regression probe.
  let tail: Vec<Range> = dec.discontinuity().collect();
  assert_eq!(
    tail,
    [Range::new(0, 2)],
    "the open run finalizes on the break"
  );
  assert_eq!(sm_disc.get(), 1, "smoother received discontinuity");
  assert_eq!(ga_disc.get(), 1, "gate received discontinuity");
  assert_eq!(
    sm_reset.get(),
    0,
    "smoother reset must not fire on discontinuity"
  );
  assert_eq!(
    ga_reset.get(),
    0,
    "gate reset must not fire on discontinuity"
  );

  // `reset`, by contrast, forwards `reset` to both stages and leaves the
  // discontinuity counts untouched.
  dec.reset();
  assert_eq!(sm_reset.get(), 1, "smoother received reset");
  assert_eq!(ga_reset.get(), 1, "gate received reset");
  assert_eq!(
    sm_disc.get(),
    1,
    "smoother discontinuity count unchanged by reset"
  );
  assert_eq!(
    ga_disc.get(),
    1,
    "gate discontinuity count unchanged by reset"
  );
}

#[test]
fn discontinuity_forwards_through_boxed_stages() {
  // The two-hop path Decoder -> Box -> concrete stage: the boxed stage's
  // `discontinuity` must reach the probe, not be flattened to its `reset`.
  let (sm, _sm_reset, sm_disc) = ProbeSmoother::new();
  let (ga, _ga_reset, ga_disc) = ProbeGate::new(false);
  let boxed_sm: Box<dyn Smoother<f32>> = Box::new(sm);
  let boxed_ga: Box<dyn Gate<f32>> = Box::new(ga);
  let mut dec: Decoder<Box<dyn Smoother<f32>>, Box<dyn Gate<f32>>, f32> =
    Decoder::new(boxed_sm, boxed_ga, SegmentOptions::new());

  let _ = dec.push(Windowed::new(0.0, Span::new(0, 1, 1))).unwrap();
  let _ = dec.discontinuity();
  assert_eq!(sm_disc.get(), 1, "discontinuity reached the boxed smoother");
  assert_eq!(ga_disc.get(), 1, "discontinuity reached the boxed gate");
}

// ── The dyn / manifest path ──────────────────────────────────────────────────

#[test]
fn dyn_manifest_path_matches_concrete() {
  let seq = windows(&[0.1, 0.8, 0.9, 0.2, 0.7, 0.7, 0.1], 1, 1, 1);
  let opts = SegmentOptions::new().with_min_len(2);

  // The run-time-selected manifest path: dyn stages held *inside* the decoder,
  // instantiable only because `Box<dyn Smoother<f32>>` and `Box<dyn Gate<f32>>`
  // themselves implement the stage traits (the forwarding impls).
  let sm: Box<dyn Smoother<f32>> = Box::new(Ema::new(0.5).smoother());
  let ga: Box<dyn Gate<f32>> = Box::new(Threshold::new(0.5).gate());
  let mut dyn_dec: Decoder<Box<dyn Smoother<f32>>, Box<dyn Gate<f32>>, f32> =
    Decoder::new(sm, ga, opts);

  let mut dyn_flags = Vec::new();
  let mut dyn_ranges = Vec::new();
  for w in &seq {
    let step = dyn_dec.push(*w).unwrap();
    dyn_flags.push(step.active());
    dyn_ranges.extend(step.finalized());
  }
  dyn_ranges.extend(dyn_dec.finish());

  // Identical to the concrete decoder over the same policies...
  let (want_flags, want_ranges) = fresh_run(&Ema::new(0.5), &Threshold::new(0.5), opts, &seq);
  assert_eq!(dyn_flags, want_flags);
  assert_eq!(dyn_ranges, want_ranges);

  // ...and to the batch composition (full manifest-path parity, both planes).
  let smoothed = Ema::new(0.5).smooth(&seq).unwrap();
  let batch_ranges = Threshold::new(0.5).segment(&opts, &smoothed).unwrap();
  assert_eq!(dyn_ranges, batch_ranges);
}

// ── Bound placement, Step, and error recovery ────────────────────────────────

#[test]
fn new_opts_and_finish_are_bound_free() {
  // `()` implements neither `Smoother<()>` nor `Gate<()>`. That this compiles is
  // the proof `new`/`opts`/`finish` carry no stage bounds — only `push`,
  // `discontinuity`, and `reset` do.
  let opts = SegmentOptions::new().with_min_len(3).with_merge_gap(2);
  let dec: Decoder<(), (), ()> = Decoder::new((), (), opts);
  assert_eq!(dec.opts(), opts);
  // A fresh segmenter has nothing pending, so the drained tail is empty.
  assert_eq!(dec.finish().count(), 0);
}

#[test]
fn clone_and_debug_do_not_require_v_bounds() {
  // A marker `V` that is deliberately neither `Clone` nor `Debug`. The manual
  // impls bound `S`/`G` alone, so the decoder stays `Clone`/`Debug`; a derive
  // would have demanded `V: Clone`/`V: Debug` through the phantom and failed to
  // compile here.
  #[allow(dead_code)]
  struct NotCloneNotDebug;
  let dec: Decoder<EmaState, ThresholdState, NotCloneNotDebug> = Decoder::new(
    Ema::new(0.5).smoother(),
    Threshold::new(0.5).gate(),
    SegmentOptions::new(),
  );
  let cloned = dec.clone();
  let _debug: &dyn core::fmt::Debug = &cloned;
}

#[test]
fn push_error_is_non_transactional_and_recoverable() {
  let mut dec: Decoder<_, _, f32> = Decoder::new(
    Ema::new(0.5).smoother(),
    Threshold::new(0.5).gate(),
    SegmentOptions::new(),
  );
  // Establish a timeline at start 5.
  let _ = dec.push(Windowed::new(0.9, Span::new(5, 1, 1))).unwrap();
  // A backward span: only the segmenter reads spans here, so exactly one error
  // surfaces, from it — even though the value-only smoother and gate already
  // advanced on this same push (the non-transactional contract).
  let err = dec
    .push(Windowed::new(0.9, Span::new(3, 1, 1)))
    .unwrap_err();
  assert_eq!(
    err,
    WinditError::NonMonotonicSpan {
      prev_start: 5,
      start: 3
    }
  );

  // Recovery via discontinuity re-arms every stage; the next epoch may restart
  // its span positions, so a start of 0 is now in order.
  let _ = dec.discontinuity();
  let step = dec.push(Windowed::new(0.9, Span::new(0, 1, 1))).unwrap();
  assert!(step.active());

  // reset is the other recovery path and likewise clears the timeline.
  dec.reset();
  let _ = dec.push(Windowed::new(0.9, Span::new(0, 1, 1))).unwrap();
}

#[test]
fn step_accessors_and_derives() {
  let mut dec: Decoder<_, _, f32> = Decoder::new(
    Ema::new(1.0).smoother(), // pass-through: smoothed == score
    Threshold::new(0.5).gate(),
    SegmentOptions::new(),
  );

  // An inactive first window opens no run and finalizes nothing.
  let inactive = dec.push(Windowed::new(0.2, Span::new(0, 1, 1))).unwrap();
  assert!(!inactive.active());
  assert_eq!(inactive.finalized(), None);

  // The derived value semantics: Copy (the original stays usable after the
  // bind), PartialEq and Debug (through assert_eq), and Hash (equal Steps hash
  // equally). Clone and Eq come with the same derive.
  let copied = inactive;
  assert_eq!(inactive, copied);
  assert_eq!(hash_of(&inactive), hash_of(&copied));

  // An active window opens a run; the next inactive push closes it into the
  // fold; a later push whose start clears the (zero) gap horizon finalizes it.
  let active = dec.push(Windowed::new(0.9, Span::new(1, 1, 1))).unwrap();
  assert!(active.active());
  assert_eq!(active.finalized(), None);
  let _ = dec.push(Windowed::new(0.2, Span::new(2, 1, 1))).unwrap();
  let closing = dec.push(Windowed::new(0.2, Span::new(3, 1, 1))).unwrap();
  assert!(!closing.active());
  assert_eq!(closing.finalized(), Some(Range::new(1, 2)));
}
