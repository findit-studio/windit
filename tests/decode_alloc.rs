//! The streaming [`Decoder`](windit::decode::Decoder) allocates nothing: each
//! `push` advances three fixed-size stages (a smoother, a gate, a `Segmenter`) and
//! `finish` drains a fixed-size [`SegmentTail`](windit::segment::SegmentTail).
//! Driving a long input through the heaviest shipped composition — a `CadenceEma`
//! smoother feeding a `Hangover(Dwell(Vote))` gate, every P3 state machine active —
//! touches the heap zero times.
//!
//! Proving that needs an allocator that refuses, which is a process-wide setting,
//! so the whole file is a single test in its own binary: nothing else runs while
//! the refusal is armed. The refusal is armed at a one-byte threshold around the
//! streaming drives alone, so *any* heap allocation on that path is caught.
//!
//! The failure mode: the streaming path has no error channel for an allocation (it
//! never asks for one), so a refused allocation reaches `handle_alloc_error` and
//! aborts the test binary. That abort *is* the regression signal — a path that
//! stays allocation-free simply never asks, so the armed drive returns the ordinary
//! answer instead.
//!
//! The same test carries the `size_of` / `align_of` pins that back the O(1) state
//! table. They allocate nothing and run before the refusal is armed. `VoteState`'s
//! sixteen bytes are the hard one: the `of <= 64` ring bound is what stops the vote
//! history from growing the state, and it holds on every target width.
//!
//! Gated on `alloc`: without it there is no batch `segment` to cross-check against,
//! and the file compiles to an empty test binary so the rest of the feature matrix
//! still builds. (The `Decoder` itself is featureless; this suite gates only
//! because it compares against the `alloc`-tier batch composition.)
#![cfg(any(feature = "std", feature = "alloc"))]

use core::mem::{align_of, size_of};
use std::{
  alloc::{GlobalAlloc, Layout, System},
  sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use windit::{
  decode::{Decoder, Step},
  plan::Span,
  segment::{
    Dwell, DwellState, GatePolicy, Hangover, HangoverState, Range, SegmentOptions, ThresholdState,
    Vote, VoteState,
  },
  smooth::{CadenceEma, CadenceEmaState, SmoothPolicy},
  windowed::Windowed,
};

/// Refuse allocations of at least `LIMIT` bytes while `ARMED`; defer everything
/// else to the system allocator.
struct Refusing;

/// Whether the refusal is in effect.
static ARMED: AtomicBool = AtomicBool::new(false);

/// The size at or above which an armed allocation is refused. Left at `MAX`
/// (refuse nothing) except around the streaming drives, where it drops to one byte
/// so any heap allocation is caught.
static LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);

// SAFETY: every branch forwards to `System`, a correct allocator, or returns null —
// the documented way for `alloc` to report failure. No branch returns a pointer
// `System` did not hand out, and `dealloc` / `realloc` always forward, so nothing
// is freed by an allocator that did not allocate it.
unsafe impl GlobalAlloc for Refusing {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    if ARMED.load(Ordering::Relaxed) && layout.size() >= LIMIT.load(Ordering::Relaxed) {
      return core::ptr::null_mut();
    }
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    if ARMED.load(Ordering::Relaxed) && new_size >= LIMIT.load(Ordering::Relaxed) {
      return core::ptr::null_mut();
    }
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static ALLOC: Refusing = Refusing;

/// Fold a finalized range into a checksum, so the armed drive can be verified to
/// compute the same segmentation as the batch composition without collecting into
/// a heap `Vec` under the armed allocator.
fn mix(acc: u64, r: Range) -> u64 {
  acc ^ (r.start() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (r.end() as u64).rotate_left(32)
}

/// The heaviest shipped smoother, denominated in elements.
fn smoother_config() -> CadenceEma {
  CadenceEma::new(4.0)
}

/// The canonical gate nesting — onset confirmation inside release hold, over an
/// N-of-M vote — so a single drive exercises `VoteState`, `DwellState`, and
/// `HangoverState` at once.
fn gate_config() -> Hangover<Dwell<Vote>> {
  Hangover::new(Dwell::new(Vote::new(2, 3, 0.5), 3), 5)
}

/// Drive a fresh `Decoder` over the input to completion, checksumming every
/// finalized range. Constructs its stages and threads the pipeline with no heap
/// allocation, so it is safe to call with the refusal armed.
fn drive_decoder_checksum(input: &[Windowed<f32>], opts: SegmentOptions) -> u64 {
  let mut dec = Decoder::new(smoother_config().smoother(), gate_config().gate(), opts);
  let mut acc = 0u64;
  for w in input {
    if let Some(r) = dec.push(*w).unwrap().finalized() {
      acc = mix(acc, r);
    }
  }
  for r in dec.finish() {
    acc = mix(acc, r);
  }
  acc
}

/// The batch composition of the same policies as a checksum — smooth, then segment
/// the smoothed sequence. Allocates (returns `Vec`s), so it must be computed while
/// the heap is free; it is the parity the alloc pin backs up.
fn batch_checksum(input: &[Windowed<f32>], opts: SegmentOptions) -> u64 {
  let smoothed = smoother_config().smooth(input).unwrap();
  gate_config()
    .segment(&opts, &smoothed)
    .unwrap()
    .into_iter()
    .fold(0u64, mix)
}

#[test]
fn decoder_drive_does_not_allocate_and_state_sizes_are_pinned() {
  // ── Size pins (drift detection backing the O(1) state table) ──
  //
  // Run first, before the refusal is armed; `size_of`/`align_of` are const and
  // allocate nothing. `VoteState` is sixteen bytes on *every* target: the
  // `of <= 64` ring bound keeps it from growing with the vote history, so this pin
  // is not gated on pointer width.
  assert_eq!(size_of::<VoteState>(), 16);
  assert_eq!(align_of::<VoteState>(), 8);

  // The usize-carrying states are pinned on 64-bit, where `usize` is eight bytes.
  #[cfg(target_pointer_width = "64")]
  {
    assert_eq!(size_of::<CadenceEmaState>(), 32);
    assert_eq!(align_of::<CadenceEmaState>(), 8);
    assert_eq!(size_of::<DwellState<ThresholdState>>(), 48);
    assert_eq!(size_of::<HangoverState<ThresholdState>>(), 48);
    assert_eq!(size_of::<Step>(), 32);
    assert_eq!(align_of::<Step>(), 8);
    // The whole pipeline the drive below runs: CadenceEma smoother, the canonical
    // gate nesting, and the segmenter, all held inline.
    assert_eq!(
      size_of::<Decoder<CadenceEmaState, HangoverState<DwellState<VoteState>>, f32>>(),
      208
    );
  }

  // ── Allocation pin: the decoder drive touches the heap zero times ──
  //
  // 4096 unit-span windows in eight 512-element blocks alternating 0.9 / 0.1 and
  // starting active — many run boundaries, so every stage transitions repeatedly.
  const N: usize = 4096;
  const BLOCK: usize = 512;
  let input: Vec<Windowed<f32>> = (0..N)
    .map(|i| {
      let active = (i / BLOCK).is_multiple_of(2);
      Windowed::new(if active { 0.9 } else { 0.1 }, Span::new(i, 1, 1))
    })
    .collect();

  // Two morphologies: no merging (separate runs) and an unbounded merge_gap (one
  // ever-widening pending accumulator, the O(1) case).
  let separate = SegmentOptions::new();
  let unbounded = SegmentOptions::new().with_merge_gap(usize::MAX);

  // The batch composition's answer, as a checksum, computed while the heap is free.
  // This ties the streaming decoder to the batch smooth+segment — the parity the
  // alloc pin backs up.
  let expect_separate = batch_checksum(&input, separate);
  let expect_unbounded = batch_checksum(&input, unbounded);

  // Unarmed reference — makes the armed run evidence about allocation, not
  // geometry.
  assert_eq!(drive_decoder_checksum(&input, separate), expect_separate);
  assert_eq!(drive_decoder_checksum(&input, unbounded), expect_unbounded);

  // Arm the refusal at one byte around the streaming drives only. Reaching the
  // assertions at all means neither drive asked the heap for anything: a refusal
  // would have aborted this binary through `handle_alloc_error`.
  LIMIT.store(1, Ordering::Relaxed);
  ARMED.store(true, Ordering::Relaxed);
  let armed_separate = drive_decoder_checksum(&input, separate);
  let armed_unbounded = drive_decoder_checksum(&input, unbounded);
  ARMED.store(false, Ordering::Relaxed);

  assert_eq!(armed_separate, expect_separate);
  assert_eq!(armed_unbounded, expect_unbounded);
}
