//! `HysteresisSegment::segment` does not allocate the intermediate gated
//! sequence: the fused single pass carries one bool of latch state instead of
//! building a `Vec<Windowed<f32>>` the length of the input.
//!
//! Proving that needs an allocator that refuses, which is a process-wide
//! setting, so this suite is one test in its own binary: nothing else runs while
//! the refusal is armed. The refusal is size-gated above the fused path's own
//! small allocations (the run vectors) and below the length-of-input gated
//! vector it must never build, so the failure that would be observed is exactly
//! the reintroduced intermediate and nothing else.
//!
//! The failure mode differs from the other allocation suites. `chunk` and
//! `aggregate` grow on a `Result`-returning path, so a refused allocation comes
//! back as `WinditError::AllocFailed`. `SegmentPolicy::segment` is infallible: a
//! reintroduced full-length intermediate allocation has no error channel to
//! report on, so a refusal reaches `handle_alloc_error` and aborts the test
//! binary. That abort *is* the regression signal — a fused path that stays fused
//! simply never asks for the gated vector, so the armed run returns the ordinary
//! answer instead.
//!
//! Gated on `alloc`: without it there is no segmentation to drive, and the file
//! compiles to an empty test binary so the rest of the feature matrix still
//! builds.
#![cfg(any(feature = "std", feature = "alloc"))]

use std::{
  alloc::{GlobalAlloc, Layout, System},
  sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use windit::{
  plan::Span,
  segment::{HysteresisSegment, Range, SegmentPolicy},
  windowed::Windowed,
};

/// Refuse allocations of at least `LIMIT` bytes while `ARMED`; defer everything
/// else to the system allocator.
struct Refusing;

/// Whether the refusal is in effect.
static ARMED: AtomicBool = AtomicBool::new(false);

/// The size at or above which an armed allocation is refused. Set above the
/// fused path's own small run vectors and below the length-of-input gated vector
/// the fused path must never build, so the refusal lands on a reintroduced
/// intermediate and on nothing else.
static LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);

// SAFETY: every branch forwards to `System`, a correct allocator, or returns
// null — the documented way for `alloc` to report failure. No branch returns a
// pointer `System` did not hand out, and `dealloc` / `realloc` always forward,
// so nothing is freed by an allocator that did not allocate it.
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

/// The refusal threshold, in bytes: above the fused path's own allocations (four
/// runs of 16-byte `Range`s, well under a kilobyte) and far below the gated
/// `Vec<Windowed<f32>>` the two-pass path used to build (`4096 * 32` bytes).
const GATE_LIMIT: usize = 4096;

#[test]
fn fused_hysteresis_segment_does_not_allocate_the_gated_sequence() {
  // 4096 unit-span windows in eight 512-element blocks alternating 0.9 / 0.1 and
  // starting active. With on 0.6 / off 0.3 the four active blocks segment to four
  // 512-element runs; the inactive blocks separate them (default opts do not
  // merge across the gaps).
  const N: usize = 4096;
  const BLOCK: usize = 512;
  let input: Vec<Windowed<f32>> = (0..N)
    .map(|i| {
      let active = (i / BLOCK).is_multiple_of(2);
      Windowed::new(if active { 0.9 } else { 0.1 }, Span::new(i, 1, 1))
    })
    .collect();
  let policy = HysteresisSegment::new(0.6, 0.3);
  let expected = vec![
    Range::new(0, 512),
    Range::new(1024, 1536),
    Range::new(2048, 2560),
    Range::new(3072, 3584),
  ];

  // Unarmed, the ordinary answer — which is what makes the armed run below
  // evidence about allocation rather than about geometry.
  let unarmed = policy.segment(&input);
  assert_eq!(unarmed, expected);

  // The gate is honest only if the intermediate it must not build is above the
  // refusal threshold: the length-of-input gated vector would be one allocation
  // of `N * size_of::<Windowed<f32>>()` bytes, comfortably over `GATE_LIMIT`.
  assert!(
    input.len() * core::mem::size_of::<Windowed<f32>>() >= GATE_LIMIT,
    "the gated vector must exceed the refusal threshold for this test to bite"
  );

  LIMIT.store(GATE_LIMIT, Ordering::Relaxed);
  ARMED.store(true, Ordering::Relaxed);
  let armed = policy.segment(&input);
  ARMED.store(false, Ordering::Relaxed);

  // Reaching here at all means the fused path never asked for the gated vector: a
  // refusal would have aborted this binary through `handle_alloc_error`. The
  // equal result confirms the armed run computed the same segmentation.
  assert_eq!(armed, expected);
  assert_eq!(armed, unarmed);
}
