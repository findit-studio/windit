//! The incremental [`Segmenter`](windit::segment::Segmenter) allocates nothing:
//! `push` advances four machine words of state, and `finish` returns a
//! fixed-size [`SegmentTail`](windit::segment::SegmentTail) that holds its ranges
//! inline. Driving a long input to completion — including the unbounded
//! `merge_gap` case, where the pending accumulator only ever widens — touches
//! the heap zero times.
//!
//! Proving that needs an allocator that refuses, which is a process-wide
//! setting, so this suite is one test in its own binary: nothing else runs while
//! the refusal is armed. The refusal is armed around the streaming drive alone,
//! at a one-byte threshold, so *any* heap allocation on that path is caught.
//!
//! The failure mode: the streaming path has no error channel for an allocation
//! (it never asks for one), so a refused allocation reaches `handle_alloc_error`
//! and aborts the test binary. That abort *is* the regression signal — a path
//! that stays allocation-free simply never asks, so the armed drive returns the
//! ordinary answer instead.
//!
//! Gated on `alloc`: without it there is no batch `runs` to cross-check against,
//! and the file compiles to an empty test binary so the rest of the feature
//! matrix still builds. (The `Segmenter` itself is featureless; this suite gates
//! only because it compares against the `alloc`-tier `runs`.)
#![cfg(any(feature = "std", feature = "alloc"))]

use std::{
  alloc::{GlobalAlloc, Layout, System},
  sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use windit::{
  plan::Span,
  segment::{runs, Range, SegmentOptions, Segmenter},
  windowed::Windowed,
};

/// Refuse allocations of at least `LIMIT` bytes while `ARMED`; defer everything
/// else to the system allocator.
struct Refusing;

/// Whether the refusal is in effect.
static ARMED: AtomicBool = AtomicBool::new(false);

/// The size at or above which an armed allocation is refused. Left at `MAX`
/// (refuse nothing) except around the streaming drive, where it drops to one
/// byte so any heap allocation is caught.
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

/// Fold a finalized range into a checksum, so the armed drive can be verified to
/// compute the same segmentation as the batch driver without collecting into a
/// heap `Vec` under the armed allocator.
fn mix(acc: u64, r: Range) -> u64 {
  acc ^ (r.start() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (r.end() as u64).rotate_left(32)
}

/// Drive a fresh `Segmenter` over the gated input to completion, checksumming
/// every finalized range. Allocates nothing itself, so it is safe to call with
/// the refusal armed.
fn drive_checksum(input: &[Windowed<f32>], opts: SegmentOptions) -> u64 {
  let mut seg = Segmenter::new(opts);
  let mut acc = 0u64;
  for w in input {
    if let Some(r) = seg.push(*w.value() >= 0.5, w.span()).unwrap() {
      acc = mix(acc, r);
    }
  }
  for r in seg.finish() {
    acc = mix(acc, r);
  }
  acc
}

#[test]
fn segmenter_push_and_finish_do_not_allocate() {
  // 4096 unit-span windows in eight 512-element blocks alternating 0.9 / 0.1 and
  // starting active — many run boundaries to exercise every transition.
  const N: usize = 4096;
  const BLOCK: usize = 512;
  let input: Vec<Windowed<f32>> = (0..N)
    .map(|i| {
      let active = (i / BLOCK).is_multiple_of(2);
      Windowed::new(if active { 0.9 } else { 0.1 }, Span::new(i, 1, 1))
    })
    .collect();

  // Two configurations: no merging (four separate runs) and an unbounded
  // merge_gap (one ever-widening pending accumulator, the O(1) case).
  let separate = SegmentOptions::new();
  let unbounded = SegmentOptions::new().with_merge_gap(usize::MAX);

  // The batch driver's answer, as a checksum, computed while the heap is free.
  // This ties the streaming drive to `runs` — the parity the alloc pin backs up.
  let expect_separate = runs(&input, |&v| v >= 0.5, &separate)
    .unwrap()
    .into_iter()
    .fold(0u64, mix);
  let expect_unbounded = runs(&input, |&v| v >= 0.5, &unbounded)
    .unwrap()
    .into_iter()
    .fold(0u64, mix);

  // Unarmed reference — makes the armed run evidence about allocation, not
  // geometry.
  assert_eq!(drive_checksum(&input, separate), expect_separate);
  assert_eq!(drive_checksum(&input, unbounded), expect_unbounded);

  // Arm the refusal at one byte around the streaming drives only. Reaching the
  // assertions at all means neither drive asked the heap for anything: a refusal
  // would have aborted this binary through `handle_alloc_error`.
  LIMIT.store(1, Ordering::Relaxed);
  ARMED.store(true, Ordering::Relaxed);
  let armed_separate = drive_checksum(&input, separate);
  let armed_unbounded = drive_checksum(&input, unbounded);
  ARMED.store(false, Ordering::Relaxed);

  assert_eq!(armed_separate, expect_separate);
  assert_eq!(armed_unbounded, expect_unbounded);
}
