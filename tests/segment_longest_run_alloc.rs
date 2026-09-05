//! [`longest_run`](windit::segment::longest_run) keeps one `Range`, not every
//! one: its answer is a fold over the finalized ranges, so it allocates nothing
//! however many runs the input produces.
//!
//! The property under test is *space in the output-run count*, which is why the
//! harness counts allocator traffic rather than wall-clock time — a machine with
//! anything else running cannot resolve the difference, and a count can. The
//! input is a million unit spans alternating accepted/rejected, so a
//! materializing definition has about 500,000 one-element ranges to store
//! (roughly 8 MiB of `Vec<Range>` growth) while the fold has one `Option<Range>`.
//!
//! A counting global allocator is a process-wide setting, so this suite is one
//! test in its own binary: nothing else runs while the count is armed. The
//! counter is armed around the call under test alone — the input is built
//! before it — so every call it records belongs to `longest_run`.
//!
//! Gated on `alloc`: `longest_run` is an `alloc`-tier convenience, and the file
//! compiles to an empty test binary without it so the rest of the feature matrix
//! still builds.
#![cfg(any(feature = "std", feature = "alloc"))]
// Coverage instrumentation allocates inside the guarded window; the sanitizer
// and plain lanes already cover this guard, so it steps aside under tarpaulin.
#![cfg(not(tarpaulin))]

use std::{
  alloc::{GlobalAlloc, Layout, System},
  sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use windit::{
  plan::Span,
  segment::{longest_run, runs, Range, SegmentOptions},
  windowed::Windowed,
};

/// Count allocations while `ARMED`; always forward to the system allocator.
///
/// Counting rather than refusing: a refusal makes the failure an opaque
/// `AllocFailed` (the checked drivers ask with `try_reserve`), where a count
/// reports how many allocations the call actually made, which is the quantity
/// the complexity claim is about.
struct Counting;

/// Whether the count is in effect.
static ARMED: AtomicBool = AtomicBool::new(false);

/// Allocation calls made while armed.
static CALLS: AtomicUsize = AtomicUsize::new(0);

/// Bytes requested while armed.
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to `System`, a correct allocator, and adds only
// relaxed counter arithmetic beside it. No pointer is invented, and `dealloc` /
// `realloc` always forward, so nothing is freed by an allocator that did not
// allocate it.
unsafe impl GlobalAlloc for Counting {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    if ARMED.load(Ordering::Relaxed) {
      CALLS.fetch_add(1, Ordering::Relaxed);
      BYTES.fetch_add(layout.size(), Ordering::Relaxed);
    }
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    if ARMED.load(Ordering::Relaxed) {
      CALLS.fetch_add(1, Ordering::Relaxed);
      BYTES.fetch_add(new_size, Ordering::Relaxed);
    }
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Run `f` with the allocation counter armed, returning its value alongside
/// `(calls, bytes)`.
fn counted<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
  CALLS.store(0, Ordering::Relaxed);
  BYTES.store(0, Ordering::Relaxed);
  ARMED.store(true, Ordering::Relaxed);
  let out = f();
  ARMED.store(false, Ordering::Relaxed);
  (
    out,
    CALLS.load(Ordering::Relaxed),
    BYTES.load(Ordering::Relaxed),
  )
}

#[test]
fn longest_run_allocates_nothing_however_many_runs() {
  // One million unit spans alternating accepted / rejected: ~500,000 finalized
  // one-element ranges, the shape that separates a fold from a materialization.
  const N: usize = 1_000_000;
  let input: Vec<Windowed<f32>> = (0..N)
    .map(|i| {
      Windowed::new(
        if i.is_multiple_of(2) { 0.9 } else { 0.1 },
        Span::new(i, 1, 1),
      )
    })
    .collect();
  let opts = SegmentOptions::new();

  // The materializing definition, computed while the counter is idle: this is
  // the answer `longest_run` must keep returning, so the space fix cannot be
  // mistaken for a semantic change.
  let expect = runs(&input, |&v| v >= 0.5, &opts)
    .unwrap()
    .into_iter()
    .fold(None::<Range>, |best, r| match best {
      Some(b) if b.len() >= r.len() => Some(b),
      _ => Some(r),
    });
  assert_eq!(expect, Some(Range::new(0, 1)));

  let (got, calls, bytes) = counted(|| longest_run(&input, |&v| v >= 0.5, &opts));

  assert_eq!(got.unwrap(), expect, "the fold must not change the answer");
  assert_eq!(
    (calls, bytes),
    (0, 0),
    "longest_run kept output ranges it does not need: {calls} allocation call(s), {bytes} byte(s)"
  );
}
