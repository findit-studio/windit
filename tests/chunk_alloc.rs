//! `ContentAware::chunk` reports an allocator that refuses, rather than aborting.
//!
//! The chunk count is not known before packing runs, so the output cannot be
//! reserved up front the way `WindowPlan::spans` reserves its plan; it grows as
//! chunks are proven. That growth is on a `Result`-returning path, so an
//! allocation it cannot get must come back as `WinditError::AllocFailed` — an
//! abort would break the same contract `try_slice_pad_mask` exists to keep.
//!
//! Proving that needs an allocator that refuses, and the allocator is
//! process-wide: the harness's own threads allocate while the test runs. So the
//! refusal is armed on the test's own thread alone, and it is size-gated, so the
//! test's own small allocations still succeed and the failure that is observed
//! is the one the chunker asked for.
//!
//! Gated on `text`: without it there is no chunker, and the file compiles to an
//! empty test binary so the rest of the feature matrix still builds.
#![cfg(feature = "text")]
// Coverage instrumentation allocates inside the guarded window; the sanitizer
// and plain lanes already cover this guard, so it steps aside under tarpaulin.
#![cfg(not(tarpaulin))]

use std::{
  alloc::{GlobalAlloc, Layout, System},
  cell::Cell,
  sync::atomic::{AtomicUsize, Ordering},
};

use windit::{
  plan::WindowOptions,
  split::{ContentAware, MeasureText},
  WinditError,
};

/// Refuse allocations of at least `limit` bytes while armed; defer everything
/// else to the system allocator.
struct Refusing;

std::thread_local! {
  /// Whether the refusal is in effect on this thread. Only the test's own
  /// thread arms it, so the harness's threads, which allocate while the test
  /// runs, are never refused. A `const`-initialized `Cell` has no destructor,
  /// so it is read without allocating, which is what lets the allocator read
  /// it.
  static ARMED: Cell<bool> = const { Cell::new(false) };
}

/// The size at or above which an armed allocation is refused. Above the
/// chunker's own fixed-size scratch (a descent stack of at most one frame per
/// boundary level, and an atom buffer holding one chunk's atoms) so that the
/// refusal lands on the growing output.
static LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);

// SAFETY: every branch forwards to `System`, which is a correct allocator, or
// returns null — the documented way for `alloc` to report failure. No branch
// returns a pointer `System` did not hand out, and `dealloc` / `realloc` always
// forward, so nothing is freed by an allocator that did not allocate it.
unsafe impl GlobalAlloc for Refusing {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    if ARMED.get() && layout.size() >= LIMIT.load(Ordering::Relaxed) {
      return core::ptr::null_mut();
    }
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    if ARMED.get() && new_size >= LIMIT.load(Ordering::Relaxed) {
      return core::ptr::null_mut();
    }
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static ALLOC: Refusing = Refusing;

#[test]
fn a_refused_output_growth_is_a_typed_error() {
  // 4000 one-token words at window 4 pack into 1000 chunks, so the output
  // outgrows any small reservation many times over. Everything the run needs
  // besides that output is allocated before the refusal is armed.
  let mut text = String::new();
  for i in 0..4000 {
    if i > 0 {
      text.push(' ');
    }
    text.push('w');
  }
  let count = |s: &str| s.split_whitespace().count();
  let len_fn: &dyn MeasureText = &count;
  let opts = WindowOptions::new(4);
  let chunker = ContentAware::new(len_fn);

  // Unarmed, this is the ordinary answer -- which is also what makes the armed
  // run below evidence about allocation rather than about geometry.
  assert_eq!(chunker.chunk(&text, &opts).unwrap().len(), 1000);

  LIMIT.store(4096, Ordering::Relaxed);
  ARMED.set(true);
  let refused = chunker.chunk(&text, &opts);
  ARMED.set(false);

  assert!(
    matches!(refused, Err(WinditError::AllocFailed { .. })),
    "a refused allocation must be reported, not aborted on; got {refused:?}"
  );
}
