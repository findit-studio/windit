//! `SmoothPolicy::smooth` reports an allocator that refuses, rather than
//! aborting.
//!
//! The batch convenience reserves its whole output up front — one window out per
//! window in, a length the *caller* chooses and which need not correspond to
//! memory that exists — on a `Result`-returning path. An allocation it cannot get
//! must therefore come back as `WinditError::AllocFailed`, carrying the element
//! count it asked for, which is the same contract `try_slice_pad_mask`,
//! `ContentAware::chunk`, and every aggregation buffer keep; an abort would break
//! it.
//!
//! The output vector is the *only* thing this path allocates, which is what makes
//! the case exact: `Ema` smooths `f32` in O(1) registers, so a refusal here can
//! come from nowhere else and the `elements` it reports can be pinned to the
//! input length. A `smooth` that reached for an infallible `Vec::with_capacity`,
//! or that dropped the reservation and let `push` grow the vector geometrically,
//! would abort this binary through `handle_alloc_error` instead of returning.
//!
//! Proving that needs an allocator that refuses, which is a process-wide setting,
//! so this suite is one test in its own binary: nothing else runs while the
//! refusal is armed. The refusal is size-gated above the harness's own small
//! allocations, so the failure observed is the one `smooth` asked for.
//!
//! Gated on `alloc`: without it there is no batch driver to drive, and the file
//! compiles to an empty test binary so the rest of the feature matrix still
//! builds.
#![cfg(any(feature = "std", feature = "alloc"))]

use std::{
  alloc::{GlobalAlloc, Layout, System},
  sync::atomic::{AtomicBool, AtomicUsize, Ordering},
  vec::Vec,
};

use windit::{
  plan::Span,
  smooth::{Ema, SmoothPolicy},
  windowed::Windowed,
  WinditError,
};

/// Refuse allocations of at least `LIMIT` bytes while `ARMED`; defer everything
/// else to the system allocator.
struct Refusing;

/// Whether the refusal is in effect.
static ARMED: AtomicBool = AtomicBool::new(false);

/// The size at or above which an armed allocation is refused. Left at `MAX`
/// (refuse nothing) except around the smoothing call itself.
static LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);

// SAFETY: every branch forwards to `System`, a correct allocator, or returns
// null — the documented way for `alloc` to report failure. No branch returns a
// pointer `System` did not hand out, and `dealloc` / `realloc` always forward, so
// nothing is freed by an allocator that did not allocate it.
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

/// Enough windows that the output vector is far larger than anything the harness
/// allocates, so the size gate can separate them.
const N: usize = 8192;

#[test]
fn batch_smooth_reports_a_refused_output_vector() {
  let input: Vec<Windowed<f32>> = (0..N)
    .map(|i| Windowed::new((i % 11) as f32 / 10.0, Span::new(i, 1, 1)))
    .collect();
  let cfg = Ema::new(0.5);

  // Unarmed, the ordinary answer — proof the input itself is fine, so the armed
  // run below is evidence about allocation and not about the data.
  assert_eq!(cfg.smooth(&input).expect("unarmed smooth").len(), N);

  // The output is `N` `Windowed<f32>`s; gate just under that so the reservation
  // under test is refused while the harness's smaller allocations pass. Read from
  // the type rather than written as a literal, so the gate cannot drift from the
  // layout it is aimed at.
  let output_bytes = N * core::mem::size_of::<Windowed<f32>>();
  LIMIT.store(output_bytes - 64, Ordering::Relaxed);
  ARMED.store(true, Ordering::Relaxed);
  let refused = cfg.smooth(&input);
  ARMED.store(false, Ordering::Relaxed);

  // Reaching this line at all is half the assertion: a `smooth` that asked for
  // its output infallibly would have aborted this binary instead of returning.
  assert_eq!(
    refused,
    Err(WinditError::AllocFailed { elements: N }),
    "smooth must report the refused output vector, and the count it asked for"
  );
}
