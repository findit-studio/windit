//! Every aggregation buffer grown on a `Result`-returning path reports a
//! refusing allocator, rather than aborting.
//!
//! `aggregate` and each built-in policy size their scratch by the caller's window
//! count and embedding dimension — counts that need not correspond to memory that
//! exists — and grow it on a path that returns `Result`. An allocation those
//! cannot get must therefore come back as `WinditError::AllocFailed`, the same
//! contract `try_slice_pad_mask` and `ContentAware::chunk` keep; an abort would
//! break it.
//!
//! Proving that needs an allocator that refuses, which is a process-wide setting,
//! so this suite is one binary: nothing else runs while the refusal is armed. The
//! refusal is size-gated above the harness's own small allocations, so the
//! failure observed is the one the aggregation asked for. Each built-in policy
//! gets its own case: they grow different buffers (`MeanRenormalized` and
//! `CoverageWeightedMean` the accumulator, `EmaRenormalized` its state,
//! `SaliencyWeighted` its weights and accumulator), so a sweep that missed one
//! would still pass the others.
//!
//! Gated on `alloc`: without it there is no aggregation to drive, and the file
//! compiles to an empty test binary so the rest of the feature matrix still
//! builds.
#![cfg(any(feature = "std", feature = "alloc"))]
// Coverage instrumentation allocates inside the guarded window; the sanitizer
// and plain lanes already cover this guard, so it steps aside under tarpaulin.
#![cfg(not(tarpaulin))]

use std::{
  alloc::{GlobalAlloc, Layout, System},
  sync::atomic::{AtomicBool, AtomicUsize, Ordering},
  vec::Vec,
};

use windit::{
  aggregate::{
    AggregatePolicy, CoverageWeightedMean, EmaRenormalized, MeanRenormalized, SaliencyWeighted,
  },
  WinditError,
};

/// Refuse allocations of at least `LIMIT` bytes while `ARMED`; defer everything
/// else to the system allocator.
struct Refusing;

/// Whether the refusal is in effect.
static ARMED: AtomicBool = AtomicBool::new(false);

/// The size at or above which an armed allocation is refused. Set above the
/// harness's own churn and below the dimension-sized buffer under test, so the
/// refusal lands on the aggregation's growth and not on the test's bookkeeping.
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

/// Run `policy.aggregate_values` over two `dim`-length windows with the refusal
/// armed for allocations of at least `dim` `f64`s, and return its result.
///
/// The embeddings are ordinary unit-ish directions, so unarmed the call
/// succeeds — which is what makes the armed run evidence about allocation and not
/// about the data. Everything the call needs but the buffer under test is
/// allocated before the refusal is armed.
fn refused<P: AggregatePolicy<f64>>(policy: &P, dim: usize) -> Result<Vec<f64>, WinditError> {
  let a: Vec<f64> = (0..dim).map(|i| 1.0 + (i % 7) as f64).collect();
  let b: Vec<f64> = (0..dim).map(|i| 2.0 + (i % 5) as f64).collect();
  let embeddings: [&[f64]; 2] = [&a, &b];
  let coverages = [1.0_f64, 1.0];

  // Unarmed, the ordinary answer — proof the geometry itself is fine.
  policy
    .aggregate_values(&embeddings, &coverages, dim)
    .expect("unarmed aggregation must succeed");

  // A dim-length f64 buffer is 8 * dim bytes; gate just under that so the buffer
  // under test is refused while the harness's smaller allocations pass.
  LIMIT.store(dim * 8 - 64, Ordering::Relaxed);
  ARMED.store(true, Ordering::Relaxed);
  let out = policy.aggregate_values(&embeddings, &coverages, dim);
  ARMED.store(false, Ordering::Relaxed);
  out
}

/// A dimension whose `f64` buffer (`8 * DIM` bytes) is comfortably larger than
/// the harness's own allocations, so the size gate can separate them.
const DIM: usize = 8192;

/// One test, not four: the refusal is a process-wide setting, and the default
/// harness runs a binary's tests in parallel, so each built-in is checked
/// sequentially here rather than racing the others to arm the allocator. Each
/// grows a different buffer — the accumulator for the two means, the state for
/// the EMA, the weights then accumulator for saliency — so a sweep that left one
/// on an infallible `Vec::with_capacity`/`vec!`/`to_vec`/`collect` would surface
/// as that policy aborting while the rest returned `AllocFailed`.
#[test]
fn every_builtin_policy_reports_a_refused_buffer() {
  assert!(
    matches!(
      refused(&CoverageWeightedMean, DIM),
      Err(WinditError::AllocFailed { .. })
    ),
    "CoverageWeightedMean must report a refused accumulator, not abort on it"
  );
  assert!(
    matches!(
      refused(&MeanRenormalized, DIM),
      Err(WinditError::AllocFailed { .. })
    ),
    "MeanRenormalized must report a refused accumulator, not abort on it"
  );
  assert!(
    matches!(
      refused(&EmaRenormalized::new(0.5), DIM),
      Err(WinditError::AllocFailed { .. })
    ),
    "EmaRenormalized must report a refused state, not abort on it"
  );
  assert!(
    matches!(
      refused(&SaliencyWeighted, DIM),
      Err(WinditError::AllocFailed { .. })
    ),
    "SaliencyWeighted must report a refused weight/accumulator buffer, not abort on it"
  );
}
