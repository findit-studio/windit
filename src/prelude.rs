//! Commonly used `windit` types, re-exported for a single glob import.
//!
//! ```
//! use windit::prelude::*;
//! ```

pub use crate::{
  error::WinditError,
  plan::{Span, TailPolicy, WindowOptions},
};

#[cfg(feature = "alloc")]
pub use crate::plan::WindowPlan;
