//! Commonly used `windit` types, re-exported for a single glob import.
//!
//! ```
//! use windit::prelude::*;
//! ```

pub use crate::{
  error::WinditError,
  plan::{Span, TailPolicy, WindowOptions},
  windowed::{Vector, WindowEmbedding, Windowed},
};

#[cfg(feature = "alloc")]
pub use crate::plan::WindowPlan;
