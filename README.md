<div align="center">
<h1>windit</h1>
</div>
<div align="center">

Generic windowed-sequence processing — chunk, pad/mask, aggregate, smooth, segment, split — for embeddings, VAD, and ASR.

[<img alt="crates.io" src="https://img.shields.io/crates/v/windit?style=for-the-badge&logo=rust" height="22">][crates-url]
[<img alt="docs.rs" src="https://img.shields.io/docsrs/windit?style=for-the-badge&logo=docs.rs" height="22">][doc-url]
[<img alt="license" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue?style=for-the-badge" height="22">][license-url]

</div>

## What it is

`windit` is the windowing and post-processing machinery — chunk an input into
window spans, pad and mask them, then aggregate / smooth / segment / split the
per-window results — in one standalone, generic crate. It follows the same
standalone-crate pattern as the `diaric` and `zuoer` pipelines: the shared
machinery lives in a generic crate of its own rather than being reimplemented
inside each model.

It owns no model code: no tokenizer, no resampler, no mel front-end, no
inference runtime. It is pure computation plus pluggable policy traits. The
same code windows 512-token embeddings, 8192-token long-context embeddings,
480 000-sample audio clips, and per-frame VAD probabilities with **no change** —
only the configuration differs. That window-size- and unit-agnosticism is the
crate's contract, enforced by an acceptance suite.

```toml
[dependencies]
windit = "0.1"
```

## The unifying idea: one geometry drives both ends

A single [`WindowPlan`] maps an input length plus a fully configurable
[`WindowOptions`] (window size, hop / overlap, tail handling, an optional window
cap) to a list of [`Span`]s. A `Span` is plain `usize` element counts, so
samples, tokens, patches, and frames are treated identically.

Those same spans drive **both** ends of a pipeline:

- **pre**: slice each span out of the input, right-pad it to a full window, and
  emit a `1`/`0` real-vs-pad mask (`slice_pad_mask`).
- **post**: carry each window's result in a [`Windowed<V>`] (a value paired with
  its span) and aggregate, smooth, or segment it.

The span is the thread that ties the two ends together: the geometry computed
once for slicing is the same geometry used to weight, place, and merge the
results.

```rust
use windit::prelude::*;

// 1500 tokens, non-overlapping windows of 512 -> three spans (two full, one tail).
let opts = WindowOptions::new(512);
let spans = WindowPlan::spans(&opts, 1500).unwrap();
assert_eq!(spans.len(), 3);
assert_eq!(spans[2].coverage() < 1.0, true); // the ragged tail
```

## The four strategy families

Each family is an object-safe trait plus shipped built-in policies; bring your
own by implementing the trait.

- **aggregate** — collapse a sequence of window embeddings into one embedding.
  Built-ins: [`CoverageWeightedMean`] (default), [`MeanRenormalized`],
  [`EmaRenormalized`], [`SaliencyWeighted`], plus `keep_separate` for the
  multi-vector path. Embeddings live in f32 space and are reconstructed through
  the minimal [`Vector`] trait, so any 384-, 512-, or 768-dimension type fits.
- **smooth** — rewrite each window's value while preserving its span. Built-ins:
  [`Ema`] (temporal low-pass) and [`Hysteresis`] (a latching two-threshold gate
  for binary VAD).
- **segment** — reduce a windowed score sequence to continuous element
  [`Range`]s: `runs`, `longest_run`, and `runs_sorted`, with `min_len` and
  `merge_gap` post-passes. Policies [`Threshold`] and [`HysteresisSegment`]
  package a predicate with its options.
- **split** — decide how an input is divided before windowing. [`FixedWindow`]
  delegates to the planner; [`ContentAware`] (feature `text`) chunks strings.

```rust
use windit::prelude::*;

// Find the longest continuous speech region from per-frame probabilities.
let frames: Vec<Windowed<f32>> = /* one Windowed<f32> per VAD frame */;
let opts = SegmentOptions::new().with_min_len(3).with_merge_gap(2);
let speech = longest_run(&frames, |&p| p >= 0.5, &opts);
```

## Content-aware text chunking: the `len_fn` callback

[`ContentAware`] (feature `text`) is a tokenizer-free string chunker. It splits
on recursive boundaries — paragraphs, then sentences, then words — and greedily
packs the pieces into chunks no longer than the window. Crucially, it measures
length through a **caller-supplied `len_fn` callback**, so *you* define what a
"token" is (a word count, a real tokenizer's id count, code points) without this
crate ever depending on a tokenizer:

```rust
use windit::split::ContentAware;
use windit::WindowOptions;

// "tokens" = whitespace-separated words, windows of 32, overlap of 4.
let count = |s: &str| s.split_whitespace().count();
let chunker = ContentAware { len_fn: &count };
let opts = WindowOptions::new(32).with_overlap(4);
let ranges = chunker.chunk(document, &opts); // Vec<(usize, usize)> byte ranges
```

## Custom policies

Every family is a trait, so a project-specific strategy is a small `impl`. An
aggregate policy that simply keeps the first window:

```rust
use windit::aggregate::AggregatePolicy;
use windit::WinditError;

struct FirstWindow;

impl AggregatePolicy for FirstWindow {
    fn aggregate_f32(
        &self,
        embeddings: &[&[f32]],
        _coverages: &[f32],
        dim: usize,
    ) -> Result<Vec<f32>, WinditError> {
        let first = embeddings.first().ok_or(WinditError::Empty)?;
        if first.len() != dim {
            return Err(WinditError::DimMismatch { got: first.len(), expected: dim });
        }
        Ok(first.to_vec())
    }
}
```

The aggregate trait works entirely in f32 space, which keeps it object-safe
(`&dyn AggregatePolicy`) while embedding reconstruction stays generic through
the free `aggregate` function.

## `no_std`

`windit` is `no_std + alloc`. `default = ["alloc"]`, since every operation
returns a `Vec`; the type and trait surface compiles even without it. The
optional features are additive:

| Feature   | Adds                                                              |
|-----------|------------------------------------------------------------------|
| `alloc`   | (default) the `Vec`-returning planner and the strategy families  |
| `std`     | the [`std::error::Error`] implementation for `WinditError`       |
| `text`    | content-aware string chunking (`unicode-segmentation`)           |
| `serde`   | `Serialize` / `Deserialize` for the configuration and policy enums |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

[crates-url]: https://crates.io/crates/windit
[doc-url]: https://docs.rs/windit
[license-url]: https://github.com/findit-studio/windit#license

[`WindowPlan`]: https://docs.rs/windit/latest/windit/plan/struct.WindowPlan.html
[`WindowOptions`]: https://docs.rs/windit/latest/windit/plan/struct.WindowOptions.html
[`Span`]: https://docs.rs/windit/latest/windit/plan/struct.Span.html
[`Windowed<V>`]: https://docs.rs/windit/latest/windit/windowed/struct.Windowed.html
[`Vector`]: https://docs.rs/windit/latest/windit/windowed/trait.Vector.html
[`CoverageWeightedMean`]: https://docs.rs/windit/latest/windit/aggregate/struct.CoverageWeightedMean.html
[`MeanRenormalized`]: https://docs.rs/windit/latest/windit/aggregate/struct.MeanRenormalized.html
[`EmaRenormalized`]: https://docs.rs/windit/latest/windit/aggregate/struct.EmaRenormalized.html
[`SaliencyWeighted`]: https://docs.rs/windit/latest/windit/aggregate/struct.SaliencyWeighted.html
[`Ema`]: https://docs.rs/windit/latest/windit/smooth/struct.Ema.html
[`Hysteresis`]: https://docs.rs/windit/latest/windit/smooth/struct.Hysteresis.html
[`Range`]: https://docs.rs/windit/latest/windit/segment/struct.Range.html
[`Threshold`]: https://docs.rs/windit/latest/windit/segment/struct.Threshold.html
[`HysteresisSegment`]: https://docs.rs/windit/latest/windit/segment/struct.HysteresisSegment.html
[`FixedWindow`]: https://docs.rs/windit/latest/windit/split/struct.FixedWindow.html
[`ContentAware`]: https://docs.rs/windit/latest/windit/split/struct.ContentAware.html
[`std::error::Error`]: https://doc.rust-lang.org/std/error/trait.Error.html
