<div align="center">
<h1>windit</h1>
</div>
<div align="center">

Generic windowed-sequence processing — chunk, pad/mask, aggregate, smooth, segment, split — for embeddings, VAD, and ASR.

[<img alt="github" src="https://img.shields.io/badge/github-findit--studio/windit-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fwindit" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-studio/windit/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-studio/windit?style=for-the-badge&token=6R3QFWRWHL&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-windit-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/windit?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPC9zdmc+DQo=" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/windit?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
[<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge" height="22">][license-url]


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

// Per-frame speech probabilities: one `Windowed<f32>` per VAD frame, each frame
// covering a single element.
let probs = [0.1, 0.9, 0.8, 0.2, 0.7, 0.9, 0.6];
let frames: Vec<Windowed<f32>> = probs
  .iter()
  .enumerate()
  .map(|(i, &p)| Windowed::new(p, Span::new(i, 1, 1)))
  .collect();

// Find the longest continuous speech region, ignoring runs under two frames.
let opts = SegmentOptions::new().with_min_len(2);
let speech = longest_run(&frames, |&p| p >= 0.5, &opts);
assert_eq!(speech, Some(Range::new(4, 7)));
```

## Content-aware text chunking: the `len_fn` callback

[`ContentAware`] (feature `text`) is a tokenizer-free string chunker. It splits
on recursive boundaries — paragraphs, then sentences, then words — and greedily
packs the pieces into chunks no longer than the window. Crucially, it measures
length through a **caller-supplied `len_fn` callback**, so *you* define what a
"token" is (a word count, a real tokenizer's id count, code points) without this
crate ever depending on a tokenizer:

```rust,ignore
use windit::plan::WindowOptions;
use windit::split::ContentAware;

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

[Github-url]: https://github.com/Findit-AI/windit
[CI-url]: https://github.com/Findit-AI/windit/actions/workflows/ci.yml
[codecov-url]: https://app.codecov.io/gh/Findit-AI/windit/
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
