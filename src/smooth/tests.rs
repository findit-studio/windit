use alloc::{vec, vec::Vec};

use super::{Ema, Hysteresis, SmoothPolicy};
use crate::{plan::Span, windowed::Windowed};

/// One `Windowed<f32>` per value, each covering a single element (window 1).
fn seq(values: &[f32]) -> Vec<Windowed<f32>> {
  values
    .iter()
    .enumerate()
    .map(|(i, &v)| {
      Windowed::new(
        v,
        Span {
          start: i,
          len: 1,
          window: 1,
        },
      )
    })
    .collect()
}

fn values(seq: &[Windowed<f32>]) -> Vec<f32> {
  seq.iter().map(|w| w.value).collect()
}

fn spans(seq: &[Windowed<f32>]) -> Vec<Span> {
  seq.iter().map(|w| w.span).collect()
}

#[test]
fn ema_pinned_and_preserves_spans() {
  // alpha 0.5, s_0 = x_0: 0, 0.5*1, 0.5*1 + 0.5*0.5, 0.5*0.75 -> exact dyadics.
  let input = seq(&[0.0, 1.0, 1.0, 0.0]);
  let out = Ema { alpha: 0.5 }.smooth(&input);
  assert_eq!(values(&out), vec![0.0, 0.5, 0.75, 0.375]);
  assert_eq!(spans(&out), spans(&input));
}

#[test]
fn hysteresis_latches_and_holds() {
  // on=0.6, off=0.3: 0.1 off, 0.7 on, 0.5 hold(on), 0.2 off, 0.6 on.
  let input = seq(&[0.1, 0.7, 0.5, 0.2, 0.6]);
  let out = Hysteresis { on: 0.6, off: 0.3 }.smooth(&input);
  assert_eq!(values(&out), vec![0.0, 1.0, 1.0, 0.0, 1.0]);
  assert_eq!(spans(&out), spans(&input));
}

#[test]
fn ema_clamps_alpha_and_never_emits_nan() {
  // alpha 2.0 clamps to 1.0 (track the input exactly): s_t = x_t.
  let input = seq(&[0.5, 0.8]);
  let out = Ema { alpha: 2.0 }.smooth(&input);
  assert_eq!(values(&out), vec![0.5, 0.8]);

  // A NaN alpha clamps to 0.0 (hold the seed): every output is finite, so the
  // infallible smooth path never leaks a NaN downstream.
  let out = Ema { alpha: f32::NAN }.smooth(&input);
  assert!(
    out.iter().all(|w| w.value.is_finite()),
    "outputs must be finite"
  );
  assert_eq!(values(&out), vec![0.5, 0.5]);
}

#[test]
fn empty_input_yields_empty() {
  let input: Vec<Windowed<f32>> = Vec::new();
  assert!(Ema { alpha: 0.5 }.smooth(&input).is_empty());
  assert!(Hysteresis { on: 0.6, off: 0.3 }.smooth(&input).is_empty());
}
