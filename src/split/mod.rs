//! Split policies: decide how an input is divided before windowing.
//!
//! `SplitPolicy` maps an input length and `WindowOptions` to the `Span`s
//! that cover it. `FixedWindow` is the mechanical element-count split — it
//! delegates to `WindowPlan::spans`, so any unit (samples, tokens, patches)
//! is divided the same way.
//!
//! `ContentAware` (feature `text`) is the tokenizer-free string chunker: it
//! packs text into chunks that respect paragraph, sentence, and word boundaries,
//! measuring length through a caller-supplied `MeasureText` — any
//! `Fn(&str) -> usize` closure — so the caller's own tokenizer defines "how
//! long". It is a separate surface from `SplitPolicy` because it returns
//! `Chunk`s — half-open UTF-8 byte ranges — rather than element `Span`s.

use std::vec::Vec;

use crate::{
  error::WinditError,
  plan::{Span, WindowOptions, WindowPlan},
};

#[cfg(test)]
mod tests;

/// A policy that divides an input of `input_len` elements into [`Span`]s.
///
/// The trait is object-safe (`&dyn SplitPolicy` works): the only method takes an
/// element count and [`WindowOptions`] and returns the plan.
pub trait SplitPolicy {
  /// Plan the spans covering `input_len` elements under `opts`.
  ///
  /// # Errors
  ///
  /// Returns whatever the underlying geometry rejects — see
  /// [`WindowPlan::spans`] (for example [`WinditError::ZeroWindow`] or
  /// [`WinditError::TooManyWindows`]).
  fn split(&self, input_len: usize, opts: &WindowOptions) -> Result<Vec<Span>, WinditError>;
}

/// The mechanical fixed-window split: a direct delegation to [`WindowPlan::spans`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FixedWindow;

impl SplitPolicy for FixedWindow {
  fn split(&self, input_len: usize, opts: &WindowOptions) -> Result<Vec<Span>, WinditError> {
    WindowPlan::spans(opts, input_len)
  }
}

/// A half-open UTF-8 byte range `[start, end)` into a source text (feature
/// `text`).
///
/// [`ContentAware::chunk`] guarantees every `Chunk` it returns falls on `char`
/// boundaries of the text it was cut from, so [`as_str`](Chunk::as_str) never
/// returns `None` for that text. The guarantee belongs to the chunker: a
/// `Chunk` is a bare pair of byte offsets, borrowing nothing, so nothing on
/// this type itself can check it.
///
/// Deliberately not [`Range`](crate::segment::Range): `Range` counts *input
/// elements* (samples, tokens, patches, frames), one per index, so it is
/// independent of encoding. A `Chunk` counts UTF-8 *bytes*, several of which
/// can make up one `char`. Sharing one type for both would let a byte offset
/// and an element index trade places at a call site with no compiler error —
/// the exact ambiguity this type exists to rule out.
#[cfg(feature = "text")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Chunk {
  start: usize,
  end: usize,
}

/// Measures text length in the caller's own units, with a bounded query that
/// may stop early (feature `text`).
///
/// [`ContentAware`] measures every candidate range through this trait, so the
/// caller defines what a "token" is — a word count, a tokenizer's id count, code
/// points — without this crate depending on a tokenizer. Every
/// `Fn(&str) -> usize` closure implements it through a blanket impl, so an
/// ordinary caller writes a closure and nothing else; a tokenizer that wants to
/// bound its work on untrusted input implements the trait directly.
///
/// # Bounding untrusted input
///
/// [`measure`](MeasureText::measure) measures a whole string and must run to
/// completion, so on untrusted input it scans the entire text before the chunker
/// can decide anything — including before [`WindowOptions::max_windows`] can
/// reject it. [`measure_within`](MeasureText::measure_within) is the query the
/// chunker actually makes: it may stop as soon as the count is known to exceed a
/// `limit`, so a range many times longer than a window is rejected after
/// measuring only about a window's worth of it. A measurer that counts
/// incrementally overrides it to gain that bound; the blanket closure impl
/// cannot, and falls back to a full [`measure`](MeasureText::measure).
#[cfg(feature = "text")]
#[cfg_attr(docsrs, doc(cfg(feature = "text")))]
pub trait MeasureText {
  /// Measure the whole of `text`, in the caller's units.
  fn measure(&self, text: &str) -> usize;

  /// Measure `text` only far enough to compare it against `limit`: return
  /// `Some(n)` with `n` equal to [`measure`](MeasureText::measure) when that
  /// count is at most `limit`, and `None` as soon as it is known to exceed
  /// `limit`.
  ///
  /// The chunker asks this instead of [`measure`](MeasureText::measure) so a
  /// range far longer than a window is not scanned in full merely to learn it
  /// does not fit. An implementation that can count incrementally should stop at
  /// the first unit past `limit` and return `None`; the default runs a full
  /// [`measure`](MeasureText::measure) and then compares, which is correct but
  /// unbounded — overriding it is what lets a large untrusted input be rejected
  /// cheaply.
  ///
  /// An override must agree with [`measure`](MeasureText::measure): return
  /// `Some(self.measure(text))` exactly when that value is `<= limit`, and
  /// `None` otherwise. The chunker relies on this equivalence to place the same
  /// boundaries a full measure would, so a disagreeing override moves chunk
  /// boundaries rather than merely changing cost.
  fn measure_within(&self, text: &str, limit: usize) -> Option<usize> {
    let measured = self.measure(text);
    (measured <= limit).then_some(measured)
  }
}

/// Every `Fn(&str) -> usize` closure is a [`MeasureText`], measured in full with
/// no early stop, so an ordinary caller supplies a closure without writing a
/// trait impl. `?Sized` is not needed: closures and function pointers are
/// `Sized`, and a pre-erased `&dyn Fn(&str) -> usize` cannot coerce to
/// `&dyn MeasureText` regardless.
#[cfg(feature = "text")]
impl<F: Fn(&str) -> usize> MeasureText for F {
  fn measure(&self, text: &str) -> usize {
    self(text)
  }
}

/// A tokenizer-free, content-aware string chunker (feature `text`).
///
/// [`chunk`](ContentAware::chunk) splits text on recursive boundaries —
/// paragraphs (`\n\n`), then sentences, then words — and greedily packs the
/// pieces into chunks no longer than [`WindowOptions::window`], as measured by
/// the caller's [`MeasureText`]. Because length is whatever that measurer
/// reports, the caller supplies its own notion of a token (word count, a
/// tokenizer's id count, code points) without this crate depending on a
/// tokenizer.
#[cfg(feature = "text")]
#[derive(Clone, Copy)]
pub struct ContentAware<'a> {
  measurer: &'a dyn MeasureText,
}

#[cfg(feature = "text")]
#[cfg_attr(docsrs, doc(cfg(feature = "text")))]
const _: () = {
  use core::iter::Peekable;

  use std::collections::VecDeque;
  use unicode_segmentation::{USentenceBoundIndices, UnicodeSegmentation, UnicodeWordIndices};

  /// The paragraph separator the coarsest boundary level splits on.
  const PARAGRAPH_SEPARATOR: &str = "\n\n";

  impl Chunk {
    /// The half-open byte range from `start` up to (but not including) `end`.
    ///
    /// The infallible counterpart to [`try_new`](Chunk::try_new), for the
    /// callers that know their bounds and would only unwrap.
    ///
    /// # Panics
    ///
    /// Panics, in every build, if `start > end`. Use
    /// [`try_new`](Chunk::try_new) to handle untrusted bounds instead.
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
      match Self::try_new(start, end) {
        Ok(chunk) => chunk,
        Err(_) => panic!("a chunk must satisfy start <= end"),
      }
    }

    /// The checked counterpart of [`new`](Chunk::new): validate the bounds
    /// rather than panic on them.
    ///
    /// # Errors
    ///
    /// Returns [`WinditError::InvalidChunk`] if `start > end`.
    pub const fn try_new(start: usize, end: usize) -> Result<Self, WinditError> {
      if start > end {
        return Err(WinditError::InvalidChunk { start, end });
      }
      Ok(Self { start, end })
    }

    /// The first byte offset covered by the chunk.
    #[must_use]
    pub const fn start(&self) -> usize {
      self.start
    }

    /// One past the last byte offset covered by the chunk.
    #[must_use]
    pub const fn end(&self) -> usize {
      self.end
    }

    /// The number of bytes the chunk covers (`end - start`).
    ///
    /// Never underflows: both constructors reject `start > end` in every
    /// build.
    #[must_use]
    pub const fn len(&self) -> usize {
      self.end.saturating_sub(self.start)
    }

    /// Whether the chunk covers no bytes (`start >= end`).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
      self.start >= self.end
    }

    /// The slice of `text` this chunk names, or `None` if `text` does not have
    /// `char` boundaries at both `start` and `end` — for instance because
    /// `text` is not the string the chunk was cut from.
    ///
    /// A `Chunk` does not borrow the text it was produced from, so nothing
    /// stops a caller from passing a different string here; `Option` makes
    /// that honestly fallible rather than panicking the way indexing
    /// (`&text[start..end]`) would.
    #[must_use]
    pub fn as_str<'a>(&self, text: &'a str) -> Option<&'a str> {
      text.get(self.start..self.end)
    }
  }

  impl<'a> ContentAware<'a> {
    /// A chunker that measures text length with `measurer`.
    ///
    /// `measurer` is any [`MeasureText`], which every `Fn(&str) -> usize`
    /// closure implements, so passing `&|s: &str| s.split_whitespace().count()`
    /// is a whitespace-word chunker. [`chunk`](ContentAware::chunk) guarantees
    /// every returned chunk measures at most the window under it, save for a
    /// single `char` that alone exceeds the window (it cannot be split further).
    #[must_use]
    pub const fn new(measurer: &'a dyn MeasureText) -> Self {
      Self { measurer }
    }

    /// The caller-supplied measurer, in whose units the window is measured.
    #[must_use]
    pub const fn measurer(&self) -> &'a dyn MeasureText {
      self.measurer
    }

    /// Chunk `text` into [`Chunk`]s, each measuring at most
    /// [`WindowOptions::window`] under the [`MeasureText`].
    ///
    /// Each returned [`Chunk`] falls on `char` boundaries, so
    /// [`Chunk::as_str`] applied to this same `text` never returns `None`.
    /// Boundaries are preferred coarse-to-fine: a paragraph, sentence, or word
    /// that fits is kept whole, and only a unit that overflows the window on
    /// its own is split further — a long sentence into words, a long word
    /// into measured character slices. Chunks are packed greedily;
    /// when [`WindowOptions::overlap`] is non-zero, consecutive chunks repeat
    /// at most that many trailing tokens' worth of whole boundary units (as
    /// measured over the exact repeated text).
    ///
    /// Chunks cover the tokenized content in order; inter-token whitespace
    /// falls inside a chunk's range but is never a chunk of its own. An empty
    /// or whitespace-only input yields no chunks.
    ///
    /// Every returned chunk measures at most the window, with a single
    /// exception: a lone `char` whose own measure exceeds the window is
    /// emitted as-is, because it cannot be split further. In the
    /// oversized-sentence word fallback, punctuation lying outside word
    /// boundaries (a trailing period, say) is not covered by any chunk.
    ///
    /// # Cost
    ///
    /// Length is queried through [`MeasureText::measure_within`] `O(a)` times for
    /// `a` atoms: the packing never re-measures a range whose measure it already
    /// knows, and it locates each overlap boundary by a linear scan over just the
    /// trailing atoms of the chunk it closes. Those scans cover adjacent atom
    /// ranges that tile the atom stream, so together they stay linear in `a`. The
    /// scan is not a bisection because a context-sensitive measurer's token count
    /// need not fall monotonically as the repeated suffix shortens, so only a walk
    /// from the longest candidate suffix inward finds the earliest one that fits.
    ///
    /// `a` counts the atoms the emitted chunks are built from, not the atoms the
    /// whole text contains. Atoms are produced on demand and packed as they are
    /// produced, so [`WindowOptions::max_windows`] bounds the number of atoms
    /// produced, and the memory, as well as the chunk count: a capped chunking
    /// stops at the first chunk past the cap and never splits the text beyond it.
    /// Peak memory is one chunk's worth of atoms — the block the overlap search
    /// scans back through — plus the chunks emitted so far.
    ///
    /// Whether the cap also bounds the *measurement* is up to the measurer. Each
    /// descent level, down to the first atom, measures its whole range once, and
    /// the first such range is the entire input. A measurer whose
    /// [`measure_within`](MeasureText::measure_within) stops early reads only
    /// about a window of any such range, so that measurement costs a window, not
    /// the input length. A plain `Fn(&str) -> usize` closure cannot stop early
    /// and measures each range in full, so an untrusted input is still scanned a
    /// few times — once per descent level above the first atom — even under a
    /// cap of zero; implement [`MeasureText`] with an early stop to bound that.
    ///
    /// # Errors
    ///
    /// - Whatever [`WindowOptions::validate`] rejects — [`WinditError::ZeroWindow`]
    ///   for a zero window, [`WinditError::OverlapGeWindow`] for an overlap at or
    ///   above it — since neither can produce a chunk that honours the window.
    /// - [`WinditError::TooManyWindows`] if the packing exceeds
    ///   [`WindowOptions::max_windows`], reported exactly as [`WindowPlan::spans`]
    ///   reports it.
    /// - [`WinditError::AllocFailed`] if the chunk list or the packer's atom
    ///   buffer cannot be grown. Neither size is known before packing runs, so
    ///   neither can be reserved up front the way [`WindowPlan::spans`] reserves
    ///   its plan; both grow fallibly instead, so an allocator that refuses is
    ///   reported here rather than aborting a call that returns `Result`.
    ///
    /// ```
    /// use windit::plan::WindowOptions;
    /// use windit::split::ContentAware;
    ///
    /// // "tokens" = whitespace-separated words, so the window counts words.
    /// let count = |s: &str| s.split_whitespace().count();
    /// let chunker = ContentAware::new(&count);
    ///
    /// let text = "a b c d e f g h i j k l";
    /// let chunks = chunker.chunk(text, &WindowOptions::new(4)).unwrap();
    /// assert_eq!(chunks.len(), 3);
    /// for chunk in &chunks {
    ///   assert!(count(chunk.as_str(text).unwrap()) <= 4);
    /// }
    /// ```
    pub fn chunk(&self, text: &str, opts: &WindowOptions) -> Result<Vec<Chunk>, WinditError> {
      opts.validate()?;
      pack(text, opts, self.measurer)
    }
  }

  /// The recursion level: which boundary to split on when a range overflows.
  #[derive(Clone, Copy)]
  enum Level {
    Paragraph,
    Sentence,
    Word,
  }

  /// What advancing one level of the descent produced.
  enum Step {
    /// An atom, ready to be packed.
    Atom(Chunk),
    /// A sub-range, to be divided at `level` if it overflows the window.
    Split {
      start: usize,
      end: usize,
      level: Level,
    },
    /// One Unicode word: an atom if it fits the window, `char`-aligned slices if
    /// it does not.
    Word { start: usize, end: usize },
    /// The level has no ranges left.
    Done,
  }

  /// One boundary level of the descent, suspended at the position it has
  /// reached.
  enum Frame<'a> {
    /// Walk the `\n\n`-separated paragraphs of a range.
    Paragraphs {
      cursor: usize,
      end: usize,
      done: bool,
    },
    /// Walk the Unicode sentences of a paragraph.
    Sentences {
      iter: USentenceBoundIndices<'a>,
      base: usize,
    },
    /// Walk the Unicode words of a sentence (whitespace and punctuation between
    /// words are excluded).
    ///
    /// Peekable because the level's fallback turns on whether the sentence has
    /// any word at all, which is settled before the frame is suspended; the
    /// peeked word is still the frame's next item, so no boundary is walked
    /// twice.
    Words {
      iter: Peekable<UnicodeWordIndices<'a>>,
      base: usize,
    },
    /// Walk one oversized unit as maximal `char`-aligned slices that fit the
    /// window.
    Chars {
      seg_start: usize,
      cursor: usize,
      end: usize,
    },
  }

  impl Frame<'_> {
    /// Advance to this level's next range, without descending into it.
    ///
    /// The `Chars` arm is the crate's other measurement walk over a growing
    /// substring, and it is left as a walk deliberately. Each slice is measured
    /// from its own start, so the measurement resets at every emitted boundary:
    /// the cost is the range's length times one window, not times itself, which
    /// is the same shape the packing bound has. Probing instead would need the
    /// measure to be monotone over a growing prefix, and a BPE tokenizer is not.
    fn advance(&mut self, text: &str, window: usize, measurer: &dyn MeasureText) -> Step {
      match self {
        Self::Paragraphs { cursor, end, done } => {
          if *done {
            return Step::Done;
          }
          let start = *cursor;
          // Searching from the cursor rather than over the whole range is what
          // makes the paragraph level lazy, and it matches the non-overlapping
          // scan it replaces: the cursor sits just past the previous separator,
          // exactly where that scan would have resumed.
          let para_end = match text[start..*end].find(PARAGRAPH_SEPARATOR) {
            Some(off) => {
              let para_end = start + off;
              *cursor = para_end + PARAGRAPH_SEPARATOR.len();
              para_end
            }
            None => {
              *done = true;
              *end
            }
          };
          Step::Split {
            start,
            end: para_end,
            level: Level::Sentence,
          }
        }
        Self::Sentences { iter, base } => match iter.next() {
          Some((off, sentence)) => Step::Split {
            start: *base + off,
            end: *base + off + sentence.len(),
            level: Level::Word,
          },
          None => Step::Done,
        },
        Self::Words { iter, base } => match iter.next() {
          Some((off, word)) => Step::Word {
            start: *base + off,
            end: *base + off + word.len(),
          },
          None => Step::Done,
        },
        Self::Chars {
          seg_start,
          cursor,
          end,
        } => {
          while let Some(ch) = text[*cursor..*end].chars().next() {
            let abs = *cursor;
            let next = abs + ch.len_utf8();
            *cursor = next;
            // Close the current slice before the char that would overflow it, but
            // only once it holds at least one char, so a single oversized char
            // still emits.
            if abs > *seg_start
              && measurer
                .measure_within(&text[*seg_start..next], window)
                .is_none()
            {
              let atom = Chunk::new(*seg_start, abs);
              *seg_start = abs;
              return Step::Atom(atom);
            }
          }
          if *seg_start < *end {
            let atom = Chunk::new(*seg_start, *end);
            *seg_start = *end;
            return Step::Atom(atom);
          }
          Step::Done
        }
      }
    }
  }

  /// The atom producer: the boundary descent, suspended between atoms.
  ///
  /// Descending recursively would settle every atom of the input before the
  /// first one could be packed, which is how a bound on the chunk count came to
  /// be checked only after the work it bounds had been done. Holding the
  /// recursion as an explicit stack instead lets each `next` advance the descent
  /// by the least it can: nothing past the current position exists yet, so a
  /// packing that stops early leaves the rest of the text unsplit and
  /// unmeasured.
  ///
  /// Which ranges are visited, and in what order, is unchanged — so is the
  /// measurement that each visit makes. An uncapped chunking therefore returns
  /// the same chunks at the same cost as the recursion did.
  struct Atoms<'a> {
    text: &'a str,
    window: usize,
    measurer: &'a dyn MeasureText,
    /// The whole-input descent, held until the first `next` so that construction
    /// stays infallible.
    root: Option<(usize, usize)>,
    stack: Vec<Frame<'a>>,
  }

  impl<'a> Atoms<'a> {
    fn new(text: &'a str, window: usize, measurer: &'a dyn MeasureText) -> Self {
      Self {
        text,
        window,
        measurer,
        root: Some((0, text.len())),
        stack: Vec::new(),
      }
    }

    /// The next atom in input order, or `None` once the text has no more.
    ///
    /// # Errors
    ///
    /// [`WinditError::AllocFailed`] if the descent stack cannot be grown. It
    /// holds at most one frame per boundary level, so this is the allocator
    /// refusing a handful of bytes rather than a size the input can drive.
    fn next(&mut self) -> Result<Option<Chunk>, WinditError> {
      if let Some((start, end)) = self.root.take() {
        if let Some(atom) = self.split(start, end, Level::Paragraph)? {
          return Ok(Some(atom));
        }
      }
      let (text, window, measurer) = (self.text, self.window, self.measurer);
      loop {
        let step = match self.stack.last_mut() {
          Some(frame) => frame.advance(text, window, measurer),
          None => return Ok(None),
        };
        match step {
          Step::Atom(atom) => return Ok(Some(atom)),
          Step::Split { start, end, level } => {
            if let Some(atom) = self.split(start, end, level)? {
              return Ok(Some(atom));
            }
          }
          Step::Word { start, end } => {
            if let Some(atom) = self.word(start, end)? {
              return Ok(Some(atom));
            }
          }
          Step::Done => {
            self.stack.pop();
          }
        }
      }
    }

    /// Take `text[start..end]` whole when it fits the window, drop it when it
    /// holds no content, and otherwise suspend the level that divides it.
    fn split(
      &mut self,
      start: usize,
      end: usize,
      level: Level,
    ) -> Result<Option<Chunk>, WinditError> {
      let text: &'a str = self.text;
      if text[start..end].trim().is_empty() {
        return Ok(None);
      }
      // The measurement that ends the descent's unbounded parent scan: a range
      // far longer than the window — the whole input, at the first `next` — is
      // only measured far enough to learn it does not fit, so an early-stopping
      // measurer reads about a window of it rather than all of it.
      if self
        .measurer
        .measure_within(&text[start..end], self.window)
        .is_some()
      {
        return Ok(Some(Chunk::new(start, end)));
      }
      let frame = match level {
        Level::Paragraph => Frame::Paragraphs {
          cursor: start,
          end,
          done: false,
        },
        Level::Sentence => Frame::Sentences {
          iter: text[start..end].split_sentence_bound_indices(),
          base: start,
        },
        Level::Word => {
          // A range the word level cannot divide at all — no Unicode word in it —
          // goes straight to the `char` fallback, as it did when the word ranges
          // were collected and found empty.
          let mut iter: Peekable<UnicodeWordIndices<'a>> =
            text[start..end].unicode_word_indices().peekable();
          if iter.peek().is_none() {
            Frame::Chars {
              seg_start: start,
              cursor: start,
              end,
            }
          } else {
            Frame::Words { iter, base: start }
          }
        }
      };
      try_push(&mut self.stack, frame)?;
      Ok(None)
    }

    /// Take one Unicode word whole when it fits the window, and otherwise
    /// suspend the `char`-aligned fallback over it.
    fn word(&mut self, start: usize, end: usize) -> Result<Option<Chunk>, WinditError> {
      if self
        .measurer
        .measure_within(&self.text[start..end], self.window)
        .is_some()
      {
        return Ok(Some(Chunk::new(start, end)));
      }
      try_push(
        &mut self.stack,
        Frame::Chars {
          seg_start: start,
          cursor: start,
          end,
        },
      )?;
      Ok(None)
    }
  }

  /// Random access over the atoms of the chunk being packed.
  ///
  /// The packer reads atoms by absolute index and probes backwards within the
  /// chunk it has just closed, so that block must stay addressable; everything
  /// before it is dropped as the packing advances. Peak occupancy is therefore
  /// one chunk's worth of atoms rather than the whole input's — and one chunk's
  /// worth is what producing even a single chunk costs.
  struct AtomWindow<'a> {
    atoms: Atoms<'a>,
    buf: VecDeque<Chunk>,
    /// The absolute index of `buf`'s front.
    base: usize,
  }

  impl<'a> AtomWindow<'a> {
    fn new(text: &'a str, window: usize, measurer: &'a dyn MeasureText) -> Self {
      Self {
        atoms: Atoms::new(text, window, measurer),
        buf: VecDeque::new(),
        base: 0,
      }
    }

    /// The atom at absolute index `idx`, produced if it does not exist yet, or
    /// `None` once the text has no atom there.
    ///
    /// Production stops at the first atom the caller does not ask for, which is
    /// what keeps a capped packing from splitting text past its last chunk.
    fn get(&mut self, idx: usize) -> Result<Option<Chunk>, WinditError> {
      while self.base + self.buf.len() <= idx {
        let Some(atom) = self.atoms.next()? else {
          return Ok(None);
        };
        self
          .buf
          .try_reserve(1)
          .map_err(|_| WinditError::AllocFailed {
            elements: self.buf.len().saturating_add(1),
          })?;
        self.buf.push_back(atom);
      }
      Ok(Some(self.buf[idx - self.base]))
    }

    /// The atom at absolute index `idx`, which must already have been produced
    /// by [`get`](AtomWindow::get) and not yet discarded.
    fn buffered(&self, idx: usize) -> Chunk {
      self.buf[idx - self.base]
    }

    /// Drop every atom below absolute index `idx`.
    fn discard_before(&mut self, idx: usize) {
      // `idx` is the next chunk's first atom, always at least one past the
      // current chunk's, so the subtraction is exact. Saturating keeps a future
      // caller that moved it backwards from wrapping into a drain of everything
      // buffered, and advancing `base` by what was dropped keeps the two
      // consistent whatever it was handed.
      let dropped = idx.saturating_sub(self.base);
      self.buf.drain(..dropped);
      self.base += dropped;
    }
  }

  /// Push `value`, growing `out` fallibly.
  ///
  /// Neither the chunk count nor the atom count is known before packing runs, so
  /// neither collection can be sized up front the way [`WindowPlan::spans`] sizes
  /// its plan. `try_reserve` grows them amortized all the same, and — unlike
  /// `push` — reports an allocator that refuses instead of aborting a call whose
  /// signature promises a `Result`.
  fn try_push<T>(out: &mut Vec<T>, value: T) -> Result<(), WinditError> {
    out.try_reserve(1).map_err(|_| WinditError::AllocFailed {
      elements: out.len().saturating_add(1),
    })?;
    out.push(value);
    Ok(())
  }

  /// Greedily pack `text`'s atoms into chunks no longer than the window,
  /// repeating at most `opts.overlap()` tokens' worth of trailing whole atoms
  /// between consecutive chunks (the overlap is a token budget measured over the
  /// exact repeated text, not an atom count).
  ///
  /// Atoms arrive in input order, each already known to fit the window on its
  /// own, and are produced only as this loop asks for them. Asking is what
  /// splits the text, so the cap below bounds the tokenization rather than
  /// merely reporting on it: the loop stops at the first chunk past the cap, and
  /// no atom beyond that chunk is ever built.
  ///
  /// Every boundary emitted here is decided by measuring the exact contiguous
  /// text it delimits, never by adding up per-atom measurements: a BPE or
  /// wordpiece tokenizer is not additive, so measuring `"a b"` is not generally
  /// measuring `"a"` plus measuring `"b"`, and a per-atom cache would silently
  /// move chunk boundaries. Each threshold check is a
  /// [`MeasureText::measure_within`] over that exact range, so the measurer may
  /// stop early without changing which boundary is chosen — the answer turns
  /// only on whether the range fits, which is what `measure_within` reports.
  /// What the two searches drop is only measurement that cannot change an answer
  /// — a range whose measure is already known, and interior positions of a
  /// bracket the search is narrowing.
  fn pack(
    text: &str,
    opts: &WindowOptions,
    measurer: &dyn MeasureText,
  ) -> Result<Vec<Chunk>, WinditError> {
    let (window, overlap) = (opts.window(), opts.overlap());
    let mut atoms = AtomWindow::new(text, window, measurer);
    let mut chunks = Vec::new();
    let mut i = 0usize;
    // One atom past the end of the block the previous chunk's overlap probe
    // measured against the current `i`, or `0` when nothing was repeated. That
    // probe measured `text[atoms[i].start()..atoms[carried - 1].end()]` at no
    // more than the overlap, and the overlap is strictly below the window, so
    // the block is already known to fit and the forward walk resumes past it
    // instead of re-measuring every prefix inside it. The measure is a function
    // of the text alone, so re-measuring that identical range could only return
    // the same answer.
    let mut carried = 0usize;
    while let Some(first) = atoms.get(i)? {
      let chunk_start = first.start();
      let mut j = if carried > i + 1 { carried } else { i + 1 };
      let mut chunk_end = atoms.buffered(j - 1).end();
      let exhausted = loop {
        let Some(atom) = atoms.get(j)? else {
          break true;
        };
        let candidate_end = atom.end();
        if measurer
          .measure_within(&text[chunk_start..candidate_end], window)
          .is_none()
        {
          break false;
        }
        chunk_end = candidate_end;
        j += 1;
      };
      try_push(&mut chunks, Chunk::new(chunk_start, chunk_end))?;
      if let Some(max) = opts.max_windows() {
        if chunks.len() > max {
          return Err(WinditError::TooManyWindows {
            got: chunks.len(),
            max,
          });
        }
      }
      if exhausted {
        break;
      }
      // Back up for the next chunk by TOKEN budget, not atom count: repeat as many
      // trailing whole atoms as fit in the overlap, but always advance at least
      // one atom so packing terminates. `i + 1` is that floor; `j` repeats nothing.
      let next = first_accepted(i + 1, j, |t| {
        measurer
          .measure_within(&text[atoms.buffered(t).start()..chunk_end], overlap)
          .is_some()
      });
      carried = if next < j { j } else { 0 };
      atoms.discard_before(next);
      i = next;
    }
    Ok(chunks)
  }

  /// The smallest `t` in `lo..=hi` that `accepts` admits, taking `hi` — the
  /// "repeat nothing" sentinel — as admitted by definition.
  ///
  /// The scan is a linear walk from `lo` upward that returns the first admitted
  /// `t`, because `accepts` is not monotonic in `t`: a context-sensitive tokenizer
  /// (BPE, wordpiece) can measure a longer trailing suffix as *fewer* tokens than
  /// a shorter one, so admission can toggle off and back on as the suffix shortens.
  /// A doubling-and-bisection search assumes a single reject-to-admit flip and can
  /// bracket straight past the earliest admitted `t`, returning a later suffix — or
  /// the sentinel — and silently dropping configured overlap. A linear walk finds
  /// the earliest admitted `t` for an arbitrary `accepts`.
  ///
  /// It stays bounded: `lo..hi` spans only the just-closed chunk's trailing atoms,
  /// each `accepts` is a `measure_within` capped at the overlap budget, and across
  /// the whole packing successive overlap searches scan adjacent atom ranges that
  /// tile the atom stream — so the walks cost `O(a)` in total, not per chunk.
  /// `accepts` is consulted only at real candidates, never at the `hi` sentinel.
  fn first_accepted(lo: usize, hi: usize, accepts: impl Fn(usize) -> bool) -> usize {
    (lo..hi).find(|&t| accepts(t)).unwrap_or(hi)
  }
};
