//! Split policies: decide how an input is divided before windowing.
//!
//! `SplitPolicy` maps an input length and `WindowOptions` to the `Span`s
//! that cover it. `FixedWindow` is the mechanical element-count split — it
//! delegates to `WindowPlan::spans`, so any unit (samples, tokens, patches)
//! is divided the same way.
//!
//! `ContentAware` (feature `text`) is the tokenizer-free string chunker: it
//! packs text into chunks that respect paragraph, sentence, and word boundaries,
//! measuring length through a caller-supplied `len_fn` so the caller's own
//! tokenizer defines "how long". It is a separate surface from `SplitPolicy`
//! because it returns `Chunk`s — half-open UTF-8 byte ranges — rather than
//! element `Span`s.

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

/// A tokenizer-free, content-aware string chunker (feature `text`).
///
/// [`chunk`](ContentAware::chunk) splits text on recursive boundaries —
/// paragraphs (`\n\n`), then sentences, then words — and greedily packs the
/// pieces into chunks no longer than [`WindowOptions::window`], as measured by
/// the caller's `len_fn`. Because length is whatever `len_fn` returns, the caller
/// supplies its own notion of a token (word count, a tokenizer's id count, code
/// points) without this crate depending on a tokenizer.
#[cfg(feature = "text")]
#[derive(Clone, Copy)]
pub struct ContentAware<'a> {
  len_fn: &'a dyn Fn(&str) -> usize,
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
    /// A chunker that measures text length with `len_fn`.
    ///
    /// [`chunk`](ContentAware::chunk) guarantees every returned chunk measures
    /// at most the window under this function, save for a single `char` that
    /// alone exceeds the window (it cannot be split further).
    #[must_use]
    pub const fn new(len_fn: &'a dyn Fn(&str) -> usize) -> Self {
      Self { len_fn }
    }

    /// The caller-supplied length function, in whose units the window is
    /// measured.
    // No `#[must_use]`: the `Fn` trait object already carries one.
    pub const fn len_fn(&self) -> &'a dyn Fn(&str) -> usize {
      self.len_fn
    }

    /// Chunk `text` into [`Chunk`]s, each measuring at most
    /// [`WindowOptions::window`] under `len_fn`.
    ///
    /// Each returned [`Chunk`] falls on `char` boundaries, so
    /// [`Chunk::as_str`] applied to this same `text` never returns `None`.
    /// Boundaries are preferred coarse-to-fine: a paragraph, sentence, or word
    /// that fits is kept whole, and only a unit that overflows the window on
    /// its own is split further — a long sentence into words, a long word
    /// into `len_fn`-measured character slices. Chunks are packed greedily;
    /// when [`WindowOptions::overlap`] is non-zero, consecutive chunks repeat
    /// at most that many trailing tokens' worth of whole boundary units (as
    /// measured by `len_fn`).
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
    /// `len_fn` is called `O(a log a)` times for `a` atoms, each call measuring
    /// at most one window's worth of text: the packing never re-measures a range
    /// whose measure it already knows, and it locates each overlap boundary by
    /// probing rather than by walking one atom at a time.
    ///
    /// `a` counts the atoms the emitted chunks are built from, not the atoms the
    /// whole text contains. Atoms are produced on demand and packed as they are
    /// produced, so [`WindowOptions::max_windows`] bounds the tokenization and
    /// the memory as well as the chunk count: a capped chunking stops at the
    /// first chunk past the cap and never splits the text beyond it. Peak memory
    /// is one chunk's worth of atoms — the block the overlap search probes
    /// backwards through — plus the chunks emitted so far.
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
      pack(text, opts, self.len_fn)
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
    /// The `Chars` arm is the crate's other `len_fn` walk over a growing
    /// substring, and it is left as a walk deliberately. Each slice is measured
    /// from its own start, so the measurement resets at every emitted boundary:
    /// the cost is the range's length times one window, not times itself, which
    /// is the same shape the packing bound has. Probing instead would need
    /// `len_fn` to be monotone over a growing prefix, and a BPE tokenizer is not.
    fn advance(&mut self, text: &str, window: usize, len_fn: &dyn Fn(&str) -> usize) -> Step {
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
            if abs > *seg_start && len_fn(&text[*seg_start..next]) > window {
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
  /// `len_fn` call that each visit makes. An uncapped chunking therefore returns
  /// the same chunks at the same cost as the recursion did.
  struct Atoms<'a> {
    text: &'a str,
    window: usize,
    len_fn: &'a dyn Fn(&str) -> usize,
    /// The whole-input descent, held until the first `next` so that construction
    /// stays infallible.
    root: Option<(usize, usize)>,
    stack: Vec<Frame<'a>>,
  }

  impl<'a> Atoms<'a> {
    fn new(text: &'a str, window: usize, len_fn: &'a dyn Fn(&str) -> usize) -> Self {
      Self {
        text,
        window,
        len_fn,
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
      let (text, window, len_fn) = (self.text, self.window, self.len_fn);
      loop {
        let step = match self.stack.last_mut() {
          Some(frame) => frame.advance(text, window, len_fn),
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
      if (self.len_fn)(&text[start..end]) <= self.window {
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
      if (self.len_fn)(&self.text[start..end]) <= self.window {
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
    fn new(text: &'a str, window: usize, len_fn: &'a dyn Fn(&str) -> usize) -> Self {
      Self {
        atoms: Atoms::new(text, window, len_fn),
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
  /// between consecutive chunks (the overlap is a token budget measured by
  /// `len_fn`, not an atom count).
  ///
  /// Atoms arrive in input order, each already known to fit the window on its
  /// own, and are produced only as this loop asks for them. Asking is what
  /// splits the text, so the cap below bounds the tokenization rather than
  /// merely reporting on it: the loop stops at the first chunk past the cap, and
  /// no atom beyond that chunk is ever built.
  ///
  /// Every boundary emitted here is decided by a `len_fn` measurement of the
  /// exact contiguous text it delimits, never by adding up per-atom measurements:
  /// a BPE or wordpiece tokenizer is not additive, so `len_fn("a b")` is not
  /// generally `len_fn("a") + len_fn("b")` and a per-atom cache would silently
  /// move chunk boundaries. What the two searches drop is only measurement that
  /// cannot change an answer — a range whose measure is already known, and
  /// interior positions of a bracket the search is narrowing.
  fn pack(
    text: &str,
    opts: &WindowOptions,
    len_fn: &dyn Fn(&str) -> usize,
  ) -> Result<Vec<Chunk>, WinditError> {
    let (window, overlap) = (opts.window(), opts.overlap());
    let mut atoms = AtomWindow::new(text, window, len_fn);
    let mut chunks = Vec::new();
    let mut i = 0usize;
    // One atom past the end of the block the previous chunk's overlap probe
    // measured against the current `i`, or `0` when nothing was repeated. That
    // probe measured `text[atoms[i].start()..atoms[carried - 1].end()]` at no
    // more than the overlap, and the overlap is strictly below the window, so
    // the block is already known to fit and the forward walk resumes past it
    // instead of re-measuring every prefix inside it. `len_fn` is an `Fn` of
    // the text alone, so re-measuring that identical range could only return
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
        if len_fn(&text[chunk_start..candidate_end]) > window {
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
        len_fn(&text[atoms.buffered(t).start()..chunk_end]) <= overlap
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
  /// Probes outward from `lo` in doubling steps, then bisects the bracket that
  /// establishes. A budget wide enough to repeat the whole chunk therefore costs
  /// one measurement and a budget that repeats a few trailing atoms costs a few,
  /// where walking one atom at a time cost one measurement per repeated atom —
  /// each over a substring that grew with it, which is what made packing at a
  /// near-window overlap quadratic per chunk.
  ///
  /// `accepts` is consulted only at real candidates, and the returned `t` is
  /// either `hi` or a candidate it admitted, so the block finally repeated has
  /// always been measured directly however the search reached it.
  fn first_accepted(lo: usize, hi: usize, accepts: impl Fn(usize) -> bool) -> usize {
    let (mut low, mut high) = (lo, hi);
    let mut step = 1usize;
    while low < high {
      let probe = low.saturating_add(step - 1);
      if probe >= high {
        break;
      }
      if accepts(probe) {
        high = probe;
        break;
      }
      low = probe + 1;
      step = step.saturating_mul(2);
    }
    while low < high {
      let mid = low + (high - low) / 2;
      if accepts(mid) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    low
  }
};
