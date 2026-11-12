//! windit's own reader of a struct's keys.
//!
//! [`Keys`] reads each key to the index of the field it names, through a
//! visitor whose expected text names the fields, so the format makes a
//! refusal of a key at the key: an unknown name and a repeated field in
//! serde's own words, and a key of any other shape, such as YAML's
//! `? [window]`, named by the fields it could have been. A key is read as
//! serde's derive reads a field identifier: by its name as text or bytes, or
//! by the field's index.

use core::fmt;

use serde::de::{DeserializeSeed, Deserializer, Expected, MapAccess, Unexpected, Visitor};

/// The fields a key of `owner`'s may name: what a key expects.
#[derive(Clone, Copy)]
struct FieldList {
  owner: &'static str,
  fields: &'static [&'static str],
}

impl Expected for FieldList {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "a field of {}: ", self.owner)?;
    for (at, field) in self.fields.iter().enumerate() {
      if at > 0 {
        f.write_str(", ")?;
      }
      write!(f, "`{field}`")?;
    }
    Ok(())
  }
}

/// A struct's keys, read by windit.
pub(crate) struct Keys {
  list: FieldList,
  /// The fields already read, one bit each.
  seen: u64,
}

impl Keys {
  /// `owner`'s keys, whose fields are `fields`, at most 64 of them.
  pub(crate) const fn new(owner: &'static str, fields: &'static [&'static str]) -> Self {
    Self {
      list: FieldList { owner, fields },
      seen: 0,
    }
  }

  /// The index of the field the map's next key names, or `None` at the
  /// map's end.
  pub(crate) fn next<'de, A: MapAccess<'de>>(
    &mut self,
    map: &mut A,
  ) -> Result<Option<usize>, A::Error> {
    map.next_key_seed(Key {
      list: self.list,
      seen: &mut self.seen,
    })
  }
}

/// One key.
struct Key<'a> {
  list: FieldList,
  seen: &'a mut u64,
}

impl Key<'_> {
  /// The field at `at`, refused where it was read already.
  fn first<E: serde::de::Error>(self, at: usize) -> Result<usize, E> {
    let bit = 1u64 << at;
    if *self.seen & bit != 0 {
      return Err(E::duplicate_field(self.list.fields[at]));
    }
    *self.seen |= bit;
    Ok(at)
  }

  /// The field `name` names, refused where it names none.
  fn named<E: serde::de::Error>(self, name: &[u8]) -> Result<usize, E> {
    match self
      .list
      .fields
      .iter()
      .position(|field| field.as_bytes() == name)
    {
      Some(at) => self.first(at),
      None => Err(match core::str::from_utf8(name) {
        Ok(text) => E::unknown_field(text, self.list.fields),
        Err(_) => E::invalid_value(Unexpected::Bytes(name), &self.list),
      }),
    }
  }
}

impl<'de> DeserializeSeed<'de> for Key<'_> {
  type Value = usize;

  fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<usize, D::Error> {
    deserializer.deserialize_identifier(self)
  }
}

impl Visitor<'_> for Key<'_> {
  type Value = usize;

  fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    Expected::fmt(&self.list, f)
  }

  fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<usize, E> {
    match usize::try_from(v)
      .ok()
      .filter(|at| *at < self.list.fields.len())
    {
      Some(at) => self.first(at),
      None => Err(E::invalid_value(Unexpected::Unsigned(v), &self.list)),
    }
  }

  fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<usize, E> {
    self.named(v.as_bytes())
  }

  fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<usize, E> {
    self.named(v)
  }
}
