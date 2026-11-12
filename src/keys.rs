//! windit's own reader of a derived struct's keys.
//!
//! A derived reader reads each key through serde's field-identifier reader,
//! whose refusal of a key that is not a name, such as YAML's `? [window]`,
//! expects a "field identifier" and names nothing of windit's. [`Fields`] reads
//! each key itself instead, through a visitor whose expected text names the
//! struct's fields, so the format makes that refusal at the key, named. A
//! repeated field and an unknown name are refused there too, at the key, in
//! serde's own words. Every key it takes is handed on to the derived reader in
//! the form it arrived in: text, bytes, or a field's index.

use core::fmt;

use serde::de::{
  value::{BorrowedBytesDeserializer, BorrowedStrDeserializer, BytesDeserializer},
  DeserializeSeed, Deserializer, Expected, IntoDeserializer, MapAccess, Unexpected, Visitor,
};

/// The fields a key of `owner`'s may name: what a [`Fields`] key expects.
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

/// A derived reader's map, its keys read by windit.
pub(crate) struct Fields<A> {
  map: A,
  list: FieldList,
  /// The fields already read, one bit each.
  seen: u64,
}

impl<A> Fields<A> {
  /// `owner`'s map, whose fields are `fields`, at most 64 of them.
  pub(crate) const fn new(map: A, owner: &'static str, fields: &'static [&'static str]) -> Self {
    Self {
      map,
      list: FieldList { owner, fields },
      seen: 0,
    }
  }
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for Fields<A> {
  type Error = A::Error;

  fn next_key_seed<K: DeserializeSeed<'de>>(
    &mut self,
    seed: K,
  ) -> Result<Option<K::Value>, A::Error> {
    self.map.next_key_seed(FieldKey {
      seed,
      list: self.list,
      seen: &mut self.seen,
    })
  }

  fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, A::Error> {
    self.map.next_value_seed(seed)
  }

  fn size_hint(&self) -> Option<usize> {
    self.map.size_hint()
  }
}

/// One key of a [`Fields`] map.
struct FieldKey<'a, K> {
  seed: K,
  list: FieldList,
  seen: &'a mut u64,
}

impl<K> FieldKey<'_, K> {
  /// The field at `at`, refused where it was read already.
  fn first<E: serde::de::Error>(&mut self, at: usize) -> Result<(), E> {
    let bit = 1u64 << at;
    if *self.seen & bit != 0 {
      return Err(E::duplicate_field(self.list.fields[at]));
    }
    *self.seen |= bit;
    Ok(())
  }

  /// The index of the field `name` names, refused where it names none.
  fn named<E: serde::de::Error>(&self, name: &[u8]) -> Result<usize, E> {
    match self
      .list
      .fields
      .iter()
      .position(|field| field.as_bytes() == name)
    {
      Some(at) => Ok(at),
      None => Err(match core::str::from_utf8(name) {
        Ok(text) => E::unknown_field(text, self.list.fields),
        Err(_) => E::invalid_value(Unexpected::Bytes(name), &self.list),
      }),
    }
  }
}

impl<'de, K: DeserializeSeed<'de>> DeserializeSeed<'de> for FieldKey<'_, K> {
  type Value = K::Value;

  fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<K::Value, D::Error> {
    deserializer.deserialize_identifier(self)
  }
}

impl<'de, K: DeserializeSeed<'de>> Visitor<'de> for FieldKey<'_, K> {
  type Value = K::Value;

  fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    Expected::fmt(&self.list, f)
  }

  fn visit_u64<E: serde::de::Error>(mut self, v: u64) -> Result<K::Value, E> {
    let at = usize::try_from(v)
      .ok()
      .filter(|at| *at < self.list.fields.len())
      .ok_or_else(|| E::invalid_value(Unexpected::Unsigned(v), &self.list))?;
    self.first(at)?;
    self.seed.deserialize(v.into_deserializer())
  }

  fn visit_str<E: serde::de::Error>(mut self, v: &str) -> Result<K::Value, E> {
    let at = self.named(v.as_bytes())?;
    self.first(at)?;
    self.seed.deserialize(v.into_deserializer())
  }

  fn visit_borrowed_str<E: serde::de::Error>(mut self, v: &'de str) -> Result<K::Value, E> {
    let at = self.named(v.as_bytes())?;
    self.first(at)?;
    self.seed.deserialize(BorrowedStrDeserializer::new(v))
  }

  fn visit_bytes<E: serde::de::Error>(mut self, v: &[u8]) -> Result<K::Value, E> {
    let at = self.named(v)?;
    self.first(at)?;
    self.seed.deserialize(BytesDeserializer::new(v))
  }

  fn visit_borrowed_bytes<E: serde::de::Error>(mut self, v: &'de [u8]) -> Result<K::Value, E> {
    let at = self.named(v)?;
    self.first(at)?;
    self.seed.deserialize(BorrowedBytesDeserializer::new(v))
  }
}
