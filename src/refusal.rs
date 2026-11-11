//! How windit's readers name what they refuse, shared so that every wire type
//! refuses alike.
//!
//! In JSON, YAML and TOML a unit variant is read as its name, or as a YAML tag
//! on no value. A one-entry map is how a variant carrying a value is written,
//! so a unit variant written as one is refused by name, whatever its value,
//! `null` included. `AggregatePolicyKind` and `TailPolicy`'s `kind` both read
//! by this rule, and refuse in [`unit_given_a_value`]'s words.
//!
//! A value refused inside a field or a variant is named once, where it is
//! met: `` `window`: … ``, `` `drop_below_min`'s `value`: … ``. [`Fields`] and
//! [`Elements`] name the values a derived reader refuses in a map and in a
//! sequence, each through [`Named`], the seed the value is read with: the name
//! is put on the refusal before the format sees it, so a format that marks a
//! refused value with its own span (TOML) still marks the value. A field
//! whose own reader names its refusals is left unnamed, so that a refusal is
//! not named twice and TOML keeps marking the value inside it.

use core::fmt;

use serde::de::{
  value::{BorrowedBytesDeserializer, BorrowedStrDeserializer, BytesDeserializer},
  DeserializeSeed, Deserializer, IntoDeserializer, MapAccess, SeqAccess, Visitor,
};

/// A unit variant written as a one-entry map, refused whatever its value.
pub(crate) fn unit_given_a_value<E: serde::de::Error>(name: &str) -> E {
  E::custom(format_args!(
    "`{name}` takes no value, and the document gives it one"
  ))
}

/// A refusal met inside `name`, a variant or a field, named by it.
pub(crate) fn within<E: serde::de::Error>(name: &str, refusal: impl fmt::Display) -> E {
  E::custom(format_args!("`{name}`: {refusal}"))
}

/// A refusal met inside `owner`'s `field`, named by both.
pub(crate) fn within_field<E: serde::de::Error>(
  owner: &str,
  field: &str,
  refusal: impl fmt::Display,
) -> E {
  E::custom(format_args!("`{owner}`'s `{field}`: {refusal}"))
}

/// A seed whose refusal is named by `field`, and by `owner` where one is
/// given.
pub(crate) struct Named<S> {
  pub(crate) seed: S,
  pub(crate) owner: Option<&'static str>,
  pub(crate) field: &'static str,
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for Named<S> {
  type Value = S::Value;

  fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<S::Value, D::Error> {
    let (owner, field) = (self.owner, self.field);
    self
      .seed
      .deserialize(deserializer)
      .map_err(|refusal| match owner {
        Some(owner) => within_field(owner, field, refusal),
        None => within(field, refusal),
      })
  }
}

/// A derived reader's map, each value it refuses named by its field, and by
/// `owner` where one is given.
///
/// Each key is matched against `fields`, the reader's own field list, on its
/// way to the reader's own seed, and handed on in the form it arrived in:
/// text, bytes, or a field's index. A key that names no field passes on
/// unchanged, for the reader to refuse by name. A field in `nested` names its
/// own refusals and is left unnamed here.
pub(crate) struct Fields<A> {
  map: A,
  owner: Option<&'static str>,
  fields: &'static [&'static str],
  nested: &'static [&'static str],
  /// The field the key just read named, whose value comes next.
  current: Option<&'static str>,
}

impl<A> Fields<A> {
  pub(crate) fn new(
    map: A,
    owner: Option<&'static str>,
    fields: &'static [&'static str],
    nested: &'static [&'static str],
  ) -> Self {
    Self {
      map,
      owner,
      fields,
      nested,
      current: None,
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
      fields: self.fields,
      nested: self.nested,
      current: &mut self.current,
    })
  }

  fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, A::Error> {
    match self.current.take() {
      Some(field) => self.map.next_value_seed(Named {
        seed,
        owner: self.owner,
        field,
      }),
      None => self.map.next_value_seed(seed),
    }
  }

  fn size_hint(&self) -> Option<usize> {
    self.map.size_hint()
  }
}

/// One key of a [`Fields`] map: noted, then handed on as it arrived.
struct FieldKey<'a, K> {
  seed: K,
  fields: &'static [&'static str],
  nested: &'static [&'static str],
  current: &'a mut Option<&'static str>,
}

impl<K> FieldKey<'_, K> {
  /// The field `name` names, if it is one this map names.
  fn field(&self, name: &[u8]) -> Option<&'static str> {
    self
      .fields
      .iter()
      .copied()
      .find(|field| field.as_bytes() == name)
      .filter(|field| !self.nested.contains(field))
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
    write!(f, "one of {:?}", self.fields)
  }

  fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<K::Value, E> {
    *self.current = usize::try_from(v)
      .ok()
      .and_then(|at| self.fields.get(at))
      .copied()
      .filter(|field| !self.nested.contains(field));
    self.seed.deserialize(v.into_deserializer())
  }

  fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<K::Value, E> {
    *self.current = self.field(v.as_bytes());
    self.seed.deserialize(v.into_deserializer())
  }

  fn visit_borrowed_str<E: serde::de::Error>(self, v: &'de str) -> Result<K::Value, E> {
    *self.current = self.field(v.as_bytes());
    self.seed.deserialize(BorrowedStrDeserializer::new(v))
  }

  fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<K::Value, E> {
    *self.current = self.field(v);
    self.seed.deserialize(BytesDeserializer::new(v))
  }

  fn visit_borrowed_bytes<E: serde::de::Error>(self, v: &'de [u8]) -> Result<K::Value, E> {
    *self.current = self.field(v);
    self.seed.deserialize(BorrowedBytesDeserializer::new(v))
  }
}

/// A derived reader's sequence, each element it refuses named by the field
/// at its position. A field in `nested` names its own refusals and is left
/// unnamed here.
pub(crate) struct Elements<A> {
  seq: A,
  fields: &'static [&'static str],
  nested: &'static [&'static str],
  at: usize,
}

impl<A> Elements<A> {
  pub(crate) fn new(
    seq: A,
    fields: &'static [&'static str],
    nested: &'static [&'static str],
  ) -> Self {
    Self {
      seq,
      fields,
      nested,
      at: 0,
    }
  }
}

impl<'de, A: SeqAccess<'de>> SeqAccess<'de> for Elements<A> {
  type Error = A::Error;

  fn next_element_seed<T: DeserializeSeed<'de>>(
    &mut self,
    seed: T,
  ) -> Result<Option<T::Value>, A::Error> {
    let field = self
      .fields
      .get(self.at)
      .copied()
      .filter(|field| !self.nested.contains(field));
    self.at = self.at.saturating_add(1);
    match field {
      Some(field) => self.seq.next_element_seed(Named {
        seed,
        owner: None,
        field,
      }),
      None => self.seq.next_element_seed(seed),
    }
  }

  fn size_hint(&self) -> Option<usize> {
    self.seq.size_hint()
  }
}
