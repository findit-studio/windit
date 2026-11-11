//! The one rule windit's human-readable readers hold a unit variant to.
//!
//! In JSON, YAML and TOML a unit variant is read as its name, or as a YAML tag
//! on no value. A one-entry map is how a variant carrying a value is written,
//! so a unit variant written as one is refused by name, whatever its value,
//! `null` included. `AggregatePolicyKind` and `TailPolicy`'s `kind` both read
//! by this rule: they refuse the map in [`unit_given_a_value`]'s words, and
//! read a tag's value through [`UnitTag`].
//!
//! Every refusal here is made inside a visitor the format called, so the
//! format locates it at the value itself; none is built around another error.

use core::fmt;

use serde::de::{DeserializeSeed, Deserializer, Visitor};

/// A unit variant written as a one-entry map, refused whatever its value.
pub(crate) fn unit_given_a_value<E: serde::de::Error>(name: &str) -> E {
  E::custom(format_args!(
    "`{name}` takes no value, and the document gives it one"
  ))
}

/// The value under a unit variant's YAML tag, `!pad_full`: none. Read as it
/// stands, so a value there is refused by this reader, its expected text
/// naming the variant, at the value's own location.
#[derive(Clone, Copy)]
pub(crate) struct UnitTag(pub(crate) &'static str);

impl<'de> DeserializeSeed<'de> for UnitTag {
  type Value = ();

  fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
    deserializer.deserialize_any(self)
  }
}

impl Visitor<'_> for UnitTag {
  type Value = ();

  fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "no value under `{}`, which takes none", self.0)
  }

  fn visit_unit<E: serde::de::Error>(self) -> Result<(), E> {
    Ok(())
  }

  fn visit_none<E: serde::de::Error>(self) -> Result<(), E> {
    Ok(())
  }
}
