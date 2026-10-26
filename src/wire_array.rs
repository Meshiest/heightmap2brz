//! Typed `ArrayVar` contents.
//!
//! An `ArrayVar`'s `Value` is a `dynamic_array`: an element type from the
//! save's `types` table, then the elements. [`WireArray`] holds the element
//! types this crate writes, keeping each in its native Rust form so a large
//! frame bank is written straight from its `Vec` without conversion, and
//! decodes a saved array back for the tests.

use brdb::{
    BrdbSchemaError,
    schema::{
        BrdbValue, SavedBrickRef, SavedComponentRef,
        as_brdb::{AsBrdbValue, BrdbArrayIter},
    },
};

/// One `ArrayVar` value. Tuple elements encode as the schema struct of the
/// same field order: `LinearColor` is `(R, G, B, A)` (linear 0-1) and `Quat`
/// is `(X, Y, Z, W)`.
#[derive(Clone, Debug, PartialEq)]
pub enum WireArray {
    Str(Vec<String>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    LinearColor(Vec<(f32, f32, f32, f32)>),
    Quat(Vec<(f64, f64, f64, f64)>),
    /// `brick_handle` elements; see [`crate::graph::main_grid_brick_refs`]
    /// for the form that resolves.
    Bricks(Vec<SavedBrickRef>),
    /// `component_handle` elements; refs to other `ArrayVar`s make an array
    /// of arrays.
    Components(Vec<SavedComponentRef>),
}

impl WireArray {
    /// The `types` table entry for this array's elements.
    pub fn element_type(&self) -> &'static str {
        match self {
            WireArray::Str(_) => "str",
            WireArray::F64(_) => "f64",
            WireArray::I64(_) => "i64",
            WireArray::LinearColor(_) => "LinearColor",
            WireArray::Quat(_) => "Quat",
            WireArray::Bricks(_) => "brick_handle",
            WireArray::Components(_) => "component_handle",
        }
    }

    pub fn len(&self) -> usize {
        match self {
            WireArray::Str(v) => v.len(),
            WireArray::F64(v) => v.len(),
            WireArray::I64(v) => v.len(),
            WireArray::LinearColor(v) => v.len(),
            WireArray::Quat(v) => v.len(),
            WireArray::Bricks(v) => v.len(),
            WireArray::Components(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Roughly how many bytes the save spends on the elements before
    /// compression. Only used to keep a chunk's array data under the `.brz`
    /// blob limit, so it errs high.
    pub fn approx_bytes(&self) -> usize {
        match self {
            WireArray::Str(v) => v.iter().map(|s| s.len() + 5).sum(),
            WireArray::F64(v) => v.len() * 9,
            WireArray::I64(v) => v.len() * 9,
            WireArray::LinearColor(v) => v.len() * 20,
            WireArray::Quat(v) => v.len() * 36,
            WireArray::Bricks(v) => v.len() * 16,
            WireArray::Components(v) => v.len() * 20,
        }
    }
}

fn iter<T: AsBrdbValue>(v: &[T]) -> BrdbArrayIter<'_> {
    Box::new(v.iter().map(|x| x as &dyn AsBrdbValue))
}

impl AsBrdbValue for WireArray {
    fn as_brdb_dynamic_array(&self) -> Result<(&str, BrdbArrayIter<'_>), BrdbSchemaError> {
        let items = match self {
            WireArray::Str(v) => iter(v),
            WireArray::F64(v) => iter(v),
            WireArray::I64(v) => iter(v),
            WireArray::LinearColor(v) => iter(v),
            WireArray::Quat(v) => iter(v),
            WireArray::Bricks(v) => iter(v),
            WireArray::Components(v) => iter(v),
        };
        Ok((self.element_type(), items))
    }
}

/// Decode a saved `ArrayVar` `Value`.
impl TryFrom<&BrdbValue> for WireArray {
    type Error = BrdbSchemaError;

    fn try_from(value: &BrdbValue) -> Result<Self, Self::Error> {
        let BrdbValue::DynamicArray(array) = value else {
            return Err(BrdbSchemaError::ExpectedType(
                "dynamic_array".into(),
                value.get_type().to_string(),
            ));
        };
        let values = &array.values;
        let f32_of = |v: &BrdbValue, p: &str| v.prop(p)?.as_brdb_f32();
        let f64_of = |v: &BrdbValue, p: &str| v.prop(p)?.as_brdb_f64();
        Ok(match array.element_type.to_ascii_lowercase().as_str() {
            "str" => WireArray::Str(
                values
                    .iter()
                    .map(|v| v.as_str().map(str::to_owned))
                    .collect::<Result<_, _>>()?,
            ),
            "f64" => WireArray::F64(
                values
                    .iter()
                    .map(|v| v.as_brdb_f64())
                    .collect::<Result<_, _>>()?,
            ),
            "i64" => WireArray::I64(
                values
                    .iter()
                    .map(|v| v.as_brdb_i64())
                    .collect::<Result<_, _>>()?,
            ),
            "linearcolor" => WireArray::LinearColor(
                values
                    .iter()
                    .map(|v| {
                        Ok((
                            f32_of(v, "R")?,
                            f32_of(v, "G")?,
                            f32_of(v, "B")?,
                            f32_of(v, "A")?,
                        ))
                    })
                    .collect::<Result<_, BrdbSchemaError>>()?,
            ),
            "quat" => WireArray::Quat(
                values
                    .iter()
                    .map(|v| {
                        Ok((
                            f64_of(v, "X")?,
                            f64_of(v, "Y")?,
                            f64_of(v, "Z")?,
                            f64_of(v, "W")?,
                        ))
                    })
                    .collect::<Result<_, BrdbSchemaError>>()?,
            ),
            "brick_handle" => WireArray::Bricks(
                values
                    .iter()
                    .map(SavedBrickRef::try_from)
                    .collect::<Result<_, _>>()?,
            ),
            "component_handle" => WireArray::Components(
                values
                    .iter()
                    .map(SavedComponentRef::try_from)
                    .collect::<Result<_, _>>()?,
            ),
            other => {
                return Err(BrdbSchemaError::ExpectedType(
                    "a WireArray element type".into(),
                    other.to_owned(),
                ));
            }
        })
    }
}
