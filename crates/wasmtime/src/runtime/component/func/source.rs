//! The single dynamic lowering driver for component values.
//!
//! Every dynamic way of providing a component value to a guest is a
//! [`ValSource`] node, and this module owns the one recursive traversal that
//! lowers a source either to flat core values ([`ValSource::lower`]) or into
//! linear memory ([`ValSource::store`]). [`Val::lower`] and [`Val::store`] are
//! thin shims over the [`ValSource::Val`] leaf, so the dynamic `Func::call`
//! path (and every other dynamic caller) runs on this driver.
//!
//! Today the only source is an owned dynamic [`Val`]; future variants (e.g.
//! pre-encoded canonical-ABI byte images, per-node mixed trees) are intended
//! to slot in as additional variants of [`ValSource`] handled by the same
//! traversal, rather than as parallel lowering implementations.

use crate::ValRaw;
use crate::component::Val;
use crate::component::concurrent;
use crate::component::func::{Lower, LowerContext, ValidatedCabiBytes, desc};
use crate::component::values::ErrorContextAny;
use crate::prelude::*;
use core::mem::MaybeUninit;
use core::slice::IterMut;
use wasmtime_component_util::{DiscriminantSize, FlagsSize};
use wasmtime_environ::component::{
    CanonicalAbiInfo, InterfaceType, TypeEnum, TypeFlags, TypeMap, TypeOption, TypeResult,
    TypeVariant, VariantInfo,
};

/// A single dynamic strategy for producing the bytes/core-values of one
/// component value.
///
/// This is the vocabulary of the dynamic lowering engine: every dynamic way
/// of providing an argument is a `ValSource`, and one type-directed traversal
/// (private to this module) lowers any of them. [`Func::call`] uses the
/// [`Val`](ValSource::Val) strategy for every argument;
/// [`Func::prepare_call`] lets each argument choose its strategy
/// independently, so one call freely mixes them.
///
/// This enum is `#[non_exhaustive]`: further strategies (per-field mixed
/// trees, host-computed sources, …) are anticipated and will be added without
/// a breaking change.
///
/// [`Func::call`]: crate::component::Func::call
/// [`Func::prepare_call`]: crate::component::Func::prepare_call
#[non_exhaustive]
pub enum ValSource<'a> {
    /// A borrowed dynamic value, lowered element by element — the
    /// always-available strategy, compatible with every parameter type.
    Val(&'a Val),

    /// A `list<T>` provided as a run of pre-validated canonical-ABI element
    /// images, copied into guest memory with a single `memcpy`.
    ///
    /// The parameter must be a `list` whose element type is
    /// [`is_cabi_inline`](crate::component::Type::is_cabi_inline) and
    /// structurally equal to the [`ValidatedCabiBytes::ty`] the proof was
    /// minted for. Because validation already happened at
    /// [`ValidatedCabiBytes::checked`], lowering re-checks nothing per
    /// element.
    ListFlat(ValidatedCabiBytes<'a>),

    /// A `record` provided field by field, each field with its own
    /// independently-chosen source.
    ///
    /// This is what lets a single value mix strategies: a record holding a
    /// large `list<f32>` column and a scalar can supply the column as
    /// [`ListFlat`](ValSource::ListFlat) bytes and the scalar as a
    /// [`Val`](ValSource::Val). Sources are given in declaration order and
    /// must match the record's field count.
    Record(Vec<ValSource<'a>>),

    /// A `list<T>` provided element by element, each element with its own
    /// independently-chosen source.
    ///
    /// The elements are lowered into one contiguous guest allocation, exactly
    /// as a [`Val::List`] would be — the guest cannot tell which host-side
    /// representation supplied any element.
    ListElems(Vec<ValSource<'a>>),
}

impl ValSource<'_> {
    /// A short human-readable description of this source's strategy, for
    /// error messages.
    pub(crate) fn desc(&self) -> &'static str {
        match self {
            ValSource::Val(val) => val.desc(),
            ValSource::ListFlat(_) => "flat list bytes",
            ValSource::Record(_) => "per-field record sources",
            ValSource::ListElems(_) => "per-element list sources",
        }
    }

    /// Serialize this source as core Wasm stack values.
    pub(crate) fn lower<T>(
        &self,
        cx: &mut LowerContext<'_, T>,
        ty: InterfaceType,
        dst: &mut IterMut<'_, MaybeUninit<ValRaw>>,
    ) -> Result<()> {
        let val = match self {
            ValSource::Val(val) => val,
            ValSource::ListFlat(vb) => {
                let InterfaceType::List(t) = ty else {
                    bail!(
                        "type mismatch: cannot provide flat list bytes for {}",
                        desc(&ty)
                    );
                };
                let element = cx.types[t].element;
                let (ptr, len) = lower_list_flat(cx, element, vb)?;
                next_mut(dst).write(ValRaw::i64(ptr as i64));
                next_mut(dst).write(ValRaw::i64(len as i64));
                return Ok(());
            }
            ValSource::Record(fields) => {
                let InterfaceType::Record(t) = ty else {
                    bail!(
                        "type mismatch: cannot provide per-field record sources for {}",
                        desc(&ty)
                    );
                };
                let t = &cx.types[t];
                if t.fields.len() != fields.len() {
                    bail!("expected {} fields, got {}", t.fields.len(), fields.len());
                }
                for (source, field) in fields.iter().zip(t.fields.iter()) {
                    source.lower(cx, field.ty, dst)?;
                }
                return Ok(());
            }
            ValSource::ListElems(items) => {
                let InterfaceType::List(t) = ty else {
                    bail!(
                        "type mismatch: cannot provide per-element list sources for {}",
                        desc(&ty)
                    );
                };
                let element = cx.types[t].element;
                let (ptr, len) = lower_list_elems(cx, element, items)?;
                next_mut(dst).write(ValRaw::i64(ptr as i64));
                next_mut(dst).write(ValRaw::i64(len as i64));
                return Ok(());
            }
        };
        match (ty, val) {
            (InterfaceType::Bool, Val::Bool(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::Bool, _) => unexpected(ty, val),
            (InterfaceType::S8, Val::S8(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::S8, _) => unexpected(ty, val),
            (InterfaceType::U8, Val::U8(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::U8, _) => unexpected(ty, val),
            (InterfaceType::S16, Val::S16(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::S16, _) => unexpected(ty, val),
            (InterfaceType::U16, Val::U16(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::U16, _) => unexpected(ty, val),
            (InterfaceType::S32, Val::S32(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::S32, _) => unexpected(ty, val),
            (InterfaceType::U32, Val::U32(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::U32, _) => unexpected(ty, val),
            (InterfaceType::S64, Val::S64(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::S64, _) => unexpected(ty, val),
            (InterfaceType::U64, Val::U64(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::U64, _) => unexpected(ty, val),
            (InterfaceType::Float32, Val::Float32(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::Float32, _) => unexpected(ty, val),
            (InterfaceType::Float64, Val::Float64(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::Float64, _) => unexpected(ty, val),
            (InterfaceType::Char, Val::Char(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::Char, _) => unexpected(ty, val),
            // NB: `lower` on `ResourceAny` does its own type-checking, so skip
            // looking at it here.
            (InterfaceType::Borrow(_) | InterfaceType::Own(_), Val::Resource(value)) => {
                value.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::Borrow(_) | InterfaceType::Own(_), _) => unexpected(ty, val),
            (InterfaceType::String, Val::String(value)) => {
                let my_dst = &mut MaybeUninit::<[ValRaw; 2]>::uninit();
                value.linear_lower_to_flat(cx, ty, my_dst)?;
                let my_dst = unsafe { my_dst.assume_init() };
                next_mut(dst).write(my_dst[0]);
                next_mut(dst).write(my_dst[1]);
                Ok(())
            }
            (InterfaceType::String, _) => unexpected(ty, val),
            (InterfaceType::List(ty), Val::List(values)) => {
                let ty = &cx.types[ty];
                let (ptr, len) = lower_list(cx, ty.element, values)?;
                next_mut(dst).write(ValRaw::i64(ptr as i64));
                next_mut(dst).write(ValRaw::i64(len as i64));
                Ok(())
            }
            (InterfaceType::List(_), _) => unexpected(ty, val),
            (InterfaceType::Map(ty), Val::Map(pairs)) => {
                let map_ty = &cx.types[ty];
                let (ptr, len) = lower_map(cx, map_ty, pairs)?;
                next_mut(dst).write(ValRaw::i64(ptr as i64));
                next_mut(dst).write(ValRaw::i64(len as i64));
                Ok(())
            }
            (InterfaceType::Map(_), _) => unexpected(ty, val),
            (InterfaceType::Record(ty), Val::Record(values)) => {
                let ty = &cx.types[ty];
                if ty.fields.len() != values.len() {
                    bail!("expected {} fields, got {}", ty.fields.len(), values.len());
                }
                for ((name, value), field) in values.iter().zip(ty.fields.iter()) {
                    if *name != field.name {
                        bail!("expected field `{}`, got `{name}`", field.name);
                    }
                    ValSource::Val(value).lower(cx, field.ty, dst)?;
                }
                Ok(())
            }
            (InterfaceType::Record(_), _) => unexpected(ty, val),
            (InterfaceType::Tuple(ty), Val::Tuple(values)) => {
                let ty = &cx.types[ty];
                if ty.types.len() != values.len() {
                    bail!("expected {} types, got {}", ty.types.len(), values.len());
                }
                for (value, ty) in values.iter().zip(ty.types.iter()) {
                    ValSource::Val(value).lower(cx, *ty, dst)?;
                }
                Ok(())
            }
            (InterfaceType::Tuple(_), _) => unexpected(ty, val),
            (InterfaceType::Variant(ty), Val::Variant(n, v)) => {
                GenericVariant::variant(&cx.types[ty], n, v)?.lower(cx, dst)
            }
            (InterfaceType::Variant(_), _) => unexpected(ty, val),
            (InterfaceType::Option(ty), Val::Option(v)) => {
                GenericVariant::option(&cx.types[ty], v).lower(cx, dst)
            }
            (InterfaceType::Option(_), _) => unexpected(ty, val),
            (InterfaceType::Result(ty), Val::Result(v)) => {
                GenericVariant::result(&cx.types[ty], v)?.lower(cx, dst)
            }
            (InterfaceType::Result(_), _) => unexpected(ty, val),
            (InterfaceType::Enum(ty), Val::Enum(discriminant)) => {
                let discriminant = get_enum_discriminant(&cx.types[ty], discriminant)?;
                next_mut(dst).write(ValRaw::u32(discriminant));
                Ok(())
            }
            (InterfaceType::Enum(_), _) => unexpected(ty, val),
            (InterfaceType::Flags(ty), Val::Flags(value)) => {
                let ty = &cx.types[ty];
                let storage = flags_to_storage(ty, value)?;
                for value in storage {
                    next_mut(dst).write(ValRaw::u32(value));
                }
                Ok(())
            }
            (InterfaceType::Flags(_), _) => unexpected(ty, val),
            (InterfaceType::Future(_), Val::Future(f)) => {
                f.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::Future(_), _) => unexpected(ty, val),
            (InterfaceType::Stream(_), Val::Stream(s)) => {
                s.linear_lower_to_flat(cx, ty, next_mut(dst))
            }
            (InterfaceType::Stream(_), _) => unexpected(ty, val),
            (InterfaceType::ErrorContext(_), Val::ErrorContext(ErrorContextAny(rep))) => {
                concurrent::lower_error_context_to_index(*rep, cx, ty)?.linear_lower_to_flat(
                    cx,
                    InterfaceType::U32,
                    next_mut(dst),
                )
            }
            (InterfaceType::ErrorContext(_), _) => unexpected(ty, val),
            (InterfaceType::FixedLengthList(ty), Val::FixedLengthList(values)) => {
                let ty = &cx.types[ty];
                if ty.size as usize != values.len() {
                    bail!("expected vec of size {}, got {}", ty.size, values.len());
                }
                for value in values {
                    ValSource::Val(value).lower(cx, ty.element, dst)?;
                }
                Ok(())
            }
            (InterfaceType::FixedLengthList(_), _) => unexpected(ty, val),
        }
    }

    /// Serialize this source to the heap at the specified memory location.
    pub(crate) fn store<T>(
        &self,
        cx: &mut LowerContext<'_, T>,
        ty: InterfaceType,
        offset: usize,
    ) -> Result<()> {
        debug_assert!(offset % usize::try_from(cx.types.canonical_abi(&ty).align32)? == 0);

        let val = match self {
            ValSource::Val(val) => val,
            ValSource::ListFlat(vb) => {
                let InterfaceType::List(t) = ty else {
                    bail!(
                        "type mismatch: cannot provide flat list bytes for {}",
                        desc(&ty)
                    );
                };
                let element = cx.types[t].element;
                let (ptr, len) = lower_list_flat(cx, element, vb)?;
                // FIXME(#4311): needs memory64 handling
                *cx.get(offset + 0) = u32::try_from(ptr).unwrap().to_le_bytes();
                *cx.get(offset + 4) = u32::try_from(len).unwrap().to_le_bytes();
                return Ok(());
            }
            ValSource::Record(fields) => {
                let InterfaceType::Record(t) = ty else {
                    bail!(
                        "type mismatch: cannot provide per-field record sources for {}",
                        desc(&ty)
                    );
                };
                let t = &cx.types[t];
                if t.fields.len() != fields.len() {
                    bail!("expected {} fields, got {}", t.fields.len(), fields.len());
                }
                let mut offset = offset;
                for (source, field) in fields.iter().zip(t.fields.iter()) {
                    source.store(
                        cx,
                        field.ty,
                        cx.types
                            .canonical_abi(&field.ty)
                            .next_field32_size(&mut offset),
                    )?;
                }
                return Ok(());
            }
            ValSource::ListElems(items) => {
                let InterfaceType::List(t) = ty else {
                    bail!(
                        "type mismatch: cannot provide per-element list sources for {}",
                        desc(&ty)
                    );
                };
                let element = cx.types[t].element;
                let (ptr, len) = lower_list_elems(cx, element, items)?;
                // FIXME(#4311): needs memory64 handling
                *cx.get(offset + 0) = u32::try_from(ptr).unwrap().to_le_bytes();
                *cx.get(offset + 4) = u32::try_from(len).unwrap().to_le_bytes();
                return Ok(());
            }
        };
        match (ty, val) {
            (InterfaceType::Bool, Val::Bool(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::Bool, _) => unexpected(ty, val),
            (InterfaceType::U8, Val::U8(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::U8, _) => unexpected(ty, val),
            (InterfaceType::S8, Val::S8(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::S8, _) => unexpected(ty, val),
            (InterfaceType::U16, Val::U16(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::U16, _) => unexpected(ty, val),
            (InterfaceType::S16, Val::S16(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::S16, _) => unexpected(ty, val),
            (InterfaceType::U32, Val::U32(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::U32, _) => unexpected(ty, val),
            (InterfaceType::S32, Val::S32(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::S32, _) => unexpected(ty, val),
            (InterfaceType::U64, Val::U64(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::U64, _) => unexpected(ty, val),
            (InterfaceType::S64, Val::S64(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::S64, _) => unexpected(ty, val),
            (InterfaceType::Float32, Val::Float32(value)) => {
                value.linear_lower_to_memory(cx, ty, offset)
            }
            (InterfaceType::Float32, _) => unexpected(ty, val),
            (InterfaceType::Float64, Val::Float64(value)) => {
                value.linear_lower_to_memory(cx, ty, offset)
            }
            (InterfaceType::Float64, _) => unexpected(ty, val),
            (InterfaceType::Char, Val::Char(value)) => value.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::Char, _) => unexpected(ty, val),
            (InterfaceType::String, Val::String(value)) => {
                value.linear_lower_to_memory(cx, ty, offset)
            }
            (InterfaceType::String, _) => unexpected(ty, val),

            // NB: resources do type-checking when they lower.
            (InterfaceType::Borrow(_) | InterfaceType::Own(_), Val::Resource(value)) => {
                value.linear_lower_to_memory(cx, ty, offset)
            }
            (InterfaceType::Borrow(_) | InterfaceType::Own(_), _) => unexpected(ty, val),
            (InterfaceType::List(ty), Val::List(values)) => {
                let ty = &cx.types[ty];
                let (ptr, len) = lower_list(cx, ty.element, values)?;
                // FIXME(#4311): needs memory64 handling
                *cx.get(offset + 0) = u32::try_from(ptr).unwrap().to_le_bytes();
                *cx.get(offset + 4) = u32::try_from(len).unwrap().to_le_bytes();
                Ok(())
            }
            (InterfaceType::List(_), _) => unexpected(ty, val),
            (InterfaceType::Map(ty_idx), Val::Map(values)) => {
                let map_ty = &cx.types[ty_idx];
                let (ptr, len) = lower_map(cx, map_ty, values)?;
                // FIXME(#4311): needs memory64 handling
                *cx.get(offset + 0) = u32::try_from(ptr).unwrap().to_le_bytes();
                *cx.get(offset + 4) = u32::try_from(len).unwrap().to_le_bytes();
                Ok(())
            }
            (InterfaceType::Map(_), _) => unexpected(ty, val),
            (InterfaceType::Record(ty), Val::Record(values)) => {
                let ty = &cx.types[ty];
                if ty.fields.len() != values.len() {
                    bail!("expected {} fields, got {}", ty.fields.len(), values.len());
                }
                let mut offset = offset;
                for ((name, value), field) in values.iter().zip(ty.fields.iter()) {
                    if *name != field.name {
                        bail!("expected field `{}`, got `{name}`", field.name);
                    }
                    ValSource::Val(value).store(
                        cx,
                        field.ty,
                        cx.types
                            .canonical_abi(&field.ty)
                            .next_field32_size(&mut offset),
                    )?;
                }
                Ok(())
            }
            (InterfaceType::Record(_), _) => unexpected(ty, val),
            (InterfaceType::Tuple(ty), Val::Tuple(values)) => {
                let ty = &cx.types[ty];
                if ty.types.len() != values.len() {
                    bail!("expected {} types, got {}", ty.types.len(), values.len());
                }
                let mut offset = offset;
                for (value, ty) in values.iter().zip(ty.types.iter()) {
                    ValSource::Val(value).store(
                        cx,
                        *ty,
                        cx.types.canonical_abi(ty).next_field32_size(&mut offset),
                    )?;
                }
                Ok(())
            }
            (InterfaceType::Tuple(_), _) => unexpected(ty, val),

            (InterfaceType::Variant(ty), Val::Variant(n, v)) => {
                GenericVariant::variant(&cx.types[ty], n, v)?.store(cx, offset)
            }
            (InterfaceType::Variant(_), _) => unexpected(ty, val),
            (InterfaceType::Enum(ty), Val::Enum(v)) => {
                GenericVariant::enum_(&cx.types[ty], v)?.store(cx, offset)
            }
            (InterfaceType::Enum(_), _) => unexpected(ty, val),
            (InterfaceType::Option(ty), Val::Option(v)) => {
                GenericVariant::option(&cx.types[ty], v).store(cx, offset)
            }
            (InterfaceType::Option(_), _) => unexpected(ty, val),
            (InterfaceType::Result(ty), Val::Result(v)) => {
                GenericVariant::result(&cx.types[ty], v)?.store(cx, offset)
            }
            (InterfaceType::Result(_), _) => unexpected(ty, val),

            (InterfaceType::Flags(ty), Val::Flags(flags)) => {
                let ty = &cx.types[ty];
                let storage = flags_to_storage(ty, flags)?;
                match FlagsSize::from_count(ty.names.len()) {
                    FlagsSize::Size0 => {}
                    FlagsSize::Size1 => u8::try_from(storage[0]).unwrap().linear_lower_to_memory(
                        cx,
                        InterfaceType::U8,
                        offset,
                    )?,
                    FlagsSize::Size2 => u16::try_from(storage[0]).unwrap().linear_lower_to_memory(
                        cx,
                        InterfaceType::U16,
                        offset,
                    )?,
                    FlagsSize::Size4Plus(_) => {
                        let mut offset = offset;
                        for value in storage {
                            value.linear_lower_to_memory(cx, InterfaceType::U32, offset)?;
                            offset += 4;
                        }
                    }
                }
                Ok(())
            }
            (InterfaceType::Flags(_), _) => unexpected(ty, val),
            (InterfaceType::Future(_), Val::Future(f)) => f.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::Future(_), _) => unexpected(ty, val),
            (InterfaceType::Stream(_), Val::Stream(s)) => s.linear_lower_to_memory(cx, ty, offset),
            (InterfaceType::Stream(_), _) => unexpected(ty, val),
            (InterfaceType::ErrorContext(_), Val::ErrorContext(ErrorContextAny(rep))) => {
                concurrent::lower_error_context_to_index(*rep, cx, ty)?.linear_lower_to_memory(
                    cx,
                    InterfaceType::U32,
                    offset,
                )
            }
            (InterfaceType::ErrorContext(_), _) => unexpected(ty, val),
            (InterfaceType::FixedLengthList(ty), Val::FixedLengthList(values)) => {
                let ty = &cx.types[ty];
                if ty.size as usize != values.len() {
                    bail!("expected {} types, got {}", ty.size, values.len());
                }
                let elemsize = cx.types.canonical_abi(&ty.element).size32 as usize;
                for (n, value) in values.iter().enumerate() {
                    ValSource::Val(value).store(cx, ty.element, elemsize * n)?;
                }
                Ok(())
            }
            (InterfaceType::FixedLengthList(_), _) => unexpected(ty, val),
        }
    }
}

struct GenericVariant<'a> {
    discriminant: u32,
    payload: Option<(&'a Val, InterfaceType)>,
    abi: &'a CanonicalAbiInfo,
    info: &'a VariantInfo,
}

impl GenericVariant<'_> {
    fn result<'a>(
        ty: &'a TypeResult,
        r: &'a Result<Option<Box<Val>>, Option<Box<Val>>>,
    ) -> Result<GenericVariant<'a>> {
        let (discriminant, payload) = match r {
            Ok(val) => {
                let payload = match (val, ty.ok) {
                    (Some(val), Some(ty)) => Some((&**val, ty)),
                    (None, None) => None,
                    (Some(_), None) => {
                        bail!("payload provided to `ok` but not expected");
                    }
                    (None, Some(_)) => {
                        bail!("payload expected to `ok` but not provided");
                    }
                };
                (0, payload)
            }
            Err(val) => {
                let payload = match (val, ty.err) {
                    (Some(val), Some(ty)) => Some((&**val, ty)),
                    (None, None) => None,
                    (Some(_), None) => {
                        bail!("payload provided to `err` but not expected");
                    }
                    (None, Some(_)) => {
                        bail!("payload expected to `err` but not provided");
                    }
                };
                (1, payload)
            }
        };
        Ok(GenericVariant {
            discriminant,
            payload,
            abi: &ty.abi,
            info: &ty.info,
        })
    }

    fn option<'a>(ty: &'a TypeOption, r: &'a Option<Box<Val>>) -> GenericVariant<'a> {
        let (discriminant, payload) = match r {
            None => (0, None),
            Some(val) => (1, Some((&**val, ty.ty))),
        };
        GenericVariant {
            discriminant,
            payload,
            abi: &ty.abi,
            info: &ty.info,
        }
    }

    fn enum_<'a>(ty: &'a TypeEnum, discriminant: &str) -> Result<GenericVariant<'a>> {
        let discriminant = get_enum_discriminant(ty, discriminant)?;

        Ok(GenericVariant {
            discriminant,
            payload: None,
            abi: &ty.abi,
            info: &ty.info,
        })
    }

    fn variant<'a>(
        ty: &'a TypeVariant,
        discriminant_name: &str,
        payload: &'a Option<Box<Val>>,
    ) -> Result<GenericVariant<'a>> {
        let (discriminant, payload_ty) = get_variant_discriminant(ty, discriminant_name)?;

        let payload = match (payload, payload_ty) {
            (Some(val), Some(ty)) => Some((&**val, *ty)),
            (None, None) => None,
            (Some(_), None) => bail!("did not expect a payload for case `{discriminant_name}`"),
            (None, Some(_)) => bail!("expected a payload for case `{discriminant_name}`"),
        };

        Ok(GenericVariant {
            discriminant,
            payload,
            abi: &ty.abi,
            info: &ty.info,
        })
    }

    fn lower<T>(
        &self,
        cx: &mut LowerContext<'_, T>,
        dst: &mut IterMut<'_, MaybeUninit<ValRaw>>,
    ) -> Result<()> {
        next_mut(dst).write(ValRaw::u32(self.discriminant));

        // For the remaining lowered representation of this variant that
        // the payload didn't write we write out zeros here to ensure
        // the entire variant is written.
        let value_flat = match self.payload {
            Some((value, ty)) => {
                ValSource::Val(value).lower(cx, ty, dst)?;
                cx.types.canonical_abi(&ty).flat_count(usize::MAX).unwrap()
            }
            None => 0,
        };
        let variant_flat = self.abi.flat_count(usize::MAX).unwrap();
        for _ in (1 + value_flat)..variant_flat {
            next_mut(dst).write(ValRaw::u64(0));
        }
        Ok(())
    }

    fn store<T>(&self, cx: &mut LowerContext<'_, T>, offset: usize) -> Result<()> {
        match self.info.size {
            DiscriminantSize::Size1 => u8::try_from(self.discriminant)
                .unwrap()
                .linear_lower_to_memory(cx, InterfaceType::U8, offset)?,
            DiscriminantSize::Size2 => u16::try_from(self.discriminant)
                .unwrap()
                .linear_lower_to_memory(cx, InterfaceType::U16, offset)?,
            DiscriminantSize::Size4 => {
                self.discriminant
                    .linear_lower_to_memory(cx, InterfaceType::U32, offset)?
            }
        }

        if let Some((value, ty)) = self.payload {
            let offset = offset + usize::try_from(self.info.payload_offset32).unwrap();
            ValSource::Val(value).store(cx, ty, offset)?;
        }

        Ok(())
    }
}

/// Lower a `list<T>` from a run of pre-validated canonical element images:
/// one guest allocation and one `memcpy`, no per-element work.
///
/// The caller (the public `prepare_call` surface) has already checked that
/// the proof's element type structurally equals the parameter's element
/// type, so the images are byte-compatible with what per-element lowering
/// would have produced. The length re-check here is defense in depth — it
/// makes an internal type-confusion bug a clean error instead of a
/// mis-strided copy.
fn lower_list_flat<T>(
    cx: &mut LowerContext<'_, T>,
    element_type: InterfaceType,
    vb: &ValidatedCabiBytes<'_>,
) -> Result<(usize, usize)> {
    let abi = cx.types.canonical_abi(&element_type);
    let elt_size = usize::try_from(abi.size32)?;
    let elt_align = abi.align32;
    let bytes = vb.bytes();
    let count = vb.len();
    ensure!(
        count.checked_mul(elt_size) == Some(bytes.len()),
        "validated buffer of {} bytes does not agree with the parameter's \
         {elt_size}-byte element stride",
        bytes.len(),
    );
    let ptr = cx.realloc(0, 0, elt_align, bytes.len())?;
    cx.as_slice_mut()[ptr..][..bytes.len()].copy_from_slice(bytes);
    Ok((ptr, count))
}

/// Lower a `list<T>` from per-element sources: one contiguous guest
/// allocation, each element lowered by its own strategy into its slot.
///
/// The byte-for-byte mirror of `lower_list` (the all-`Val` case), which is
/// what keeps the guest oblivious to the host-side representation: the same
/// guest memory image results no matter which strategy produced each element.
fn lower_list_elems<T>(
    cx: &mut LowerContext<'_, T>,
    element_type: InterfaceType,
    items: &[ValSource<'_>],
) -> Result<(usize, usize)> {
    let abi = cx.types.canonical_abi(&element_type);
    let elt_size = usize::try_from(abi.size32)?;
    let elt_align = abi.align32;
    let size = items
        .len()
        .checked_mul(elt_size)
        .ok_or_else(|| crate::format_err!("size overflow copying a list"))?;
    let ptr = cx.realloc(0, 0, elt_align, size)?;
    let mut element_ptr = ptr;
    for item in items {
        item.store(cx, element_type, element_ptr)?;
        element_ptr += elt_size;
    }
    Ok((ptr, items.len()))
}

/// Lower a list with the specified element type and values.
fn lower_list<T>(
    cx: &mut LowerContext<'_, T>,
    element_type: InterfaceType,
    items: &[Val],
) -> Result<(usize, usize)> {
    let abi = cx.types.canonical_abi(&element_type);
    let elt_size = usize::try_from(abi.size32)?;
    let elt_align = abi.align32;
    let size = items
        .len()
        .checked_mul(elt_size)
        .ok_or_else(|| crate::format_err!("size overflow copying a list"))?;
    let ptr = cx.realloc(0, 0, elt_align, size)?;
    let mut element_ptr = ptr;
    for item in items {
        ValSource::Val(item).store(cx, element_type, element_ptr)?;
        element_ptr += elt_size;
    }
    Ok((ptr, items.len()))
}

/// Lower a map as list<tuple<k, v>> with the specified key and value types.
fn lower_map<T>(
    cx: &mut LowerContext<'_, T>,
    map_ty: &TypeMap,
    pairs: &[(Val, Val)],
) -> Result<(usize, usize)> {
    let key_type = map_ty.key;
    let value_type = map_ty.value;
    let value_offset = usize::try_from(map_ty.value_offset32).unwrap();
    let tuple_align = map_ty.entry_abi.align32;
    let tuple_size = usize::try_from(map_ty.entry_abi.size32).unwrap();

    let size = pairs
        .len()
        .checked_mul(tuple_size)
        .ok_or_else(|| crate::format_err!("size overflow copying a map"))?;
    let ptr = cx.realloc(0, 0, tuple_align, size)?;

    let mut tuple_ptr = ptr;
    for (key, value) in pairs {
        // Store key at tuple_ptr
        ValSource::Val(key).store(cx, key_type, tuple_ptr)?;
        // Store value at tuple_ptr + value_offset (properly aligned)
        ValSource::Val(value).store(cx, value_type, tuple_ptr + value_offset)?;
        tuple_ptr += tuple_size;
    }

    Ok((ptr, pairs.len()))
}

fn flags_to_storage(ty: &TypeFlags, flags: &[String]) -> Result<Vec<u32>> {
    let mut storage = match FlagsSize::from_count(ty.names.len()) {
        FlagsSize::Size0 => Vec::new(),
        FlagsSize::Size1 | FlagsSize::Size2 => vec![0],
        FlagsSize::Size4Plus(n) => vec![0; n.into()],
    };

    for flag in flags {
        let bit = ty
            .names
            .get_index_of(flag)
            .ok_or_else(|| crate::format_err!("unknown flag: `{flag}`"))?;
        storage[bit / 32] |= 1 << (bit % 32);
    }
    Ok(storage)
}

fn get_enum_discriminant(ty: &TypeEnum, n: &str) -> Result<u32> {
    ty.names
        .get_index_of(n)
        .ok_or_else(|| crate::format_err!("enum variant name `{n}` is not valid"))
        .map(|i| i.try_into().unwrap())
}

fn get_variant_discriminant<'a>(
    ty: &'a TypeVariant,
    name: &str,
) -> Result<(u32, &'a Option<InterfaceType>)> {
    let (i, _, ty) = ty
        .cases
        .get_full(name)
        .ok_or_else(|| crate::format_err!("unknown variant case: `{name}`"))?;
    Ok((i.try_into().unwrap(), ty))
}

fn next_mut<'a>(dst: &mut IterMut<'a, MaybeUninit<ValRaw>>) -> &'a mut MaybeUninit<ValRaw> {
    dst.next().unwrap()
}

#[cold]
fn unexpected<T>(ty: InterfaceType, val: &Val) -> Result<T> {
    bail!(
        "type mismatch: expected {}, found {}",
        desc(&ty),
        val.desc()
    )
}
