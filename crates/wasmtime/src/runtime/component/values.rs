use crate::ValRaw;
use crate::component::ResourceAny;
use crate::component::concurrent::{ErrorContext, FutureAny, StreamAny};
use crate::component::func::{Lift, LiftContext, LowerContext, ValSource};
use crate::prelude::*;
use core::mem::MaybeUninit;
use core::slice::{Iter, IterMut};
use wasmtime_component_util::{DiscriminantSize, FlagsSize};
use wasmtime_environ::component::{
    InterfaceType, TypeFlags, TypeListIndex, TypeMapIndex, VariantInfo,
};

/// Represents possible runtime values which a component function can either
/// consume or produce
///
/// This is a dynamic representation of possible values in the component model.
/// Note that this is not an efficient representation but is instead intended to
/// be a flexible and somewhat convenient representation. The most efficient
/// representation of component model types is to use the `bindgen!` macro to
/// generate native Rust types with specialized liftings and lowerings.
///
/// This type is used in conjunction with [`Func::call`] for example if the
/// signature of a component is not statically known ahead of time.
///
/// # Equality and `Val`
///
/// This type implements both the Rust `PartialEq` and `Eq` traits. This type
/// additionally contains values which are not necessarily easily equated,
/// however, such as floats (`Float32` and `Float64`) and resources. Equality
/// does require that two values have the same type, and then these cases are
/// handled as:
///
/// * Floats are tested if they are "semantically the same" meaning all NaN
///   values are equal to all other NaN values. Additionally zero values must be
///   exactly the same, so positive zero is not equal to negative zero. The
///   primary use case at this time is fuzzing-related equality which this is
///   sufficient for.
///
/// * Resources are tested if their types and indices into the host table are
///   equal. This does not compare the underlying representation so borrows of
///   the same guest resource are not considered equal. This additionally
///   doesn't go further and test for equality in the guest itself (for example
///   two different heap allocations of `Box<u32>` can be equal in normal Rust
///   if they contain the same value, but will never be considered equal when
///   compared as `Val::Resource`s).
///
/// In general if a strict guarantee about equality is required here it's
/// recommended to "build your own" as this equality intended for fuzzing
/// Wasmtime may not be suitable for you.
///
/// # Component model types and `Val`
///
/// The `Val` type here does not contain enough information to say what the
/// component model type of a `Val` is. This is instead more of an AST of sorts.
/// For example the `Val::Enum` only carries information about a single
/// discriminant, not the entire enumeration or what it's a discriminant of.
///
/// This means that when a `Val` is passed to Wasmtime, for example as a
/// function parameter when calling a function or as a return value from an
/// host-defined imported function, then it must pass a type-check. Instances of
/// `Val` are type-checked against what's required by the component itself.
///
/// [`Func::call`]: crate::component::Func::call
#[derive(Debug, Clone)]
#[expect(missing_docs, reason = "self-describing variants")]
pub enum Val {
    Bool(bool),
    S8(i8),
    U8(u8),
    S16(i16),
    U16(u16),
    S32(i32),
    U32(u32),
    S64(i64),
    U64(u64),
    Float32(f32),
    Float64(f64),
    Char(char),
    String(String),
    List(Vec<Val>),
    /// A map type represented as a list of key-value pairs.
    /// Duplicate keys are allowed and follow "last value wins" semantics.
    Map(Vec<(Val, Val)>),
    Record(Vec<(String, Val)>),
    Tuple(Vec<Val>),
    Variant(String, Option<Box<Val>>),
    Enum(String),
    Option(Option<Box<Val>>),
    Result(Result<Option<Box<Val>>, Option<Box<Val>>>),
    Flags(Vec<String>),
    Resource(ResourceAny),
    Future(FutureAny),
    Stream(StreamAny),
    ErrorContext(ErrorContextAny),
    FixedLengthList(Vec<Val>),
}

impl Val {
    /// Deserialize a value of this type from core Wasm stack values.
    pub(crate) fn lift(
        cx: &mut LiftContext<'_>,
        ty: InterfaceType,
        src: &mut Iter<'_, ValRaw>,
    ) -> Result<Val> {
        Ok(match ty {
            InterfaceType::Bool => Val::Bool(bool::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::S8 => Val::S8(i8::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::U8 => Val::U8(u8::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::S16 => Val::S16(i16::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::U16 => Val::U16(u16::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::S32 => Val::S32(i32::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::U32 => Val::U32(u32::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::S64 => Val::S64(i64::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::U64 => Val::U64(u64::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::Float32 => Val::Float32(f32::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::Float64 => Val::Float64(f64::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::Char => Val::Char(char::linear_lift_from_flat(cx, ty, next(src))?),
            InterfaceType::Own(_) | InterfaceType::Borrow(_) => {
                Val::Resource(ResourceAny::linear_lift_from_flat(cx, ty, next(src))?)
            }
            InterfaceType::String => Val::String(<_>::linear_lift_from_flat(
                cx,
                ty,
                &[*next(src), *next(src)],
            )?),
            InterfaceType::List(i) => {
                let (ptr, len) = lift_flat_pointer_pair(cx, src)?;
                load_list(cx, i, ptr, len)?
            }
            InterfaceType::Map(i) => {
                let (ptr, len) = lift_flat_pointer_pair(cx, src)?;
                load_map(cx, i, ptr, len)?
            }
            InterfaceType::Record(i) => Val::Record(
                cx.types[i]
                    .fields
                    .iter()
                    .map(|field| {
                        let val = Self::lift(cx, field.ty, src)?;
                        Ok((field.name.to_string(), val))
                    })
                    .collect::<Result<_>>()?,
            ),
            InterfaceType::Tuple(i) => Val::Tuple(
                cx.types[i]
                    .types
                    .iter()
                    .map(|ty| Self::lift(cx, *ty, src))
                    .collect::<Result<_>>()?,
            ),
            InterfaceType::Variant(i) => {
                let vty = &cx.types[i];
                let (discriminant, value) = lift_variant(
                    cx,
                    cx.types.canonical_abi(&ty).flat_count(usize::MAX).unwrap(),
                    vty.cases.values().copied(),
                    src,
                )?;

                let (k, _) = vty.cases.get_index(discriminant as usize).unwrap();
                Val::Variant(k.clone(), value)
            }
            InterfaceType::Enum(i) => {
                let ety = &cx.types[i];
                let (discriminant, _) = lift_variant(
                    cx,
                    cx.types.canonical_abi(&ty).flat_count(usize::MAX).unwrap(),
                    ety.names.iter().map(|_| None),
                    src,
                )?;

                Val::Enum(ety.names[discriminant as usize].clone())
            }
            InterfaceType::Option(i) => {
                let (_discriminant, value) = lift_variant(
                    cx,
                    cx.types.canonical_abi(&ty).flat_count(usize::MAX).unwrap(),
                    [None, Some(cx.types[i].ty)].into_iter(),
                    src,
                )?;

                Val::Option(value)
            }
            InterfaceType::Result(i) => {
                let result_ty = &cx.types[i];
                let (discriminant, value) = lift_variant(
                    cx,
                    cx.types.canonical_abi(&ty).flat_count(usize::MAX).unwrap(),
                    [result_ty.ok, result_ty.err].into_iter(),
                    src,
                )?;

                Val::Result(if discriminant == 0 {
                    Ok(value)
                } else {
                    Err(value)
                })
            }
            InterfaceType::Flags(i) => {
                let u32_count = cx.types.canonical_abi(&ty).flat_count(usize::MAX).unwrap();
                let ty = &cx.types[i];
                let mut flags = Vec::new();
                for i in 0..u32::try_from(u32_count).unwrap() {
                    push_flags(
                        ty,
                        &mut flags,
                        i * 32,
                        u32::linear_lift_from_flat(cx, InterfaceType::U32, next(src))?,
                    );
                }

                Val::Flags(flags)
            }
            InterfaceType::Future(_) => {
                Val::Future(FutureAny::linear_lift_from_flat(cx, ty, next(src))?)
            }
            InterfaceType::Stream(_) => {
                Val::Stream(StreamAny::linear_lift_from_flat(cx, ty, next(src))?)
            }
            InterfaceType::ErrorContext(_) => {
                ErrorContext::linear_lift_from_flat(cx, ty, next(src))?.into_val()
            }
            InterfaceType::FixedLengthList(i) => {
                let number_elements = usize::try_from(cx.types[i].size)?;
                cx.consume_fuel_array(number_elements, size_of::<Val>())?;
                Val::FixedLengthList(
                    (0..number_elements)
                        .map(|_| Self::lift(cx, cx.types[i].element, src))
                        .collect::<Result<_>>()?,
                )
            }
        })
    }

    /// Deserialize a value of this type from the heap.
    pub(crate) fn load(cx: &mut LiftContext<'_>, ty: InterfaceType, bytes: &[u8]) -> Result<Val> {
        Ok(match ty {
            InterfaceType::Bool => Val::Bool(bool::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::S8 => Val::S8(i8::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::U8 => Val::U8(u8::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::S16 => Val::S16(i16::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::U16 => Val::U16(u16::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::S32 => Val::S32(i32::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::U32 => Val::U32(u32::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::S64 => Val::S64(i64::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::U64 => Val::U64(u64::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::Float32 => Val::Float32(f32::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::Float64 => Val::Float64(f64::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::Char => Val::Char(char::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::String => Val::String(<_>::linear_lift_from_memory(cx, ty, bytes)?),
            InterfaceType::Own(_) | InterfaceType::Borrow(_) => {
                Val::Resource(ResourceAny::linear_lift_from_memory(cx, ty, bytes)?)
            }
            InterfaceType::List(i) => {
                let (ptr, len) = load_flat_pointer_pair(bytes);
                load_list(cx, i, ptr, len)?
            }
            InterfaceType::Map(i) => {
                let (ptr, len) = load_flat_pointer_pair(bytes);
                load_map(cx, i, ptr, len)?
            }

            InterfaceType::Record(i) => {
                let mut offset = 0;
                let fields = cx.types[i].fields.iter();
                Val::Record(
                    fields
                        .map(|field| -> Result<(String, Val)> {
                            let abi = cx.types.canonical_abi(&field.ty);
                            let offset = abi.next_field32(&mut offset);
                            let offset = usize::try_from(offset).unwrap();
                            let size = usize::try_from(abi.size32).unwrap();
                            Ok((
                                field.name.to_string(),
                                Val::load(cx, field.ty, &bytes[offset..][..size])?,
                            ))
                        })
                        .collect::<Result<_>>()?,
                )
            }
            InterfaceType::Tuple(i) => {
                let types = cx.types[i].types.iter().copied();
                let mut offset = 0;
                Val::Tuple(
                    types
                        .map(|ty| {
                            let abi = cx.types.canonical_abi(&ty);
                            let offset = abi.next_field32(&mut offset);
                            let offset = usize::try_from(offset).unwrap();
                            let size = usize::try_from(abi.size32).unwrap();
                            Val::load(cx, ty, &bytes[offset..][..size])
                        })
                        .collect::<Result<_>>()?,
                )
            }
            InterfaceType::Variant(i) => {
                let ty = &cx.types[i];
                let (discriminant, value) =
                    load_variant(cx, &ty.info, ty.cases.values().copied(), bytes)?;

                let (k, _) = ty.cases.get_index(discriminant as usize).unwrap();
                Val::Variant(k.clone(), value)
            }
            InterfaceType::Enum(i) => {
                let ty = &cx.types[i];
                let (discriminant, _) =
                    load_variant(cx, &ty.info, ty.names.iter().map(|_| None), bytes)?;

                Val::Enum(ty.names[discriminant as usize].clone())
            }
            InterfaceType::Option(i) => {
                let ty = &cx.types[i];
                let (_discriminant, value) =
                    load_variant(cx, &ty.info, [None, Some(ty.ty)].into_iter(), bytes)?;

                Val::Option(value)
            }
            InterfaceType::Result(i) => {
                let ty = &cx.types[i];
                let (discriminant, value) =
                    load_variant(cx, &ty.info, [ty.ok, ty.err].into_iter(), bytes)?;

                Val::Result(if discriminant == 0 {
                    Ok(value)
                } else {
                    Err(value)
                })
            }
            InterfaceType::Flags(i) => {
                let ty = &cx.types[i];
                let mut flags = Vec::new();
                match FlagsSize::from_count(ty.names.len()) {
                    FlagsSize::Size0 => {}
                    FlagsSize::Size1 => {
                        let bits = u8::linear_lift_from_memory(cx, InterfaceType::U8, bytes)?;
                        push_flags(ty, &mut flags, 0, u32::from(bits));
                    }
                    FlagsSize::Size2 => {
                        let bits = u16::linear_lift_from_memory(cx, InterfaceType::U16, bytes)?;
                        push_flags(ty, &mut flags, 0, u32::from(bits));
                    }
                    FlagsSize::Size4Plus(n) => {
                        for i in 0..n {
                            let bits = u32::linear_lift_from_memory(
                                cx,
                                InterfaceType::U32,
                                &bytes[usize::from(i) * 4..][..4],
                            )?;
                            push_flags(ty, &mut flags, u32::from(i) * 32, bits);
                        }
                    }
                }
                Val::Flags(flags)
            }
            InterfaceType::Future(_) => FutureAny::linear_lift_from_memory(cx, ty, bytes)?.into(),
            InterfaceType::Stream(_) => StreamAny::linear_lift_from_memory(cx, ty, bytes)?.into(),
            InterfaceType::ErrorContext(_) => {
                ErrorContext::linear_lift_from_memory(cx, ty, bytes)?.into_val()
            }
            InterfaceType::FixedLengthList(i) => {
                let element_type = cx.types[i].element;
                let abi = cx.types.canonical_abi(&element_type);
                let element_size = usize::try_from(abi.size32)?;
                let number_elements = usize::try_from(cx.types[i].size)?;

                match number_elements.checked_mul(element_size) {
                    Some(total_size) if total_size <= bytes.len() => {
                        cx.consume_fuel_array(number_elements, size_of::<Val>())?;
                        Val::Tuple(
                            (0..number_elements)
                                .map(|n| {
                                    // the match already checked that the whole array fits into usize
                                    let offset = element_size.wrapping_mul(n);
                                    Val::load(cx, element_type, &bytes[offset..][..element_size])
                                })
                                .collect::<Result<_>>()?,
                        )
                    }
                    _ => bail!("fixed length list out of bounds of memory"),
                }
            }
        })
    }

    /// Serialize this value as core Wasm stack values.
    ///
    /// This is a thin shim over the dynamic lowering driver: a [`Val`] is one
    /// kind of [`ValSource`], and the recursive type-directed traversal lives
    /// with the driver in `func::source`.
    pub(crate) fn lower<T>(
        &self,
        cx: &mut LowerContext<'_, T>,
        ty: InterfaceType,
        dst: &mut IterMut<'_, MaybeUninit<ValRaw>>,
    ) -> Result<()> {
        ValSource::Val(self).lower(cx, ty, dst)
    }

    /// Serialize this value to the heap at the specified memory location.
    ///
    /// As with [`Val::lower`] this is a shim over the dynamic lowering driver
    /// in `func::source`.
    pub(crate) fn store<T>(
        &self,
        cx: &mut LowerContext<'_, T>,
        ty: InterfaceType,
        offset: usize,
    ) -> Result<()> {
        ValSource::Val(self).store(cx, ty, offset)
    }

    pub(crate) fn desc(&self) -> &'static str {
        match self {
            Val::Bool(_) => "bool",
            Val::U8(_) => "u8",
            Val::S8(_) => "s8",
            Val::U16(_) => "u16",
            Val::S16(_) => "s16",
            Val::U32(_) => "u32",
            Val::S32(_) => "s32",
            Val::U64(_) => "u64",
            Val::S64(_) => "s64",
            Val::Float32(_) => "f32",
            Val::Float64(_) => "f64",
            Val::Char(_) => "char",
            Val::List(_) => "list",
            Val::Map(_) => "map",
            Val::String(_) => "string",
            Val::Record(_) => "record",
            Val::Enum(_) => "enum",
            Val::Variant(..) => "variant",
            Val::Tuple(_) => "tuple",
            Val::Option(_) => "option",
            Val::Result(_) => "result",
            Val::Resource(_) => "resource",
            Val::Flags(_) => "flags",
            Val::Future(_) => "future",
            Val::Stream(_) => "stream",
            Val::ErrorContext(_) => "error-context",
            Val::FixedLengthList(_) => "list<_, N>",
        }
    }

    /// Deserialize a [`Val`] from its [`crate::component::wasm_wave`] encoding. Deserialization
    /// requires a target [`crate::component::Type`].
    #[cfg(feature = "wave")]
    pub fn from_wave(ty: &crate::component::Type, s: &str) -> Result<Self> {
        Ok(wasm_wave::from_str(ty, s)?)
    }

    /// Serialize a [`Val`] to its [`crate::component::wasm_wave`] encoding.
    #[cfg(feature = "wave")]
    pub fn to_wave(&self) -> Result<String> {
        Ok(wasm_wave::to_string(self)?)
    }
}

impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // IEEE 754 equality considers NaN inequal to NaN and negative zero
            // equal to positive zero, however we do the opposite here, because
            // this logic is used by testing and fuzzing, which want to know
            // whether two values are semantically the same, rather than
            // numerically equal.
            (Self::Float32(l), Self::Float32(r)) => {
                (*l != 0.0 && l == r)
                    || (*l == 0.0 && l.to_bits() == r.to_bits())
                    || (l.is_nan() && r.is_nan())
            }
            (Self::Float32(_), _) => false,
            (Self::Float64(l), Self::Float64(r)) => {
                (*l != 0.0 && l == r)
                    || (*l == 0.0 && l.to_bits() == r.to_bits())
                    || (l.is_nan() && r.is_nan())
            }
            (Self::Float64(_), _) => false,

            (Self::Bool(l), Self::Bool(r)) => l == r,
            (Self::Bool(_), _) => false,
            (Self::S8(l), Self::S8(r)) => l == r,
            (Self::S8(_), _) => false,
            (Self::U8(l), Self::U8(r)) => l == r,
            (Self::U8(_), _) => false,
            (Self::S16(l), Self::S16(r)) => l == r,
            (Self::S16(_), _) => false,
            (Self::U16(l), Self::U16(r)) => l == r,
            (Self::U16(_), _) => false,
            (Self::S32(l), Self::S32(r)) => l == r,
            (Self::S32(_), _) => false,
            (Self::U32(l), Self::U32(r)) => l == r,
            (Self::U32(_), _) => false,
            (Self::S64(l), Self::S64(r)) => l == r,
            (Self::S64(_), _) => false,
            (Self::U64(l), Self::U64(r)) => l == r,
            (Self::U64(_), _) => false,
            (Self::Char(l), Self::Char(r)) => l == r,
            (Self::Char(_), _) => false,
            (Self::String(l), Self::String(r)) => l == r,
            (Self::String(_), _) => false,
            (Self::List(l), Self::List(r)) => l == r,
            (Self::List(_), _) => false,
            (Self::Map(l), Self::Map(r)) => l == r,
            (Self::Map(_), _) => false,
            (Self::Record(l), Self::Record(r)) => l == r,
            (Self::Record(_), _) => false,
            (Self::Tuple(l), Self::Tuple(r)) => l == r,
            (Self::Tuple(_), _) => false,
            (Self::Variant(ln, lv), Self::Variant(rn, rv)) => ln == rn && lv == rv,
            (Self::Variant(..), _) => false,
            (Self::Enum(l), Self::Enum(r)) => l == r,
            (Self::Enum(_), _) => false,
            (Self::Option(l), Self::Option(r)) => l == r,
            (Self::Option(_), _) => false,
            (Self::Result(l), Self::Result(r)) => l == r,
            (Self::Result(_), _) => false,
            (Self::Flags(l), Self::Flags(r)) => l == r,
            (Self::Flags(_), _) => false,
            (Self::Resource(l), Self::Resource(r)) => l == r,
            (Self::Resource(_), _) => false,
            (Self::Future(l), Self::Future(r)) => l == r,
            (Self::Future(_), _) => false,
            (Self::Stream(l), Self::Stream(r)) => l == r,
            (Self::Stream(_), _) => false,
            (Self::ErrorContext(l), Self::ErrorContext(r)) => l == r,
            (Self::ErrorContext(_), _) => false,
            (Self::FixedLengthList(l), Self::FixedLengthList(r)) => l == r,
            (Self::FixedLengthList(_), _) => false,
        }
    }
}

impl Eq for Val {}

fn lift_flat_pointer_pair(
    cx: &mut LiftContext<'_>,
    src: &mut Iter<'_, ValRaw>,
) -> Result<(usize, usize)> {
    // FIXME(#4311): needs memory64 treatment
    let ptr = u32::linear_lift_from_flat(cx, InterfaceType::U32, next(src))? as usize;
    let len = u32::linear_lift_from_flat(cx, InterfaceType::U32, next(src))? as usize;
    Ok((ptr, len))
}

fn load_flat_pointer_pair(bytes: &[u8]) -> (usize, usize) {
    let ptr = u32::from_le_bytes(*bytes[..4].as_array().unwrap()) as usize;
    let len = u32::from_le_bytes(*bytes[4..].as_array().unwrap()) as usize;
    (ptr, len)
}

fn load_list(cx: &mut LiftContext<'_>, ty: TypeListIndex, ptr: usize, len: usize) -> Result<Val> {
    let elem = cx.types[ty].element;
    let abi = cx.types.canonical_abi(&elem);
    let element_size = usize::try_from(abi.size32).unwrap();
    let element_alignment = abi.align32;

    match len
        .checked_mul(element_size)
        .and_then(|len| ptr.checked_add(len))
    {
        Some(n) if n <= cx.memory().len() => cx.consume_fuel_array(len, size_of::<Val>())?,
        _ => bail!("list pointer/length out of bounds of memory"),
    }
    if ptr % usize::try_from(element_alignment)? != 0 {
        bail!("list pointer is not aligned")
    }

    Ok(Val::List(
        (0..len)
            .map(|index| {
                Val::load(
                    cx,
                    elem,
                    &cx.memory()[ptr + (index * element_size)..][..element_size],
                )
            })
            .collect::<Result<_>>()?,
    ))
}

fn load_map(cx: &mut LiftContext<'_>, ty: TypeMapIndex, ptr: usize, len: usize) -> Result<Val> {
    // Maps are stored as list<tuple<k, v>> in canonical ABI
    let map_ty = &cx.types[ty];
    let key_ty = map_ty.key;
    let value_ty = map_ty.value;

    let key_abi = cx.types.canonical_abi(&key_ty);
    let value_abi = cx.types.canonical_abi(&value_ty);
    let key_size = usize::try_from(key_abi.size32).unwrap();
    let value_size = usize::try_from(value_abi.size32).unwrap();
    let value_offset = usize::try_from(map_ty.value_offset32).unwrap();
    let tuple_alignment = map_ty.entry_abi.align32;
    let tuple_size = usize::try_from(map_ty.entry_abi.size32).unwrap();

    // Bounds check
    match len
        .checked_mul(tuple_size)
        .and_then(|len| ptr.checked_add(len))
    {
        Some(n) if n <= cx.memory().len() => cx.consume_fuel_array(len, size_of::<(Val, Val)>())?,
        _ => bail!("map pointer/length out of bounds of memory"),
    }
    if ptr % usize::try_from(tuple_alignment)? != 0 {
        bail!("map pointer is not aligned")
    }

    // Load each tuple (key, value) into a Vec
    let mut map = Vec::with_capacity(len);
    for index in 0..len {
        let tuple_ptr = ptr + (index * tuple_size);
        let key = Val::load(cx, key_ty, &cx.memory()[tuple_ptr..][..key_size])?;
        let value = Val::load(
            cx,
            value_ty,
            &cx.memory()[tuple_ptr + value_offset..][..value_size],
        )?;
        map.push((key, value));
    }

    Ok(Val::Map(map))
}

fn load_variant(
    cx: &mut LiftContext<'_>,
    info: &VariantInfo,
    mut types: impl ExactSizeIterator<Item = Option<InterfaceType>>,
    bytes: &[u8],
) -> Result<(u32, Option<Box<Val>>)> {
    let discriminant = match info.size {
        DiscriminantSize::Size1 => u32::from(u8::linear_lift_from_memory(
            cx,
            InterfaceType::U8,
            &bytes[..1],
        )?),
        DiscriminantSize::Size2 => u32::from(u16::linear_lift_from_memory(
            cx,
            InterfaceType::U16,
            &bytes[..2],
        )?),
        DiscriminantSize::Size4 => {
            u32::linear_lift_from_memory(cx, InterfaceType::U32, &bytes[..4])?
        }
    };
    let len = types.len();
    let case_ty = types
        .nth(discriminant as usize)
        .ok_or_else(|| format_err!("discriminant {discriminant} out of range [0..{len})"))?;
    let value = match case_ty {
        Some(case_ty) => {
            let payload_offset = usize::try_from(info.payload_offset32).unwrap();
            let case_abi = cx.types.canonical_abi(&case_ty);
            let case_size = usize::try_from(case_abi.size32).unwrap();
            Some(Box::new(Val::load(
                cx,
                case_ty,
                &bytes[payload_offset..][..case_size],
            )?))
        }
        None => None,
    };
    Ok((discriminant, value))
}

fn lift_variant(
    cx: &mut LiftContext<'_>,
    flatten_count: usize,
    mut types: impl ExactSizeIterator<Item = Option<InterfaceType>>,
    src: &mut Iter<'_, ValRaw>,
) -> Result<(u32, Option<Box<Val>>)> {
    let len = types.len();
    let discriminant = next(src).get_u32();
    let ty = types
        .nth(discriminant as usize)
        .ok_or_else(|| format_err!("discriminant {discriminant} out of range [0..{len})"))?;
    let (value, value_flat) = match ty {
        Some(ty) => (
            Some(Box::new(Val::lift(cx, ty, src)?)),
            cx.types.canonical_abi(&ty).flat_count(usize::MAX).unwrap(),
        ),
        None => (None, 0),
    };
    for _ in (1 + value_flat)..flatten_count {
        next(src);
    }
    Ok((discriminant, value))
}

fn push_flags(ty: &TypeFlags, flags: &mut Vec<String>, mut offset: u32, mut bits: u32) {
    while bits > 0 && usize::try_from(offset).unwrap() < ty.names.len() {
        if bits & 1 != 0 {
            flags.push(ty.names[offset as usize].clone());
        }
        bits >>= 1;
        offset += 1;
    }
}

fn next<'a>(src: &mut Iter<'a, ValRaw>) -> &'a ValRaw {
    src.next().unwrap()
}

/// Represents a component model `error-context`.
///
/// Note that this type is not usable at this time as its implementation has not
/// been filled out. There are no operations on this and there's additionally no
/// ability to "drop" or deallocate this index.
//
// FIXME(#11161) this needs to be filled out implementation-wise
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorContextAny(pub(crate) u32);

impl From<bool> for Val {
    fn from(b: bool) -> Self {
        Val::Bool(b)
    }
}

impl From<u8> for Val {
    fn from(u: u8) -> Self {
        Val::U8(u)
    }
}

impl From<i8> for Val {
    fn from(i: i8) -> Self {
        Val::S8(i)
    }
}

impl From<u16> for Val {
    fn from(u: u16) -> Self {
        Val::U16(u)
    }
}

impl From<i16> for Val {
    fn from(i: i16) -> Self {
        Val::S16(i)
    }
}

impl From<u32> for Val {
    fn from(u: u32) -> Self {
        Val::U32(u)
    }
}

impl From<i32> for Val {
    fn from(i: i32) -> Self {
        Val::S32(i)
    }
}

impl From<u64> for Val {
    fn from(u: u64) -> Self {
        Val::U64(u)
    }
}

impl From<i64> for Val {
    fn from(i: i64) -> Self {
        Val::S64(i)
    }
}

impl From<char> for Val {
    fn from(i: char) -> Self {
        Val::Char(i)
    }
}

impl From<String> for Val {
    fn from(i: String) -> Self {
        Val::String(i)
    }
}

impl From<ResourceAny> for Val {
    fn from(i: ResourceAny) -> Self {
        Val::Resource(i)
    }
}

impl From<FutureAny> for Val {
    fn from(i: FutureAny) -> Self {
        Val::Future(i)
    }
}

impl From<StreamAny> for Val {
    fn from(i: StreamAny) -> Self {
        Val::Stream(i)
    }
}
