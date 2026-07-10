//! Proof-carrying canonical-ABI byte images.
//!
//! A [`ValidatedCabiBytes`] wraps a byte buffer together with a reflected
//! [`Type`] as *proof* that the buffer holds a contiguous run of valid
//! canonical-ABI images of that type — i.e. bytes indistinguishable from what
//! Wasmtime's own lowering of equivalent values would have produced. The
//! proof is established exactly once, at construction:
//!
//! * for types where [`Type::are_all_bit_patterns_valid`] holds (integers,
//!   floats, and records/tuples/fixed-length-lists of only those) the check
//!   is **O(1)** — only the buffer length is examined;
//! * for other [`Type::is_cabi_inline`] types (`bool`, `char`, `enum`,
//!   `flags`, and discriminated unions of inline types) construction runs a
//!   **linear validation sweep** over the images;
//! * non-inline types (anything transitively containing a `string`, `list`,
//!   or ownership such as a resource handle) are rejected outright — their
//!   values cannot be represented as a self-contained byte image at all.
//!
//! Because the fields are private and every constructor validates, holding a
//! `ValidatedCabiBytes` is unforgeable evidence that its bytes may be copied
//! into a guest's memory as-is, with no further inspection.
//!
//! Two deliberate limits of the "valid" claim, matching what Wasmtime's own
//! lowering produces rather than something stronger:
//!
//! * **Padding is not examined.** Lowering writes fields and discriminants,
//!   not the padding between them, so padding bytes carry no meaning and any
//!   content is accepted (and copied verbatim).
//! * **Float NaN payloads are not canonicalized**, matching the fully
//!   dynamic `Val` path, which also copies float bits verbatim.
//!
//! Note that validation is *stricter* than lifting for `bool`: the canonical
//! ABI's lifting tolerates any nonzero byte as `true`, but lowering only ever
//! produces `0` or `1`, so only those are accepted here.

use crate::component::types::Type;
use crate::prelude::*;
use wasmtime_component_util::{DiscriminantSize, FlagsSize};
use wasmtime_environ::component::{CanonicalAbiInfo, ComponentTypes, InterfaceType};

/// A borrowed run of canonical-ABI byte images, validated at construction.
///
/// Holding one of these is unforgeable proof that the wrapped bytes are a
/// contiguous run of `len()` valid canonical-ABI images of `ty()` (a single
/// value is a run of length 1) — bytes indistinguishable from what
/// Wasmtime's own lowering of equivalent values would have produced — so
/// they may be copied into a guest's memory as-is, with no further
/// inspection. Validation happens exactly once, in
/// [`checked`](ValidatedCabiBytes::checked): O(1) for
/// [`are_all_bit_patterns_valid`](Type::are_all_bit_patterns_valid) types, a
/// linear sweep for other [`is_cabi_inline`](Type::is_cabi_inline) types,
/// and an outright rejection for non-inline types.
///
/// Two deliberate limits of the "valid" claim, matching what lowering
/// itself produces: padding bytes are not examined (lowering never writes
/// them), and float NaN payloads are not canonicalized (the dynamic `Val`
/// path also copies float bits verbatim). Validation is *stricter* than
/// lifting for `bool`: lifting tolerates any nonzero byte as `true`, but
/// lowering only ever produces 0 or 1, so only those are accepted.
pub struct ValidatedCabiBytes<'a> {
    bytes: &'a [u8],
    ty: Type,
    elems: usize,
}

/// An owned run of canonical-ABI byte images, validated at construction.
///
/// The owning counterpart of [`ValidatedCabiBytes`], for storing validated
/// bytes beyond the lifetime of their source (e.g. collecting results from
/// one instance to replay into others later). Borrow it back down with
/// [`as_ref`](ValidatedCabiBytesBuf::as_ref).
pub struct ValidatedCabiBytesBuf {
    bytes: Vec<u8>,
    ty: Type,
    elems: usize,
}

impl core::fmt::Debug for ValidatedCabiBytes<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ValidatedCabiBytes")
            .field("ty", &self.ty)
            .field("elems", &self.elems)
            .field("bytes", &format_args!("[{} bytes]", self.bytes.len()))
            .finish()
    }
}

impl core::fmt::Debug for ValidatedCabiBytesBuf {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ValidatedCabiBytesBuf")
            .field("ty", &self.ty)
            .field("elems", &self.elems)
            .field("bytes", &format_args!("[{} bytes]", self.bytes.len()))
            .finish()
    }
}

impl<'a> ValidatedCabiBytes<'a> {
    /// Validate that `bytes` is a contiguous run of canonical-ABI images of
    /// `ty`, returning the proof-carrying wrapper on success.
    ///
    /// This is O(1) when `ty.are_all_bit_patterns_valid()`, and a linear
    /// sweep over the images otherwise. Returns an error if `ty` is not
    /// [`Type::is_cabi_inline`], if `bytes` is not a whole number of images,
    /// or if any image fails validation (an out-of-range `enum`
    /// discriminant, a `bool` other than 0/1, an invalid `char`, a `flags`
    /// value with unknown bits set, …).
    pub fn checked(bytes: &'a [u8], ty: &Type) -> Result<ValidatedCabiBytes<'a>> {
        let elems = validate_run(bytes, ty)?;
        Ok(ValidatedCabiBytes {
            bytes,
            ty: ty.clone(),
            elems,
        })
    }

    /// The validated bytes.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Copy these bytes into an owning [`ValidatedCabiBytesBuf`], carrying
    /// the proof across without re-validating.
    pub fn to_owned(&self) -> ValidatedCabiBytesBuf {
        ValidatedCabiBytesBuf {
            bytes: self.bytes.to_vec(),
            ty: self.ty.clone(),
            elems: self.elems,
        }
    }

    /// The component-model type these bytes are images of.
    pub fn ty(&self) -> &Type {
        &self.ty
    }

    /// The number of contiguous images in the run.
    pub fn len(&self) -> usize {
        self.elems
    }

    /// Whether the run contains no images.
    pub fn is_empty(&self) -> bool {
        self.elems == 0
    }
}

impl ValidatedCabiBytesBuf {
    /// Owning version of [`ValidatedCabiBytes::checked`]; identical
    /// validation over an owned buffer.
    pub fn checked(bytes: Vec<u8>, ty: &Type) -> Result<ValidatedCabiBytesBuf> {
        let elems = validate_run(&bytes, ty)?;
        Ok(ValidatedCabiBytesBuf {
            bytes,
            ty: ty.clone(),
            elems,
        })
    }

    /// Borrow this buffer as a [`ValidatedCabiBytes`], carrying the proof
    /// across without re-validating.
    pub fn as_ref(&self) -> ValidatedCabiBytes<'_> {
        ValidatedCabiBytes {
            bytes: &self.bytes,
            ty: self.ty.clone(),
            elems: self.elems,
        }
    }

    /// The validated bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The component-model type these bytes are images of.
    pub fn ty(&self) -> &Type {
        &self.ty
    }

    /// The number of contiguous images in the run.
    pub fn len(&self) -> usize {
        self.elems
    }

    /// Whether the run contains no images.
    pub fn is_empty(&self) -> bool {
        self.elems == 0
    }
}

/// Validate `bytes` as a run of images of `ty`, returning the image count.
fn validate_run(bytes: &[u8], ty: &Type) -> Result<usize> {
    ensure!(
        ty.is_cabi_inline(),
        "`{}` is not an inline type: its values cannot be represented as \
         self-contained canonical-ABI bytes",
        ty.desc(),
    );
    let (types, iface) = ty
        .as_inline_parts()
        .expect("inline types always have interface parts");
    let abi = canonical_abi(types, &iface);
    let size = usize::try_from(abi.size32).unwrap();
    if size == 0 {
        // Zero-size inline types (an empty record, zero-case flags) have a
        // single, empty, trivially-valid image; a "run" of them carries no
        // information. Only the empty buffer is meaningful.
        ensure!(
            bytes.is_empty(),
            "non-empty buffer for zero-size type `{}`",
            ty.desc(),
        );
        return Ok(0);
    }
    ensure!(
        bytes.len() % size == 0,
        "buffer length {} is not a whole number of {}-byte `{}` images",
        bytes.len(),
        size,
        ty.desc(),
    );
    if !ty.are_all_bit_patterns_valid() {
        for (i, image) in bytes.chunks_exact(size).enumerate() {
            validate_image(types, &iface, image)
                .with_context(|| format!("invalid `{}` image at element {i}", ty.desc()))?;
        }
    }
    Ok(bytes.len() / size)
}

/// The canonical ABI info for `ty`, without requiring type tables for
/// primitive leaves (which have none to offer).
fn canonical_abi<'a>(
    types: Option<&'a alloc::sync::Arc<ComponentTypes>>,
    ty: &InterfaceType,
) -> &'a CanonicalAbiInfo {
    match types {
        Some(types) => types.canonical_abi(ty),
        None => match ty {
            InterfaceType::Bool | InterfaceType::S8 | InterfaceType::U8 => {
                &CanonicalAbiInfo::SCALAR1
            }
            InterfaceType::S16 | InterfaceType::U16 => &CanonicalAbiInfo::SCALAR2,
            InterfaceType::S32
            | InterfaceType::U32
            | InterfaceType::Float32
            | InterfaceType::Char => &CanonicalAbiInfo::SCALAR4,
            InterfaceType::S64 | InterfaceType::U64 | InterfaceType::Float64 => {
                &CanonicalAbiInfo::SCALAR8
            }
            other => unreachable!("no type tables for non-primitive {other:?}"),
        },
    }
}

/// Validate a single canonical image of an inline `ty`.
///
/// `types` is only consulted for composite types, which always carry tables.
pub(crate) fn validate_image(
    types: Option<&alloc::sync::Arc<ComponentTypes>>,
    ty: &InterfaceType,
    image: &[u8],
) -> Result<()> {
    match ty {
        // Every bit pattern is a value.
        InterfaceType::S8
        | InterfaceType::U8
        | InterfaceType::S16
        | InterfaceType::U16
        | InterfaceType::S32
        | InterfaceType::U32
        | InterfaceType::S64
        | InterfaceType::U64
        | InterfaceType::Float32
        | InterfaceType::Float64 => Ok(()),

        // Canonical lowering only ever produces 0 or 1 (lifting is more
        // lenient, but "valid" here means "indistinguishable from our own
        // lowering's output").
        InterfaceType::Bool => {
            ensure!(image[0] <= 1, "bool byte is {}, not 0 or 1", image[0]);
            Ok(())
        }

        InterfaceType::Char => {
            let bits = u32::from_le_bytes(image[..4].try_into().unwrap());
            ensure!(
                char::from_u32(bits).is_some(),
                "0x{bits:08x} is not a valid char"
            );
            Ok(())
        }

        InterfaceType::Enum(i) => {
            let ty = &types.unwrap()[*i];
            let discriminant = read_discriminant(ty.info.size, image);
            ensure!(
                (discriminant as usize) < ty.names.len(),
                "enum discriminant {discriminant} out of range (< {})",
                ty.names.len(),
            );
            Ok(())
        }

        InterfaceType::Flags(i) => {
            let ty = &types.unwrap()[*i];
            let count = ty.names.len();
            match FlagsSize::from_count(count) {
                FlagsSize::Size0 => {}
                FlagsSize::Size1 => {
                    let bits = image[0];
                    let mask = u8::try_from(mask_u32(count)).unwrap();
                    ensure!(
                        bits & !mask == 0,
                        "flags value 0x{bits:02x} has unknown bits"
                    );
                }
                FlagsSize::Size2 => {
                    let bits = u16::from_le_bytes(image[..2].try_into().unwrap());
                    let mask = u16::try_from(mask_u32(count)).unwrap();
                    ensure!(
                        bits & !mask == 0,
                        "flags value 0x{bits:04x} has unknown bits"
                    );
                }
                FlagsSize::Size4Plus(words) => {
                    for w in 0..usize::from(words) {
                        let bits = u32::from_le_bytes(image[w * 4..][..4].try_into().unwrap());
                        let bits_here = count.saturating_sub(w * 32).min(32);
                        let mask = mask_u32(bits_here);
                        ensure!(
                            bits & !mask == 0,
                            "flags word {w} value 0x{bits:08x} has unknown bits"
                        );
                    }
                }
            }
            Ok(())
        }

        InterfaceType::Record(i) => {
            let types = types.unwrap();
            let ty = &types[*i];
            let mut offset = 0;
            for field in ty.fields.iter() {
                let abi = types.canonical_abi(&field.ty);
                let field_offset = abi.next_field32_size(&mut offset);
                let field_size = usize::try_from(abi.size32).unwrap();
                validate_image(Some(types), &field.ty, &image[field_offset..][..field_size])
                    .with_context(|| format!("record field `{}`", field.name))?;
            }
            Ok(())
        }

        InterfaceType::Tuple(i) => {
            let types = types.unwrap();
            let ty = &types[*i];
            let mut offset = 0;
            for (n, field_ty) in ty.types.iter().enumerate() {
                let abi = types.canonical_abi(field_ty);
                let field_offset = abi.next_field32_size(&mut offset);
                let field_size = usize::try_from(abi.size32).unwrap();
                validate_image(Some(types), field_ty, &image[field_offset..][..field_size])
                    .with_context(|| format!("tuple field {n}"))?;
            }
            Ok(())
        }

        InterfaceType::Variant(i) => {
            let types = types.unwrap();
            let ty = &types[*i];
            let discriminant = read_discriminant(ty.info.size, image);
            let (name, payload) = ty.cases.get_index(discriminant as usize).ok_or_else(|| {
                crate::format_err!(
                    "variant discriminant {discriminant} out of range (< {})",
                    ty.cases.len(),
                )
            })?;
            if let Some(payload_ty) = payload {
                let offset = usize::try_from(ty.info.payload_offset32).unwrap();
                let size = usize::try_from(types.canonical_abi(payload_ty).size32).unwrap();
                validate_image(Some(types), payload_ty, &image[offset..][..size])
                    .with_context(|| format!("variant case `{name}` payload"))?;
            }
            Ok(())
        }

        InterfaceType::Option(i) => {
            let types = types.unwrap();
            let ty = &types[*i];
            let discriminant = read_discriminant(ty.info.size, image);
            ensure!(
                discriminant <= 1,
                "option discriminant {discriminant} is not 0 or 1"
            );
            if discriminant == 1 {
                let offset = usize::try_from(ty.info.payload_offset32).unwrap();
                let size = usize::try_from(types.canonical_abi(&ty.ty).size32).unwrap();
                validate_image(Some(types), &ty.ty, &image[offset..][..size])
                    .context("option payload")?;
            }
            Ok(())
        }

        InterfaceType::Result(i) => {
            let types = types.unwrap();
            let ty = &types[*i];
            let discriminant = read_discriminant(ty.info.size, image);
            ensure!(
                discriminant <= 1,
                "result discriminant {discriminant} is not 0 or 1"
            );
            let payload = if discriminant == 0 { &ty.ok } else { &ty.err };
            if let Some(payload_ty) = payload {
                let offset = usize::try_from(ty.info.payload_offset32).unwrap();
                let size = usize::try_from(types.canonical_abi(payload_ty).size32).unwrap();
                validate_image(Some(types), payload_ty, &image[offset..][..size]).with_context(
                    || {
                        format!(
                            "result {} payload",
                            if discriminant == 0 { "ok" } else { "err" }
                        )
                    },
                )?;
            }
            Ok(())
        }

        InterfaceType::FixedLengthList(i) => {
            let types = types.unwrap();
            let ty = &types[*i];
            let elem_size = usize::try_from(types.canonical_abi(&ty.element).size32).unwrap();
            for n in 0..usize::try_from(ty.size).unwrap() {
                validate_image(
                    Some(types),
                    &ty.element,
                    &image[n * elem_size..][..elem_size],
                )
                .with_context(|| format!("fixed-length list element {n}"))?;
            }
            Ok(())
        }

        // Non-inline types are rejected before per-image validation begins.
        InterfaceType::String
        | InterfaceType::List(_)
        | InterfaceType::Map(_)
        | InterfaceType::Own(_)
        | InterfaceType::Borrow(_)
        | InterfaceType::Future(_)
        | InterfaceType::Stream(_)
        | InterfaceType::ErrorContext(_) => {
            unreachable!("non-inline type reached image validation")
        }
    }
}

fn read_discriminant(size: DiscriminantSize, image: &[u8]) -> u32 {
    match size {
        DiscriminantSize::Size1 => u32::from(image[0]),
        DiscriminantSize::Size2 => u32::from(u16::from_le_bytes(image[..2].try_into().unwrap())),
        DiscriminantSize::Size4 => u32::from_le_bytes(image[..4].try_into().unwrap()),
    }
}

/// A mask with the low `bits` bits set (`bits <= 32`).
fn mask_u32(bits: usize) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}
