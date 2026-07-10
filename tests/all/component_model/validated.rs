#![cfg(not(miri))]

//! Tests for [`ValidatedCabiBytes`] / [`ValidatedCabiBytesBuf`]: the
//! proof-carrying wrappers around canonical-ABI byte images.
//!
//! Construction must be O(1)-accepting for `are_all_bit_patterns_valid`
//! types, must sweep-and-accept valid images of other inline types, must
//! reject invalid images (out-of-range discriminants, bad `bool`s/`char`s,
//! unknown `flags` bits), and must reject non-inline types outright.

use super::reflection::list_element_type;
use wasmtime::Result;
use wasmtime::component::{Type, ValidatedCabiBytes, ValidatedCabiBytesBuf};

#[test]
fn total_types_are_length_checked_only() -> Result<()> {
    let engine = super::engine();
    let u32_ty = list_element_type(&engine, r#"(type $Foo' (list u32))"#)?;

    // Any bit pattern goes; only the length matters.
    let bytes = [0xde, 0xad, 0xbe, 0xef, 0xff, 0xff, 0xff, 0xff];
    let ok = ValidatedCabiBytes::checked(&bytes, &u32_ty)?;
    assert_eq!(ok.len(), 2);
    assert!(!ok.is_empty());
    assert_eq!(ok.bytes(), &bytes);

    // An empty run is fine.
    assert_eq!(ValidatedCabiBytes::checked(&[], &u32_ty)?.len(), 0);

    // A ragged length is not.
    assert!(ValidatedCabiBytes::checked(&bytes[..5], &u32_ty).is_err());

    Ok(())
}

#[test]
fn bool_sweep() -> Result<()> {
    let engine = super::engine();
    let bool_ty = list_element_type(&engine, r#"(type $Foo' (list bool))"#)?;

    assert_eq!(ValidatedCabiBytes::checked(&[0, 1, 0, 1], &bool_ty)?.len(), 4);
    // Lifting would tolerate a 2, but lowering never produces one, so the
    // proof-carrying wrapper rejects it.
    let err = ValidatedCabiBytes::checked(&[0, 1, 2], &bool_ty).unwrap_err();
    assert!(format!("{err:?}").contains("element 2"), "{err:?}");
    Ok(())
}

#[test]
fn char_sweep() -> Result<()> {
    let engine = super::engine();
    let char_ty = list_element_type(&engine, r#"(type $Foo' (list char))"#)?;

    let good = [
        u32::from('A').to_le_bytes(),
        u32::from('🦀').to_le_bytes(),
    ]
    .concat();
    assert_eq!(ValidatedCabiBytes::checked(&good, &char_ty)?.len(), 2);

    // A surrogate and an out-of-range scalar are both rejected.
    assert!(ValidatedCabiBytes::checked(&0xD800u32.to_le_bytes(), &char_ty).is_err());
    assert!(ValidatedCabiBytes::checked(&0x110000u32.to_le_bytes(), &char_ty).is_err());
    Ok(())
}

#[test]
fn enum_discriminant_sweep() -> Result<()> {
    let engine = super::engine();
    let enum_ty = list_element_type(
        &engine,
        r#"
        (type $e' (enum "a" "b" "c"))
        (export $e "e" (type $e'))
        (type $Foo' (list $e))
        "#,
    )?;

    // Three cases -> a one-byte discriminant; 0..=2 valid, 3 not.
    assert_eq!(ValidatedCabiBytes::checked(&[0, 1, 2], &enum_ty)?.len(), 3);
    assert!(ValidatedCabiBytes::checked(&[3], &enum_ty).is_err());
    Ok(())
}

#[test]
fn flags_unknown_bits_sweep() -> Result<()> {
    let engine = super::engine();
    let flags_ty = list_element_type(
        &engine,
        r#"
        (type $f' (flags "r" "w" "x"))
        (export $f "f" (type $f'))
        (type $Foo' (list $f))
        "#,
    )?;

    // Three flags -> one byte; every subset of the low three bits is valid.
    assert_eq!(
        ValidatedCabiBytes::checked(&[0b000, 0b101, 0b111], &flags_ty)?.len(),
        3
    );
    assert!(ValidatedCabiBytes::checked(&[0b1000], &flags_ty).is_err());
    Ok(())
}

#[test]
fn record_sweep_checks_fields_not_padding() -> Result<()> {
    let engine = super::engine();
    // record { ok: bool, n: u32 } -- size 8: bool at 0, 3 bytes padding, u32
    // at 4. Inline but not bit-pattern-total, so images are swept.
    let rec_ty = list_element_type(
        &engine,
        r#"
        (type $r' (record (field "ok" bool) (field "n" u32)))
        (export $r "r" (type $r'))
        (type $Foo' (list $r))
        "#,
    )?;

    // Garbage padding is fine -- lowering doesn't canonicalize padding either.
    let image = [1, 0xAA, 0xBB, 0xCC, 0x78, 0x56, 0x34, 0x12];
    assert_eq!(ValidatedCabiBytes::checked(&image, &rec_ty)?.len(), 1);

    // ...but a bad *field* is caught through the recursion.
    let bad = [7, 0xAA, 0xBB, 0xCC, 0x78, 0x56, 0x34, 0x12];
    let err = ValidatedCabiBytes::checked(&bad, &rec_ty).unwrap_err();
    assert!(format!("{err:?}").contains("record field `ok`"), "{err:?}");
    Ok(())
}

#[test]
fn variant_discriminant_and_payload_sweep() -> Result<()> {
    let engine = super::engine();
    // variant { a(u32), b(float64) } -- one-byte discriminant, payload at
    // offset 8 (f64 alignment), total size 16.
    let var_ty = list_element_type(
        &engine,
        r#"
        (type $v' (variant (case "a" u32) (case "b" float64)))
        (export $v "v" (type $v'))
        (type $Foo' (list $v))
        "#,
    )?;

    let mut image = [0u8; 16];
    image[0] = 0; // case "a"
    image[8..12].copy_from_slice(&42u32.to_le_bytes());
    assert_eq!(ValidatedCabiBytes::checked(&image, &var_ty)?.len(), 1);

    image[0] = 1; // case "b"
    image[8..16].copy_from_slice(&1.5f64.to_le_bytes());
    assert_eq!(ValidatedCabiBytes::checked(&image, &var_ty)?.len(), 1);

    image[0] = 2; // out of range
    let err = ValidatedCabiBytes::checked(&image, &var_ty).unwrap_err();
    assert!(format!("{err:?}").contains("discriminant 2"), "{err:?}");
    Ok(())
}

#[test]
fn option_discriminant_sweep() -> Result<()> {
    let engine = super::engine();
    let opt_ty = list_element_type(
        &engine,
        r#"
        (type $o' (option u32))
        (export $o "o" (type $o'))
        (type $Foo' (list $o))
        "#,
    )?;

    // option<u32>: one-byte discriminant, u32 payload at offset 4, size 8.
    let none = [0u8; 8];
    let mut some = [0u8; 8];
    some[0] = 1;
    some[4..8].copy_from_slice(&7u32.to_le_bytes());
    let run = [none, some].concat();
    assert_eq!(ValidatedCabiBytes::checked(&run, &opt_ty)?.len(), 2);

    let mut bad = none;
    bad[0] = 2;
    assert!(ValidatedCabiBytes::checked(&bad, &opt_ty).is_err());
    Ok(())
}

#[test]
fn non_inline_types_rejected() {
    // `string` values can't be represented as self-contained bytes at all.
    let err = ValidatedCabiBytes::checked(&[], &Type::String).unwrap_err();
    assert!(
        format!("{err:?}").contains("not an inline type"),
        "{err:?}"
    );
}

#[test]
fn owning_buffer_carries_the_proof() -> Result<()> {
    let engine = super::engine();
    let enum_ty = list_element_type(
        &engine,
        r#"
        (type $e' (enum "a" "b"))
        (export $e "e" (type $e'))
        (type $Foo' (list $e))
        "#,
    )?;

    let buf = ValidatedCabiBytesBuf::checked(vec![0, 1, 0], &enum_ty)?;
    assert_eq!(buf.len(), 3);
    let view = buf.as_ref();
    assert_eq!(view.len(), 3);
    assert_eq!(view.bytes(), &[0, 1, 0]);
    assert_eq!(view.ty(), &enum_ty);

    assert!(ValidatedCabiBytesBuf::checked(vec![2], &enum_ty).is_err());
    Ok(())
}
