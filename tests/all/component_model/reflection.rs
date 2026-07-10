#![cfg(not(miri))]

//! Tests for the [`Type::is_cabi_inline`] / [`Type::are_all_bit_patterns_valid`]
//! reflection predicates.
//!
//! These two predicates classify a component-model type along two independent
//! axes: whether its canonical-ABI representation is entirely *inline* (fixed
//! layout, no out-of-line storage, no ownership — so its bytes are
//! bulk-transferable), and whether *every* bit pattern of that representation
//! is a valid value (so a bulk transfer additionally needs no validation).

use super::{Param, Type as CoreType, make_echo_component_with_params};
use wasmtime::component::{Component, Linker, Type};
use wasmtime::{Result, Store};

/// Instantiate an echo component whose single parameter is a `list<X>` (per
/// `decls`, which must define `$Foo'` as that list type) and reflect the
/// element type `X` out of it.
///
/// Wrapping the type of interest in a `list` keeps the echo component's core
/// signature uniform (a pointer/length pair) no matter how gnarly `X` is.
pub(super) fn list_element_type(engine: &wasmtime::Engine, decls: &str) -> Result<Type> {
    let mut store = Store::new(engine, ());
    let component = Component::new(
        engine,
        make_echo_component_with_params(
            decls,
            &[Param(CoreType::I32, Some(0)), Param(CoreType::I32, Some(4))],
        ),
    )?;
    let instance = Linker::new(engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = func.ty(&store);
    let param = ty.params().next().unwrap().1;
    Ok(param.unwrap_list().ty())
}

#[test]
fn classification() -> Result<()> {
    let engine = super::engine();

    // (type declarations defining `$Foo'` as `list<X>`, X inline?, X all-bit-patterns-valid?)
    let cases: &[(&str, bool, bool)] = &[
        // Fully copyable: integers, floats, and records/tuples of only those.
        (r#"(type $Foo' (list u32))"#, true, true),
        (r#"(type $Foo' (list s64))"#, true, true),
        (r#"(type $Foo' (list float32))"#, true, true),
        (
            r#"
            (type $r' (record (field "x" float32) (field "y" float32)))
            (export $r "r" (type $r'))
            (type $Foo' (list $r))
            "#,
            true,
            true,
        ),
        (
            r#"
            (type $t' (tuple u8 u32))
            (export $t "t" (type $t'))
            (type $Foo' (list $t))
            "#,
            true,
            true,
        ),
        // Inline but *not* every-bit-pattern-valid: constrained representations
        // that would need a validation sweep before an unchecked copy.
        (r#"(type $Foo' (list bool))"#, true, false),
        (r#"(type $Foo' (list char))"#, true, false),
        (
            r#"
            (type $e' (enum "a" "b" "c"))
            (export $e "e" (type $e'))
            (type $Foo' (list $e))
            "#,
            true,
            false,
        ),
        (
            r#"
            (type $f' (flags "r" "w" "x"))
            (export $f "f" (type $f'))
            (type $Foo' (list $f))
            "#,
            true,
            false,
        ),
        (
            r#"
            (type $o' (option u32))
            (export $o "o" (type $o'))
            (type $Foo' (list $o))
            "#,
            true,
            false,
        ),
        (
            r#"
            (type $v' (variant (case "a" u32) (case "b" float64)))
            (export $v "v" (type $v'))
            (type $Foo' (list $v))
            "#,
            true,
            false,
        ),
        // A composite is only as strong as its weakest member.
        (
            r#"
            (type $r' (record (field "ok" bool) (field "n" u32)))
            (export $r "r" (type $r'))
            (type $Foo' (list $r))
            "#,
            true,
            false,
        ),
        // Not inline at all: out-of-line storage somewhere in the type.
        (r#"(type $Foo' (list string))"#, false, false),
        (
            r#"
            (type $l' (list u32))
            (export $l "l" (type $l'))
            (type $Foo' (list $l))
            "#,
            false,
            false,
        ),
        (
            r#"
            (type $v' (variant (case "a" u32) (case "s" string)))
            (export $v "v" (type $v'))
            (type $Foo' (list $v))
            "#,
            false,
            false,
        ),
        (
            r#"
            (type $r' (record (field "name" string) (field "n" u32)))
            (export $r "r" (type $r'))
            (type $Foo' (list $r))
            "#,
            false,
            false,
        ),
    ];

    for (decls, inline, total) in cases {
        let element = list_element_type(&engine, decls)?;
        assert_eq!(
            element.is_cabi_inline(),
            *inline,
            "is_cabi_inline for {decls}"
        );
        assert_eq!(
            element.are_all_bit_patterns_valid(),
            *total,
            "are_all_bit_patterns_valid for {decls}"
        );
        // The stricter predicate must imply the weaker one.
        if element.are_all_bit_patterns_valid() {
            assert!(element.is_cabi_inline(), "total implies inline: {decls}");
        }
    }

    Ok(())
}

/// `list<T>` itself is never inline — its elements live out of line — even
/// when its *element* type is fully copyable.
#[test]
fn list_itself_is_not_inline() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"(type $Foo' (list u32))"#,
            &[Param(CoreType::I32, Some(0)), Param(CoreType::I32, Some(4))],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = func.ty(&store);
    let list = ty.params().next().unwrap().1;
    assert!(!list.is_cabi_inline());
    assert!(!list.are_all_bit_patterns_valid());
    Ok(())
}

/// Fixed-length lists (`list<T, N>`) are inline exactly when their element
/// type is, and every-bit-pattern-valid exactly when their element type is —
/// unlike ordinary `list<T>`, their elements are stored inline.
#[test]
fn fixed_length_list_classification() -> Result<()> {
    let mut config = super::config();
    config.wasm_component_model_fixed_length_lists(true);
    let engine = wasmtime::Engine::new(&config)?;

    let elem = list_element_type(
        &engine,
        r#"
        (type $fll' (list u32 2))
        (export $fll "fll" (type $fll'))
        (type $Foo' (list $fll))
        "#,
    )?;
    assert!(elem.is_cabi_inline());
    assert!(elem.are_all_bit_patterns_valid());

    let elem = list_element_type(
        &engine,
        r#"
        (type $fll' (list bool 2))
        (export $fll "fll" (type $fll'))
        (type $Foo' (list $fll))
        "#,
    )?;
    assert!(elem.is_cabi_inline());
    assert!(!elem.are_all_bit_patterns_valid());

    Ok(())
}
