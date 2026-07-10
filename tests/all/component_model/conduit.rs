#![cfg(not(miri))]

//! The guest→guest conduit: values produced by one component instance flow
//! through the host into another instance — in a different store — with the
//! canonical bytes copied once, no re-encoding, and validation paid at most
//! once at the boundary where the proof is minted.
//!
//! This is the end-to-end composition of the pieces: `invoke_scoped` +
//! `Results::view`/`view_checked` mint a [`ValidatedCabiBytes`] straight off
//! instance A's memory, and [`ValSource::ListFlat`] feeds it to instance B as
//! a single `memcpy`. The type-equality check at bind is *structural*, which
//! is exactly what lets a proof minted against A's reflected type satisfy
//! B's parameter type.

use super::{engine, make_echo_component, make_echo_component_with_params};
use wasmtime::component::{Component, Linker, Val, ValSource, ValSpec};
use wasmtime::{Result, Store};

/// `list<u32>` from A's results into B's argument, zero-copy view in between:
/// one validated copy A→B, no host-side materialization at all.
#[test]
fn conduit_list_u32_zero_copy() -> Result<()> {
    let engine = engine();
    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;

    // Two instances in two *separate* stores: the source's memory stays
    // borrowed (immutably, inside the scope) while the sink's store is
    // exclusively borrowed for its own call.
    let mut store_a = Store::new(&engine, ());
    let instance_a = Linker::new(&engine).instantiate(&mut store_a, &component)?;
    let func_a = instance_a.get_func(&mut store_a, "echo").unwrap();

    let mut store_b = Store::new(&engine, ());
    let instance_b = Linker::new(&engine).instantiate(&mut store_b, &component)?;
    let func_b = instance_b.get_func(&mut store_b, "echo").unwrap();

    let values = [3u32, 1, 4, 1, 5, 9, 2, 6];
    let expected = Val::List(values.iter().copied().map(Val::U32).collect());

    let prepared_a = func_a.prepare_call(&store_a, &[ValSpec::Val])?;
    let prepared_b = func_b.prepare_call(&store_b, &[ValSpec::ListFlat])?;

    // Drive A; while its result memory is live, feed a zero-copy view of it
    // straight into B.
    let mut out_b = [Val::Bool(false)];
    prepared_a
        .bind()
        .arg_val(&expected)
        .invoke_scoped(&mut store_a, |results| {
            let view = results.view(0)?; // proof minted off A's memory, O(1)
            prepared_b
                .bind()
                .arg(ValSource::ListFlat(view)) // memcpy A's bytes into B
                .invoke(&mut store_b, &mut out_b)?;
            Ok(())
        })?;

    assert_eq!(out_b[0], expected);
    Ok(())
}

/// `list<enum>` — a type whose bit patterns need validation — through the
/// conduit: the sweep is paid exactly once (minting the proof off A's
/// results); replaying the owned copy into B twice re-validates nothing.
#[test]
fn conduit_list_enum_validated_once() -> Result<()> {
    let engine = engine();
    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"
            (type $E' (enum "x" "y" "z"))
            (export $E "e" (type $E'))
            (type $Foo' (list $E))"#,
            &[
                super::Param(super::Type::I32, Some(0)),
                super::Param(super::Type::I32, Some(4)),
            ],
        ),
    )?;

    let mut store_a = Store::new(&engine, ());
    let instance_a = Linker::new(&engine).instantiate(&mut store_a, &component)?;
    let func_a = instance_a.get_func(&mut store_a, "echo").unwrap();

    let mut store_b = Store::new(&engine, ());
    let instance_b = Linker::new(&engine).instantiate(&mut store_b, &component)?;
    let func_b = instance_b.get_func(&mut store_b, "echo").unwrap();

    let expected = Val::List(vec![
        Val::Enum("z".to_string()),
        Val::Enum("x".to_string()),
        Val::Enum("y".to_string()),
    ]);

    let prepared_a = func_a.prepare_call(&store_a, &[ValSpec::Val])?;
    let prepared_b = func_b.prepare_call(&store_b, &[ValSpec::ListFlat])?;

    // Collect an *owned* validated copy out of A: `copy` = checked view +
    // to_owned, so the discriminant sweep happens here, once.
    let collected = prepared_a
        .bind()
        .arg_val(&expected)
        .invoke_scoped(&mut store_a, |results| results.copy(0))?;
    assert_eq!(collected.bytes(), &[2, 0, 1]);

    // Replay it into B repeatedly: each call is a plain memcpy of
    // already-proven bytes.
    for _ in 0..3 {
        let mut out_b = [Val::Bool(false)];
        prepared_b
            .bind()
            .arg(ValSource::ListFlat(collected.as_ref()))
            .invoke(&mut store_b, &mut out_b)?;
        assert_eq!(out_b[0], expected);
    }

    Ok(())
}

/// The conduit's type check is structural: a proof minted from instance A's
/// reflected element type is accepted by instance B of a *different
/// component* whose parameter is structurally the same type — and rejected
/// where it structurally differs.
#[test]
fn conduit_structural_type_equality() -> Result<()> {
    let engine = engine();

    // Same shape (list<record{x: f32, y: f32}>), two independently-compiled
    // components.
    let wat_points = make_echo_component_with_params(
        r#"
        (type $R' (record (field "x" float32) (field "y" float32)))
        (export $R "r" (type $R'))
        (type $Foo' (list $R))"#,
        &[
            super::Param(super::Type::I32, Some(0)),
            super::Param(super::Type::I32, Some(4)),
        ],
    );
    let component_a = Component::new(&engine, &wat_points)?;
    let component_b = Component::new(&engine, &wat_points)?;

    let mut store_a = Store::new(&engine, ());
    let func_a = Linker::new(&engine)
        .instantiate(&mut store_a, &component_a)?
        .get_func(&mut store_a, "echo")
        .unwrap();
    let mut store_b = Store::new(&engine, ());
    let func_b = Linker::new(&engine)
        .instantiate(&mut store_b, &component_b)?
        .get_func(&mut store_b, "echo")
        .unwrap();

    let expected = Val::List(vec![Val::Record(vec![
        ("x".to_string(), Val::Float32(1.0)),
        ("y".to_string(), Val::Float32(-2.0)),
    ])]);

    let prepared_a = func_a.prepare_call(&store_a, &[ValSpec::Val])?;
    let prepared_b = func_b.prepare_call(&store_b, &[ValSpec::ListFlat])?;

    let mut out_b = [Val::Bool(false)];
    prepared_a
        .bind()
        .arg_val(&expected)
        .invoke_scoped(&mut store_a, |results| {
            // Proof minted against component A's type tables, accepted by
            // component B's structurally equal parameter.
            prepared_b
                .bind()
                .arg(ValSource::ListFlat(results.view(0)?))
                .invoke(&mut store_b, &mut out_b)?;
            Ok(())
        })?;
    assert_eq!(out_b[0], expected);

    // A structurally different sink (list<u32>) rejects the same proof.
    let component_c = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let mut store_c = Store::new(&engine, ());
    let func_c = Linker::new(&engine)
        .instantiate(&mut store_c, &component_c)?
        .get_func(&mut store_c, "echo")
        .unwrap();
    let prepared_c = func_c.prepare_call(&store_c, &[ValSpec::ListFlat])?;

    let mut out_c = [Val::Bool(false)];
    let err = prepared_a
        .bind()
        .arg_val(&expected)
        .invoke_scoped(&mut store_a, |results| {
            let res = prepared_c
                .bind()
                .arg(ValSource::ListFlat(results.view(0)?))
                .invoke(&mut store_c, &mut out_c);
            Ok(res.unwrap_err())
        })?;
    assert!(
        err.to_string().contains("element type"),
        "unexpected error: {err}"
    );

    Ok(())
}
