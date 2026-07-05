#![cfg(not(miri))]

//! Tests for the `Func::prepare_call` / `PreparedCall` dynamic calling path,
//! in particular the bulk (`ArgSource::Flat`) argument representation.

use super::{engine, make_echo_component, make_echo_component_with_params};
use wasmtime::component::{ArgSource, ArgSpec, Component, Linker, Val};
use wasmtime::{Result, Store};
use wasmtime_component_util::REALLOC_AND_FREE;

fn u32_list_bytes(vals: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// A `list<u32>` provided as pre-encoded canonical bytes round-trips through an
/// echo component and matches both the expected value and the fully-dynamic
/// `Val` path.
#[test]
fn flat_list_u32() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    // The reflection gate that authorizes the fast path. `u32` is both inline
    // and free of invalid bit patterns.
    let ty = func.ty(&store);
    let element = ty.params().next().unwrap().1.unwrap_list().ty();
    assert!(element.is_cabi_inline());
    assert!(element.are_all_bit_patterns_valid());
    drop(ty);

    let values = [32343u32, 79023439, 2084037802];
    let expected = Val::List(values.iter().copied().map(Val::U32).collect());

    let prepared = func.prepare_call(&store, &[ArgSpec::Flat])?;

    // The buffer is only borrowed for the duration of each `invoke`, so it can
    // be reused/mutated across invocations of one `PreparedCall`.
    let mut buffer = u32_list_bytes(&values);
    let mut output = [Val::Bool(false)];
    prepared
        .bind()
        .arg(ArgSource::Flat(&buffer))
        .invoke(&mut store, &mut output)?;
    assert_eq!(output[0], expected);

    // Reuse the prepared shape with a freshly-mutated buffer.
    let values2 = [1u32, 2, 3, 4];
    buffer = u32_list_bytes(&values2);
    prepared
        .bind()
        .arg(ArgSource::Flat(&buffer))
        .invoke(&mut store, &mut output)?;
    assert_eq!(
        output[0],
        Val::List(values2.iter().copied().map(Val::U32).collect())
    );

    // Identical to the fully-dynamic `Val` path.
    let mut output_val = [Val::Bool(false)];
    func.call(&mut store, &[expected.clone()], &mut output_val)?;
    let first = Val::List(values.iter().copied().map(Val::U32).collect());
    let mut output_flat = [Val::Bool(false)];
    func.prepare_call(&store, &[ArgSpec::Flat])?
        .bind()
        .arg(ArgSource::Flat(&u32_list_bytes(&values)))
        .invoke(&mut store, &mut output_flat)?;
    assert_eq!(output_flat[0], first);
    assert_eq!(output_flat, output_val);

    Ok(())
}

/// A `list<record>` of an inline (POD) record round-trips as flat bytes.
#[test]
fn flat_list_record() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    // A list of a *named* record type (an anonymous record nested in a list
    // can't be used as an exported component type). The list still lowers to a
    // (ptr, len) pair, so the echo body just copies those two core values.
    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"
            (type $R' (record (field "x" float32) (field "y" float32)))
            (export $R "r" (type $R'))
            (type $Foo' (list $R))"#,
            &[
                super::Param(super::Type::I32, Some(0)),
                super::Param(super::Type::I32, Some(4)),
            ],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    let ty = func.ty(&store);
    assert!(ty
        .params()
        .next()
        .unwrap()
        .1
        .unwrap_list()
        .ty()
        .are_all_bit_patterns_valid());
    drop(ty);

    let points = [(1.0f32, 2.0f32), (3.5, -4.25), (-0.0, 100.0)];
    let mut bytes = Vec::new();
    for (x, y) in points {
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
    }
    let expected = Val::List(
        points
            .iter()
            .map(|(x, y)| {
                Val::Record(vec![
                    ("x".to_string(), Val::Float32(*x)),
                    ("y".to_string(), Val::Float32(*y)),
                ])
            })
            .collect(),
    );

    let mut output = [Val::Bool(false)];
    func.prepare_call(&store, &[ArgSpec::Flat])?
        .bind()
        .arg(ArgSource::Flat(&bytes))
        .invoke(&mut store, &mut output)?;
    assert_eq!(output[0], expected);

    // Same via the `Val` path.
    let mut output_val = [Val::Bool(false)];
    func.call(&mut store, &[expected.clone()], &mut output_val)?;
    assert_eq!(output, output_val);

    Ok(())
}

/// One call mixes a `Flat` list argument with a dynamic `Val` argument.
#[test]
fn mixed_flat_and_val() -> Result<()> {
    let component = format!(
        r#"
        (component
            (core module $m
                (func (export "run") (param i32 i32 i32) (result i32)
                    (local $base i32)
                    (local.set $base
                        (call $realloc (i32.const 0) (i32.const 0) (i32.const 4) (i32.const 8)))
                    (i32.store offset=0 (local.get $base) (local.get 0))
                    (i32.store offset=4 (local.get $base) (local.get 1))
                    (local.get $base)
                )
                (memory (export "memory") 1)
                {REALLOC_AND_FREE}
            )
            (core instance $i (instantiate $m))
            (type $L (list u32))
            (func (export "run") (param "a" $L) (param "b" u32) (result $L)
                (canon lift
                    (core func $i "run")
                    (memory $i "memory")
                    (realloc (func $i "realloc"))
                )
            )
        )"#
    );

    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, component)?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "run").unwrap();

    let values = [10u32, 20, 30];
    let mut output = [Val::Bool(false)];
    func.prepare_call(&store, &[ArgSpec::Flat, ArgSpec::Val])?
        .bind()
        .arg(ArgSource::Flat(&u32_list_bytes(&values)))
        .arg(ArgSource::Val(Val::U32(42)))
        .invoke(&mut store, &mut output)?;

    assert_eq!(
        output[0],
        Val::List(values.iter().copied().map(Val::U32).collect())
    );
    Ok(())
}

/// When the parameter tuple is too large to pass flat, arguments are stored
/// behind a single pointer; a `Flat` list argument must work there too.
#[test]
fn flat_list_indirect_params() -> Result<()> {
    // `a: list<u32>` flattens to 2 core values and `b: tuple<15 x u32>` to 15,
    // exceeding MAX_FLAT_PARAMS (16), so the parameters are passed indirectly.
    let component = format!(
        r#"
        (component
            (core module $m
                (func (export "run") (param i32) (result i32)
                    (local $base i32)
                    (local.set $base
                        (call $realloc (i32.const 0) (i32.const 0) (i32.const 4) (i32.const 8)))
                    (i32.store offset=0 (local.get $base) (i32.load offset=0 (local.get 0)))
                    (i32.store offset=4 (local.get $base) (i32.load offset=4 (local.get 0)))
                    (local.get $base)
                )
                (memory (export "memory") 1)
                {REALLOC_AND_FREE}
            )
            (core instance $i (instantiate $m))
            (type $L (list u32))
            (type $T (tuple u32 u32 u32 u32 u32 u32 u32 u32 u32 u32 u32 u32 u32 u32 u32))
            (func (export "run") (param "a" $L) (param "b" $T) (result $L)
                (canon lift
                    (core func $i "run")
                    (memory $i "memory")
                    (realloc (func $i "realloc"))
                )
            )
        )"#
    );

    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, component)?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "run").unwrap();

    let values = [7u32, 8, 9, 10];
    let tuple = Val::Tuple((0..15).map(Val::U32).collect());
    let mut output = [Val::Bool(false)];
    func.prepare_call(&store, &[ArgSpec::Flat, ArgSpec::Val])?
        .bind()
        .arg(ArgSource::Flat(&u32_list_bytes(&values)))
        .arg(ArgSource::Val(tuple))
        .invoke(&mut store, &mut output)?;

    assert_eq!(
        output[0],
        Val::List(values.iter().copied().map(Val::U32).collect())
    );
    Ok(())
}

/// `prepare_call` rejects `ArgSpec::Flat` on a `list` whose element type has
/// out-of-line storage or invalid bit patterns, and on a parameter that isn't a
/// list at all.
#[test]
fn flat_rejects_non_inline() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());

    // list<string>: element is not inline (out-of-line storage).
    let component = Component::new(&engine, make_echo_component("(list string)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let element = func
        .ty(&store)
        .params()
        .next()
        .unwrap()
        .1
        .unwrap_list()
        .ty();
    assert!(!element.is_cabi_inline());
    assert!(!element.are_all_bit_patterns_valid());
    let err = func.prepare_call(&store, &[ArgSpec::Flat]).unwrap_err();
    assert!(
        err.to_string().contains("fixed canonical-ABI layout"),
        "unexpected error: {err}"
    );

    // list<bool>: the element *is* inline (fixed layout) but has invalid bit
    // patterns, so the unchecked `Flat` path must still reject it.
    let component = Component::new(&engine, make_echo_component("(list bool)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let element = func
        .ty(&store)
        .params()
        .next()
        .unwrap()
        .1
        .unwrap_list()
        .ty();
    assert!(element.is_cabi_inline());
    assert!(!element.are_all_bit_patterns_valid());
    let err = func.prepare_call(&store, &[ArgSpec::Flat]).unwrap_err();
    assert!(
        err.to_string().contains("invalid bit patterns"),
        "unexpected error: {err}"
    );

    // A non-list parameter.
    let component = Component::new(
        &engine,
        make_echo_component_with_params("u32", &[super::Param(super::Type::I32, Some(0))]),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let err = func.prepare_call(&store, &[ArgSpec::Flat]).unwrap_err();
    assert!(
        err.to_string().contains("requires a `list` parameter"),
        "unexpected error: {err}"
    );

    // Arity mismatch is caught at prepare time.
    let err = func
        .prepare_call(&store, &[ArgSpec::Val, ArgSpec::Val])
        .unwrap_err();
    assert!(
        err.to_string().contains("argument spec"),
        "unexpected error: {err}"
    );

    Ok(())
}

/// A `Flat` byte buffer whose length isn't a whole number of elements is a
/// recoverable error at `invoke` time.
#[test]
fn flat_ragged_length_errors() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    let prepared = func.prepare_call(&store, &[ArgSpec::Flat])?;
    let mut output = [Val::Bool(false)];
    let err = prepared
        .bind()
        .arg(ArgSource::Flat(&[0u8; 5]))
        .invoke(&mut store, &mut output)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("not a multiple of the element size"),
        "unexpected error: {err}"
    );
    Ok(())
}

fn read_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `invoke_scoped` exposes a `list<u32>` result as a zero-copy view, an owned
/// copy, and a `Val` — the same result read three ways in one scope.
#[test]
fn scoped_read_list() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    let values = [5u32, 6, 7, 8, 9];
    let prepared = func.prepare_call(&store, &[ArgSpec::Flat])?;

    let seen = prepared
        .bind()
        .arg(ArgSource::Flat(&u32_list_bytes(&values)))
        .invoke_scoped(&mut store, |results| {
            assert_eq!(results.len(), 1);

            // Zero-copy view of the guest's canonical bytes.
            let view = results.view(0)?;
            assert_eq!(read_u32s(view), values);

            // An owned copy matches the view.
            assert_eq!(results.copy(0)?, view);

            // The same result lifted as a `Val` (the view is still valid: it
            // borrows guest memory for the whole scope, not the accessor).
            let val = results.val(0)?;
            assert_eq!(
                val,
                Val::List(values.iter().copied().map(Val::U32).collect())
            );
            assert_eq!(read_u32s(view), values);

            Ok(read_u32s(view))
        })?;

    assert_eq!(seen, values);
    Ok(())
}

/// `view` is rejected for a non-list (here scalar) result, while `val` works.
#[test]
fn scoped_view_rejects_non_list() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(
        &engine,
        make_echo_component_with_params("u32", &[super::Param(super::Type::I32, Some(0))]),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    func.prepare_call(&store, &[ArgSpec::Val])?
        .bind()
        .arg(ArgSource::Val(Val::U32(1234)))
        .invoke_scoped(&mut store, |results| {
            let err = results.view(0).unwrap_err();
            assert!(err.to_string().contains("is not a list"), "{err}");
            assert_eq!(results.val(0)?, Val::U32(1234));
            Ok(())
        })?;
    Ok(())
}

/// `val` reads a result returned indirectly whose value is a tuple containing a
/// list, exercising the memory-load path.
#[test]
fn scoped_val_tuple_with_list() -> Result<()> {
    // Returns `tuple<list<u32>, u32>` (one result, returned via a pointer). The
    // core body echoes the incoming list and appends a constant scalar.
    let component = format!(
        r#"
        (component
            (core module $m
                (func (export "run") (param i32 i32) (result i32)
                    (local $base i32)
                    (local.set $base
                        (call $realloc (i32.const 0) (i32.const 0) (i32.const 4) (i32.const 12)))
                    (i32.store offset=0 (local.get $base) (local.get 0))
                    (i32.store offset=4 (local.get $base) (local.get 1))
                    (i32.store offset=8 (local.get $base) (i32.const 99))
                    (local.get $base)
                )
                (memory (export "memory") 1)
                {REALLOC_AND_FREE}
            )
            (core instance $i (instantiate $m))
            (type $L (list u32))
            (type $T (tuple $L u32))
            (func (export "run") (param "a" $L) (result $T)
                (canon lift
                    (core func $i "run")
                    (memory $i "memory")
                    (realloc (func $i "realloc"))
                )
            )
        )"#
    );

    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, component)?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "run").unwrap();

    let values = [1u32, 2, 3];
    func.prepare_call(&store, &[ArgSpec::Flat])?
        .bind()
        .arg(ArgSource::Flat(&u32_list_bytes(&values)))
        .invoke_scoped(&mut store, |results| {
            assert_eq!(
                results.val(0)?,
                Val::Tuple(vec![
                    Val::List(values.iter().copied().map(Val::U32).collect()),
                    Val::U32(99),
                ])
            );
            Ok(())
        })?;
    Ok(())
}
