#![cfg(not(miri))]

//! Tests for the `Func::prepare_call` / `PreparedCall` surface: dynamic calls
//! in which each argument chooses its lowering strategy ([`ValSpec`] /
//! [`ValSource`]), in particular the bulk `ListFlat` strategy fed by
//! [`ValidatedCabiBytes`].

use super::{engine, make_echo_component, make_echo_component_with_params};
use wasmtime::component::{
    Component, Linker, Type, Val, ValSource, ValSpec, ValidatedCabiBytes,
};
use wasmtime::{Result, Store};
use wasmtime_component_util::REALLOC_AND_FREE;

fn u32_list_bytes(vals: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// The single list parameter's element type, reflected off `func`.
fn param_elem(store: &Store<()>, func: &wasmtime::component::Func, index: usize) -> Type {
    func.ty(store).params().nth(index).unwrap().1.unwrap_list().ty()
}

/// A `list<u32>` provided as pre-validated canonical bytes round-trips through
/// an echo component and matches both the expected value and the
/// fully-dynamic `Val` path.
#[test]
fn flat_list_u32() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    let element = param_elem(&store, &func, 0);
    assert!(element.is_cabi_inline());
    assert!(element.are_all_bit_patterns_valid());

    let values = [32343u32, 79023439, 2084037802];
    let expected = Val::List(values.iter().copied().map(Val::U32).collect());

    let prepared = func.prepare_call(&store, &[ValSpec::ListFlat])?;

    // The buffer is only borrowed for the duration of each `invoke`, so it
    // can be reused/mutated across invocations of one `PreparedCall`.
    let mut buffer = u32_list_bytes(&values);
    let mut output = [Val::Bool(false)];
    prepared
        .bind()
        .arg(ValSource::ListFlat(ValidatedCabiBytes::checked(
            &buffer, &element,
        )?))
        .invoke(&mut store, &mut output)?;
    assert_eq!(output[0], expected);

    // Reuse the prepared shape with a freshly-mutated buffer.
    let values2 = [1u32, 2, 3, 4];
    buffer = u32_list_bytes(&values2);
    prepared
        .bind()
        .arg(ValSource::ListFlat(ValidatedCabiBytes::checked(
            &buffer, &element,
        )?))
        .invoke(&mut store, &mut output)?;
    assert_eq!(
        output[0],
        Val::List(values2.iter().copied().map(Val::U32).collect())
    );

    // Identical to the fully-dynamic `Val` path.
    let mut output_val = [Val::Bool(false)];
    func.call(&mut store, &[expected.clone()], &mut output_val)?;
    assert_eq!(output_val[0], expected);

    Ok(())
}

/// A `list<record>` of an inline (POD) record round-trips as flat bytes.
#[test]
fn flat_list_record() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    // A list of a *named* record type (an anonymous record nested in a list
    // can't be used as an exported component type). The list still lowers to
    // a (ptr, len) pair, so the echo body just copies those two core values.
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

    let element = param_elem(&store, &func, 0);
    assert!(element.are_all_bit_patterns_valid());

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
    func.prepare_call(&store, &[ValSpec::ListFlat])?
        .bind()
        .arg(ValSource::ListFlat(ValidatedCabiBytes::checked(
            &bytes, &element,
        )?))
        .invoke(&mut store, &mut output)?;
    assert_eq!(output[0], expected);

    // Same via the `Val` path.
    let mut output_val = [Val::Bool(false)];
    func.call(&mut store, &[expected.clone()], &mut output_val)?;
    assert_eq!(output, output_val);

    Ok(())
}

/// A `list<enum>` — inline but *not* bit-pattern-total — works through the
/// bulk path because validation was paid when the proof was minted, not
/// skipped. (The v1 sketch of this API had to reject `list<enum>` outright;
/// the proof-carrying wrapper is what makes it sound to accept.)
#[test]
fn flat_list_enum() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"
            (type $E' (enum "a" "b" "c"))
            (export $E "e" (type $E'))
            (type $Foo' (list $E))"#,
            &[
                super::Param(super::Type::I32, Some(0)),
                super::Param(super::Type::I32, Some(4)),
            ],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    let element = param_elem(&store, &func, 0);
    assert!(element.is_cabi_inline());
    assert!(!element.are_all_bit_patterns_valid());

    // Preparation succeeds: inline is enough, totality is the proof's job.
    let prepared = func.prepare_call(&store, &[ValSpec::ListFlat])?;

    // Valid discriminants: the sweep accepts, the call round-trips.
    let bytes = [0u8, 2, 1, 1];
    let mut output = [Val::Bool(false)];
    prepared
        .bind()
        .arg(ValSource::ListFlat(ValidatedCabiBytes::checked(
            &bytes, &element,
        )?))
        .invoke(&mut store, &mut output)?;
    let expected = Val::List(vec![
        Val::Enum("a".to_string()),
        Val::Enum("c".to_string()),
        Val::Enum("b".to_string()),
        Val::Enum("b".to_string()),
    ]);
    assert_eq!(output[0], expected);

    // An out-of-range discriminant can never reach the memcpy: minting the
    // proof fails.
    assert!(ValidatedCabiBytes::checked(&[3u8], &element).is_err());

    Ok(())
}

/// One call mixes a `ListFlat` argument with a dynamic `Val` argument.
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

    let element = param_elem(&store, &func, 0);
    let values = [10u32, 20, 30];
    let bytes = u32_list_bytes(&values);
    let b = Val::U32(42);
    let mut output = [Val::Bool(false)];
    func.prepare_call(&store, &[ValSpec::ListFlat, ValSpec::Val])?
        .bind()
        .arg(ValSource::ListFlat(ValidatedCabiBytes::checked(
            &bytes, &element,
        )?))
        .arg_val(&b)
        .invoke(&mut store, &mut output)?;

    assert_eq!(
        output[0],
        Val::List(values.iter().copied().map(Val::U32).collect())
    );
    Ok(())
}

/// When the parameter tuple is too large to pass flat, arguments are stored
/// behind a single pointer; a `ListFlat` argument must work there too.
#[test]
fn flat_list_indirect_params() -> Result<()> {
    // `a: list<u32>` flattens to 2 core values and `b: tuple<15 x u32>` to
    // 15, exceeding MAX_FLAT_PARAMS (16), so the parameters are passed
    // indirectly.
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

    let element = param_elem(&store, &func, 0);
    let values = [7u32, 8, 9, 10];
    let bytes = u32_list_bytes(&values);
    let tuple = Val::Tuple((0..15).map(Val::U32).collect());
    let mut output = [Val::Bool(false)];
    func.prepare_call(&store, &[ValSpec::ListFlat, ValSpec::Val])?
        .bind()
        .arg(ValSource::ListFlat(ValidatedCabiBytes::checked(
            &bytes, &element,
        )?))
        .arg_val(&tuple)
        .invoke(&mut store, &mut output)?;

    assert_eq!(
        output[0],
        Val::List(values.iter().copied().map(Val::U32).collect())
    );
    Ok(())
}

/// `prepare_call` rejects `ValSpec::ListFlat` on a `list` whose element type
/// is not inline, and on a parameter that isn't a list at all; arity
/// mismatches are also caught at prepare time.
#[test]
fn prepare_rejections() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());

    // list<string>: element is not inline (out-of-line storage).
    let component = Component::new(&engine, make_echo_component("(list string)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let err = func.prepare_call(&store, &[ValSpec::ListFlat]).unwrap_err();
    assert!(
        err.to_string().contains("inline canonical-ABI layout"),
        "unexpected error: {err}"
    );

    // A non-list parameter.
    let component = Component::new(
        &engine,
        make_echo_component_with_params("u32", &[super::Param(super::Type::I32, Some(0))]),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let err = func.prepare_call(&store, &[ValSpec::ListFlat]).unwrap_err();
    assert!(
        err.to_string().contains("requires a `list` parameter"),
        "unexpected error: {err}"
    );

    // Arity mismatch is caught at prepare time.
    let err = func
        .prepare_call(&store, &[ValSpec::Val, ValSpec::Val])
        .unwrap_err();
    assert!(
        err.to_string().contains("argument spec"),
        "unexpected error: {err}"
    );

    Ok(())
}

/// A proof minted for one element type cannot be fed to a parameter with a
/// different element type: the structural type check at invoke rejects it.
#[test]
fn proof_type_mismatch_errors() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());

    // Mint a proof for `u32` elements...
    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func_u32 = instance.get_func(&mut store, "echo").unwrap();
    let u32_elem = param_elem(&store, &func_u32, 0);

    // ...and try to feed it to a `list<float32>` parameter.
    let component = Component::new(&engine, make_echo_component("(list float32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func_f32 = instance.get_func(&mut store, "echo").unwrap();

    let bytes = u32_list_bytes(&[1, 2, 3]);
    let proof = ValidatedCabiBytes::checked(&bytes, &u32_elem)?;
    let mut output = [Val::Bool(false)];
    let err = func_f32
        .prepare_call(&store, &[ValSpec::ListFlat])?
        .bind()
        .arg(ValSource::ListFlat(proof))
        .invoke(&mut store, &mut output)
        .unwrap_err();
    assert!(
        err.to_string().contains("element type"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// Binding a source whose variant doesn't match the prepared spec is a clean
/// error before the guest runs.
#[test]
fn bind_wrong_source_variant_errors() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    // Prepared as ListFlat, bound as Val.
    let prepared = func.prepare_call(&store, &[ValSpec::ListFlat])?;
    let v = Val::List(vec![Val::U32(1)]);
    let mut output = [Val::Bool(false)];
    let err = prepared
        .bind()
        .arg_val(&v)
        .invoke(&mut store, &mut output)
        .unwrap_err();
    assert!(
        err.to_string().contains("does not match the prepared"),
        "unexpected error: {err}"
    );

    // Prepared as Val, bound as ListFlat.
    let element = param_elem(&store, &func, 0);
    let bytes = u32_list_bytes(&[1]);
    let prepared = func.prepare_call(&store, &[ValSpec::Val])?;
    let err = prepared
        .bind()
        .arg(ValSource::ListFlat(ValidatedCabiBytes::checked(
            &bytes, &element,
        )?))
        .invoke(&mut store, &mut output)
        .unwrap_err();
    assert!(
        err.to_string().contains("does not match the prepared"),
        "unexpected error: {err}"
    );

    // Too few arguments bound.
    let prepared = func.prepare_call(&store, &[ValSpec::Val])?;
    let err = prepared.bind().invoke(&mut store, &mut output).unwrap_err();
    assert!(
        err.to_string().contains("expected 1 argument"),
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

/// `invoke_scoped` exposes a `list<u32>` result as a zero-copy proof-carrying
/// view, an owned validated copy, and a `Val` — the same result read three
/// ways in one scope.
#[test]
fn scoped_read_list() -> Result<()> {
    let engine = engine();
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    let element = param_elem(&store, &func, 0);
    let values = [5u32, 6, 7, 8, 9];
    let bytes = u32_list_bytes(&values);
    let prepared = func.prepare_call(&store, &[ValSpec::ListFlat])?;

    let seen = prepared
        .bind()
        .arg(ValSource::ListFlat(ValidatedCabiBytes::checked(
            &bytes, &element,
        )?))
        .invoke_scoped(&mut store, |results| {
            assert_eq!(results.len(), 1);

            // Zero-copy proof-carrying view of the guest's canonical bytes.
            let view = results.view(0)?;
            assert_eq!(read_u32s(view.bytes()), values);
            assert_eq!(view.len(), values.len());

            // An owned validated copy matches the view.
            assert_eq!(results.copy(0)?.bytes(), view.bytes());

            // The same result lifted as a `Val` (the view is still valid: it
            // borrows guest memory for the whole scope, not the accessor).
            let val = results.val(0)?;
            assert_eq!(
                val,
                Val::List(values.iter().copied().map(Val::U32).collect())
            );
            assert_eq!(read_u32s(view.bytes()), values);

            Ok(read_u32s(view.bytes()))
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

    let v = Val::U32(1234);
    func.prepare_call(&store, &[ValSpec::Val])?
        .bind()
        .arg_val(&v)
        .invoke_scoped(&mut store, |results| {
            let err = results.view(0).unwrap_err();
            assert!(err.to_string().contains("is not a list"), "{err}");
            assert_eq!(results.val(0)?, Val::U32(1234));
            Ok(())
        })?;
    Ok(())
}

/// `val` reads a result returned indirectly whose value is a tuple containing
/// a list, exercising the memory-load path.
#[test]
fn scoped_val_tuple_with_list() -> Result<()> {
    // Returns `tuple<list<u32>, u32>` (one result, returned via a pointer).
    // The core body echoes the incoming list and appends a constant scalar.
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

    let element = param_elem(&store, &func, 0);
    let values = [1u32, 2, 3];
    let bytes = u32_list_bytes(&values);
    func.prepare_call(&store, &[ValSpec::ListFlat])?
        .bind()
        .arg(ValSource::ListFlat(ValidatedCabiBytes::checked(
            &bytes, &element,
        )?))
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

/// `view` rejects a guest-returned list pointer that is in-bounds but
/// misaligned, matching the checked `Val` path (so a host `&[T]` cast is
/// safe).
#[test]
fn view_rejects_misaligned_list() -> Result<()> {
    // The guest returns a `list<u32>` whose pointer is deliberately offset by
    // a non-multiple of 4 (but still within memory).
    let component = format!(
        r#"
        (component
            (core module $m
                (func (export "run") (result i32)
                    (local $base i32)
                    (local.set $base
                        (call $realloc (i32.const 0) (i32.const 0) (i32.const 4) (i32.const 64)))
                    ;; result block: list pointer = base+9 (misaligned), length = 1
                    (i32.store offset=0 (local.get $base) (i32.add (local.get $base) (i32.const 9)))
                    (i32.store offset=4 (local.get $base) (i32.const 1))
                    (local.get $base)
                )
                (memory (export "memory") 1)
                {REALLOC_AND_FREE}
            )
            (core instance $i (instantiate $m))
            (type $L (list u32))
            (func (export "run") (result $L)
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

    func.prepare_call(&store, &[])?
        .bind()
        .invoke_scoped(&mut store, |results| {
            let err = results.view(0).unwrap_err();
            assert!(err.to_string().contains("aligned"), "view: {err}");
            // The `Val` path rejects the same misaligned pointer.
            let err = results.val(0).unwrap_err();
            assert!(err.to_string().contains("aligned"), "val: {err}");
            Ok(())
        })?;
    Ok(())
}

/// A `list<enum>` result: `view` refuses (validation would be skipped),
/// `view_checked` sweeps and accepts valid guest data — and rejects an
/// out-of-range discriminant an adversarial guest left in memory.
#[test]
fn view_checked_sweeps_guest_results() -> Result<()> {
    // The guest returns `list<enum "a" "b">` with bytes [0, 1, DISC] where
    // DISC is a parameter, letting one component produce both valid and
    // invalid final discriminants.
    let component = format!(
        r#"
        (component
            (core module $m
                (func (export "run") (param i32) (result i32)
                    (local $base i32)
                    (local.set $base
                        (call $realloc (i32.const 0) (i32.const 0) (i32.const 4) (i32.const 16)))
                    ;; element bytes at base+8: [0, 1, DISC]
                    (i32.store8 offset=8 (local.get $base) (i32.const 0))
                    (i32.store8 offset=9 (local.get $base) (i32.const 1))
                    (i32.store8 offset=10 (local.get $base) (local.get 0))
                    ;; result block: (ptr = base+8, len = 3)
                    (i32.store offset=0 (local.get $base) (i32.add (local.get $base) (i32.const 8)))
                    (i32.store offset=4 (local.get $base) (i32.const 3))
                    (local.get $base)
                )
                (memory (export "memory") 1)
                {REALLOC_AND_FREE}
            )
            (core instance $i (instantiate $m))
            (type $E' (enum "a" "b"))
            (export $E "e" (type $E'))
            (type $L (list $E))
            (func (export "run") (param "d" u32) (result $L)
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

    let prepared = func.prepare_call(&store, &[ValSpec::Val])?;

    // Valid data: `view` refuses (enum needs validation), `view_checked`
    // sweeps and hands back the proof.
    let d = Val::U32(1);
    prepared
        .bind()
        .arg_val(&d)
        .invoke_scoped(&mut store, |results| {
            let err = results.view(0).unwrap_err();
            assert!(err.to_string().contains("view_checked"), "{err}");
            let view = results.view_checked(0)?;
            assert_eq!(view.bytes(), &[0, 1, 1]);
            assert_eq!(view.len(), 3);
            Ok(())
        })?;

    // Adversarial data: discriminant 9 is memory-safe to copy but not a valid
    // enum value; the sweep rejects it (and so does the `Val` lift).
    let d = Val::U32(9);
    prepared
        .bind()
        .arg_val(&d)
        .invoke_scoped(&mut store, |results| {
            let err = results.view_checked(0).unwrap_err();
            assert!(
                format!("{err:?}").contains("failed validation"),
                "{err:?}"
            );
            assert!(results.val(0).is_err());
            Ok(())
        })?;

    Ok(())
}
