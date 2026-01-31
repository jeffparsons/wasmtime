//! Smoke tests for WebAssembly continuation/stack-switching functionality.

use wasmtime::*;

#[test]
#[cfg_attr(miri, ignore)]
fn continuation_roundtrip() -> Result<()> {
    // Exercises the full continuation lifecycle:
    // 1. Create continuation from a function
    // 2. Resume → function suspends → get continuation back
    // 3. Resume again → function completes normally
    let source = r#"
        (module
          (type $ft (func))
          (tag $t)
          (type $ct (cont $ft))

          ;; Function that suspends once, then returns
          (func $target (suspend $t))
          (elem declare func $target)

          (func (export "roundtrip") (result i32)
            (local $k (ref $ct))

            ;; First resume: should suspend
            (local.set $k
              (block $on_suspend1 (result (ref $ct))
                (resume $ct
                  (on $t $on_suspend1)
                  (cont.new $ct (ref.func $target)))
                ;; Completed without suspending (unexpected)
                (return (i32.const 0))
              )
            )

            ;; Second resume: should complete normally
            (block $on_suspend2 (result (ref $ct))
              (resume $ct (on $t $on_suspend2) (local.get $k))
              ;; Completed normally - success
              (return (i32.const 1))
            )

            ;; Suspended again (unexpected)
            (drop)
            (i32.const 0)
          )
        )
    "#;

    let mut config = Config::new();
    config.wasm_exceptions(true).wasm_stack_switching(true);
    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, ());
    let module = Module::new(&engine, source)?;

    let instance = Instance::new(&mut store, &module, &[])?;
    let roundtrip = instance.get_typed_func::<(), i32>(&mut store, "roundtrip")?;
    let result = roundtrip.call(&mut store, ())?;

    assert_eq!(result, 1, "continuation should suspend then complete");

    Ok(())
}
