use anyhow::Result;
use wasmtime::{
    component::{Component, Linker},
    Config, Engine, Store,
};
use wasmtime_wasi::preview2::bindings::clocks::wall_clock;
use wasmtime_wasi::preview2::bindings::filesystem::types as filesystem;
use wasmtime_wasi::preview2::{self, Table, WasiCtx, WasiCtxBuilder, WasiView};

lazy_static::lazy_static! {
    static ref ENGINE: Engine = {
        let mut config = Config::new();
        config.wasm_backtrace_details(wasmtime::WasmBacktraceDetails::Enable);
        config.wasm_component_model(true);
        config.async_support(true);

        let engine = Engine::new(&config).unwrap();
        engine
    };
}

// uses ENGINE, creates a fn get_component(&str) -> Component
include!(concat!(env!("OUT_DIR"), "/reactor_tests_components.rs"));

wasmtime::component::bindgen!({
    path: "../wasi/wit",
    world: "test-reactor",
    async: true,
    with: {
       "wasi:io/streams": preview2::bindings::io::streams,
       "wasi:filesystem/types": preview2::bindings::filesystem::types,
       "wasi:filesystem/preopens": preview2::bindings::filesystem::preopens,
       "wasi:cli/environment": preview2::bindings::cli::environment,
       "wasi:cli/exit": preview2::bindings::cli::exit,
       "wasi:cli/stdin": preview2::bindings::cli::stdin,
       "wasi:cli/stdout": preview2::bindings::cli::stdout,
       "wasi:cli/stderr": preview2::bindings::cli::stderr,
       "wasi:cli/terminal_input": preview2::bindings::cli::terminal_input,
       "wasi:cli/terminal_output": preview2::bindings::cli::terminal_output,
       "wasi:cli/terminal_stdin": preview2::bindings::cli::terminal_stdin,
       "wasi:cli/terminal_stdout": preview2::bindings::cli::terminal_stdout,
       "wasi:cli/terminal_stderr": preview2::bindings::cli::terminal_stderr,
    },
    ownership: Borrowing {
        duplicate_if_necessary: false
    }
});

struct ReactorCtx {
    table: Table,
    wasi: WasiCtx,
}

impl WasiView for ReactorCtx {
    fn table(&self) -> &Table {
        &self.table
    }
    fn table_mut(&mut self) -> &mut Table {
        &mut self.table
    }
    fn ctx(&self) -> &WasiCtx {
        &self.wasi
    }
    fn ctx_mut(&mut self) -> &mut WasiCtx {
        &mut self.wasi
    }
}

async fn instantiate(
    component: Component,
    wasi_ctx: ReactorCtx,
) -> Result<(Store<ReactorCtx>, TestReactor)> {
    let mut linker = Linker::new(&ENGINE);
    preview2::command::add_to_linker(&mut linker)?;

    let mut store = Store::new(&ENGINE, wasi_ctx);
    let (testreactor, _instance) =
        TestReactor::instantiate_async(&mut store, &component, &linker).await?;
    Ok((store, testreactor))
}

#[test_log::test(tokio::test)]
async fn reactor_tests() -> Result<()> {
    let table = Table::new();
    let wasi = WasiCtxBuilder::new().env("GOOD_DOG", "gussie").build();

    let (mut store, reactor) =
        instantiate(get_component("reactor_tests"), ReactorCtx { table, wasi }).await?;

    // Show that integration with the WASI context is working - the guest will
    // interpolate $GOOD_DOG to gussie here using the environment:
    let r = reactor
        .call_add_strings(&mut store, &["hello", "$GOOD_DOG"])
        .await?;
    assert_eq!(r, 2);

    let contents = reactor.call_get_strings(&mut store).await?;
    assert_eq!(contents, &["hello", "gussie"]);

    // Show that we can pass in a resource type whose impls are defined in the
    // `host` and `wasi-common` crate.
    // Note, this works because of the add_to_linker invocations using the
    // `host` crate for `streams`, not because of `with` in the bindgen macro.
    let writepipe = preview2::pipe::MemoryOutputPipe::new(4096);
    let stream: preview2::OutputStream = Box::new(writepipe.clone());
    let table_ix = store.data_mut().table_mut().push_resource(stream)?;
    let r = reactor.call_write_strings_to(&mut store, table_ix).await?;
    assert_eq!(r, Ok(()));

    assert_eq!(writepipe.contents().as_ref(), b"hellogussie");

    // Show that the `with` invocation in the macro means we get to re-use the
    // type definitions from inside the `host` crate for these structures:
    let ds = filesystem::DescriptorStat {
        data_access_timestamp: Some(wall_clock::Datetime {
            nanoseconds: 123,
            seconds: 45,
        }),
        data_modification_timestamp: Some(wall_clock::Datetime {
            nanoseconds: 789,
            seconds: 10,
        }),
        link_count: 0,
        size: 0,
        status_change_timestamp: Some(wall_clock::Datetime {
            nanoseconds: 0,
            seconds: 1,
        }),
        type_: filesystem::DescriptorType::Unknown,
    };
    let expected = format!("{ds:?}");
    let got = reactor.call_pass_an_imported_record(&mut store, ds).await?;
    assert_eq!(expected, got);

    Ok(())
}
