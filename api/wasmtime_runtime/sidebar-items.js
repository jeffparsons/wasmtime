initSidebarItems({"constant":[["VERSION","Version number of this crate."]],"enum":[["Export","The value of an export passed from one instance to another."],["InstantiationError","An error while instantiating a module."],["Memory","Representation of a runtime wasm linear memory."],["PoolingAllocationStrategy","The allocation strategy to use for the pooling instance allocator."],["Table","Represents an instance’s table."],["TableElement","An element going into or coming out of a table."],["Trap","Stores trace message with backtrace."]],"fn":[["catch_traps","Catches any wasm traps that happen within the execution of `closure`, returning them as a `Result`."],["gc","Perform garbage collection of `VMExternRef`s."],["init_traps","This function is required to be called before any WebAssembly is entered. This will configure global state such as signal handlers to prepare the process to receive wasm traps."],["raise_lib_trap","Raises a trap from inside library code immediately."],["raise_user_trap","Raises a user-defined trap immediately."],["resume_panic","Carries a Rust panic across wasm code and resumes the panic on the other side."],["tls_eager_initialize","Eagerly initialize thread-local runtime functionality. This will be performed lazily by the runtime if users do not perform it eagerly."]],"mod":[["libcalls","Runtime library calls."]],"struct":[["CompiledModuleId","A unique identifier (within an engine or similar) for a compiled module."],["CompiledModuleIdAllocator","An allocator for compiled module IDs."],["DefaultMemoryCreator","A default memory allocator used by Wasmtime"],["ExportFunction","A function export value."],["ExportGlobal","A global export value."],["ExportMemory","A memory export value."],["ExportTable","A table export value."],["GdbJitImageRegistration","Registeration for JIT image"],["Imports","Resolved import pointers."],["InstanceAllocationRequest","Represents a request for a new runtime instance."],["InstanceHandle","A handle holding an `Instance` of a WebAssembly module."],["InstanceLimits","Represents the limits placed on instances by the pooling instance allocator."],["LinkError","An link error while instantiating a module."],["MemoryImage","One backing image for one memory."],["MemoryImageSlot","A single slot handled by the copy-on-write memory initialization mechanism."],["Mmap","A simple struct consisting of a page-aligned pointer to page-aligned and initially-zeroed memory and a length."],["MmapVec","A type akin to `Vec<u8>`, but backed by `mmap` and able to be split."],["ModuleMemoryImages","Backing images for memories in a module."],["OnDemandInstanceAllocator","Represents the on-demand instance allocator."],["PoolingInstanceAllocator","Implements the pooling instance allocator."],["StorePtr","A pointer to a Store. This Option<*mut dyn Store> is wrapped in a struct so that the function to create a &mut dyn Store is a method on a member of InstanceAllocationRequest, rather than on a &mut InstanceAllocationRequest itself, because several use-sites require a split mut borrow on the InstanceAllocationRequest."],["TlsRestore","Opaque state used to help control TLS state across stack switches for async support."],["VMCallerCheckedAnyfunc","The VM caller-checked “anyfunc” record, for caller-side signature checking. It consists of the actual function pointer and a signature id to be checked by the caller."],["VMContext","The VM “context”, which is pointed to by the `vmctx` arg in Cranelift. This has information about globals, memories, tables, and other runtime state associated with the current instance."],["VMExternRef","An external reference to some opaque data."],["VMExternRefActivationsTable","A table that over-approximizes the set of `VMExternRef`s that any Wasm activation on this thread is currently using."],["VMFunctionBody","A placeholder byte-sized type which is just used to provide some amount of type safety when dealing with pointers to JIT-compiled function bodies. Note that it’s deliberately not Copy, as we shouldn’t be carelessly copying function body bytes around."],["VMFunctionImport","An imported function."],["VMGlobalDefinition","The storage for a WebAssembly global defined within the instance."],["VMGlobalImport","The fields compiled code needs to access to utilize a WebAssembly global variable imported from another instance."],["VMInterrupts","Structure used to control interrupting wasm code."],["VMInvokeArgument","The storage for a WebAssembly invocation argument"],["VMMemoryDefinition","The fields compiled code needs to access to utilize a WebAssembly linear memory defined within the instance, namely the start address and the size in bytes."],["VMMemoryImport","The fields compiled code needs to access to utilize a WebAssembly linear memory imported from another instance."],["VMSharedSignatureIndex","An index into the shared signature registry, usable for checking signatures at indirect calls."],["VMTableDefinition","The fields compiled code needs to access to utilize a WebAssembly table defined within the instance."],["VMTableImport","The fields compiled code needs to access to utilize a WebAssembly table imported from another instance."]],"trait":[["InstanceAllocator","Represents a runtime instance allocator."],["ModuleInfo","Used by the runtime to query module information."],["ModuleInfoLookup","Used by the runtime to lookup information about a module given a program counter value."],["ModuleRuntimeInfo","Functionality required by this crate for a particular module. This is chiefly needed for lazy initialization of various bits of instance state."],["RuntimeLinearMemory","A linear memory"],["RuntimeMemoryCreator","A memory allocator"],["Store","Dynamic runtime functionality needed by this crate throughout the execution of a wasm instance."]],"type":[["SignalHandler","Function which may handle custom signals while processing traps."],["VMTrampoline","Trampoline function pointer type."]],"union":[["ValRaw","A “raw” and unsafe representation of a WebAssembly value."]]});