// TODO: Implement aarch64 stack switching trampoline.
// See x86_64.rs for the reference implementation.

use core::arch::naked_asm;

#[inline(never)]
pub fn wasmtime_continuation_start_address() -> *const () {
    wasmtime_continuation_start as *const ()
}

#[unsafe(naked)]
pub(crate) unsafe extern "C" fn wasmtime_continuation_start() {
    naked_asm!("udf #0");
}
