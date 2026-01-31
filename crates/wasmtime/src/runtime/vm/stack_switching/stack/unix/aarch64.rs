// A WORD OF CAUTION
//
// This entire file basically needs to be kept in sync with itself. It's not
// really possible to modify just one bit of this file without understanding
// all the other bits. Documentation tries to reference various bits here and
// there but try to make sure to read over everything before tweaking things!

use core::arch::naked_asm;

#[inline(never)] // FIXME(rust-lang/rust#148307)
pub fn wasmtime_continuation_start_address() -> *const () {
    wasmtime_continuation_start as *const ()
}

// This is a pretty special function that has no real signature. Its use is to
// be the "base" function of all fibers. This entrypoint is used in
// `wasmtime_continuation_init` to bootstrap the execution of a new fiber.
//
// We also use this function as a persistent frame on the stack to emit dwarf
// information to unwind into the caller. This allows us to unwind from the
// fiber's stack back to the initial stack that the fiber was called from. We use
// special dwarf directives here to do so since this is a pretty nonstandard
// function.
//
// If you're curious a decent introduction to CFI things and unwinding is at
// https://www.imperialviolet.org/2017/01/18/cfi.html
//
// Note that this function is never called directly. It is only ever entered
// when a `stack_switch` instruction loads its address when switching to a stack
// prepared by `FiberStack::initialize`.
//
// Executing `stack_switch` on a stack prepared by `FiberStack::initialize` as
// described in the comment on `FiberStack::initialize` leads to the following
// values in various registers when execution of wasmtime_continuation_start begins:
//
// SP: TOS - 0x40 - (16 * `args_capacity`)
// x29 (FP): TOS - 0x10

#[unsafe(naked)]
pub(crate) unsafe extern "C" fn wasmtime_continuation_start() {
    naked_asm!(
        "
        // ===========================================================================
        // Unwinding and Backtraces
        // ===========================================================================
        //
        // Two distinct mechanisms exist for walking the call stack:
        //
        // - Frame-pointer walking (following x29 chains): WORKS
        // - DWARF/libunwind-based unwinding: DOES NOT WORK
        //
        // This distinction matters: 'backtraces work' here means FP-based walking,
        // not debugger-quality DWARF unwinding.
        //
        // What works:
        // - Continuation backtraces via frame-pointer walking work on macOS/aarch64
        // - The continuation stack is initialized with an FP slot pointing to itself
        // - The trampoline preserves and restores x29 consistently
        // - Control context layout (SP @ 0, FP @ 8, PC @ 16) plus per-field
        //   load/store ordering ensures FP chain validity across suspend/resume
        //
        // What doesn't work:
        // - DWARF/libunwind-based unwinding is not reliable across:
        //   - wasmtime_continuation_start (entered via indirect branch,
        //     nonstandard prologue)
        //   - stack_switch (no Cranelift-emitted unwind metadata)
        // - No CFI is emitted for the trampoline or stack switch
        //
        // macOS-specific expectations:
        // - LLDB backtraces may stop at wasmtime_continuation_start
        // - This is expected behavior given the lack of CFI
        // - Tools that rely purely on DWARF unwind info will produce partial traces
        //
        // Why this is intentional and sufficient:
        // - Design prioritizes correctness and portability of stack switching
        // - FP-based backtraces are sufficient for internal debugging and crash
        //   analysis
        // - This mirrors the current state on x86_64
        // - CFI support is deferred until stack-switching semantics are stable
        //
        // Future work (separate phase):
        // Adding CFI is non-trivial due to:
        // - Indirect branches (trampoline entered via br)
        // - Multiple stacks (continuation vs parent)
        // - SP/FP swaps mid-instruction sequence in stack_switch
        //
        // Trampoline CFI would follow fiber crate's aarch64 patterns. Full solution
        // requires Cranelift machinst unwind directives for StackSwitchBasic.
        //
        // Practical guidance:
        // When debugging continuation issues on macOS/aarch64, prefer FP-based
        // backtraces or explicit logging around suspend/resume boundaries.
        // ===========================================================================

        //
        // Load the 4 arguments for fiber_start from the stack into registers.
        // Stack layout (from SP):
        //   SP + 0x00: return_value_count
        //   SP + 0x08: args
        //   SP + 0x10: caller_vmctx
        //   SP + 0x18: func_ref
        //
        // fiber_start signature: fiber_start(func_ref, caller_vmctx, args, return_value_count)
        // aarch64 calling convention: x0, x1, x2, x3 for first 4 arguments
        //

        ldr x3, [sp, #0x00]  // return_value_count
        ldr x2, [sp, #0x08]  // args
        ldr x1, [sp, #0x10]  // caller_vmctx
        ldr x0, [sp, #0x18]  // func_ref
        add sp, sp, #0x20    // advance past arguments (32 bytes)

        // Note that x29 already contains the right frame pointer to build a
        // frame pointer chain including the parent continuation:
        // The current value of x29 is where we store the parent FP in the
        // control context!
        bl {fiber_start}

        // Return to the parent continuation.
        // x29 is callee-saved, so its value is still TOS - 0x10.
        // Use that fact to obtain saved parent FP, SP, and PC from control
        // context near TOS.
        //
        // Control context layout (relative to x29 = TOS - 0x10):
        //   [x29 + 0x08] = saved instruction pointer (at TOS - 0x08)
        //   [x29 + 0x00] = parent FP (at TOS - 0x10)
        //   [x29 - 0x08] = parent SP (at TOS - 0x18)

        ldr x16, [x29, #0x08]  // load parent PC into scratch register
        ldr x17, [x29, #-0x08] // load parent SP into scratch register
        ldr x29, [x29]         // restore parent FP

        mov sp, x17            // restore parent SP

        // The stack_switch instruction uses register x0 for the payload.
        // Here, the payload indicates that we are returning (value 0).
        // See the test case below to keep this in sync with
        // ControlEffect::return_()
        mov x0, #0

        br x16
        ",
        fiber_start = sym super::fiber_start,
    );
}

#[test]
fn test_return_payload() {
    // The following assumption is baked into `wasmtime_continuation_start`.
    assert_eq!(wasmtime_environ::CONTROL_EFFECT_RETURN_DISCRIMINANT, 0);
}
