//! Stack switching support for AArch64.
//!
//! The `stack_switch` instruction loads information about the stack to switch
//! to and stores information about the current stack by receiving pointers to
//! memory laid out as in the struct `ControlContext` below.
//!
//! ```ignore
//! #[repr(C)]
//! pub struct ControlContext {
//!     pub stack_pointer: *mut u8,
//!     pub frame_pointer: *mut u8,
//!     pub instruction_pointer: *mut u8,
//! }
//! ```
//!
//! Note that this layout is deliberately chosen to make frame pointer walking
//! possible, if desired: The layout enables stack layouts where a
//! `ControlContext` is part of a frame pointer chain, putting the frame pointer
//! next to the corresponding IP.
//!
//! We never actually interact with values of that type in Cranelift, but are
//! only interested in its layout for the purposes of generating code.

use crate::isa::aarch64::inst::regs;
use crate::machinst::Reg;

/// Layout of the control context structure in memory.
pub struct ControlContextLayout {
    /// Offset of the stack pointer field within the control context.
    pub stack_pointer_offset: usize,
    /// Offset of the frame pointer field within the control context.
    pub frame_pointer_offset: usize,
    /// Offset of the instruction pointer field within the control context.
    pub ip_offset: usize,
}

/// Returns the layout of the control context structure.
pub fn control_context_layout() -> ControlContextLayout {
    ControlContextLayout {
        stack_pointer_offset: 0,
        frame_pointer_offset: 8,
        ip_offset: 16,
    }
}

/// The register used for handing over the payload when switching stacks.
///
/// We must use a fixed register for sending and receiving the payload: When
/// switching from one stack to another using two matching `stack_switch`
/// instructions, they must agree on the register where the payload is, similar
/// to a calling convention. The same holds when `stack_switch`-ing to a newly
/// initialized stack, where the entry trampoline must know which register the
/// payload is in.
///
/// On AArch64, we use x0 (first argument register) to match the calling
/// convention and be consistent with the trampoline implementation.
pub fn payload_register() -> Reg {
    regs::xreg(0)
}
