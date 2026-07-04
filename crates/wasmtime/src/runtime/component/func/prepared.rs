//! A "prepared call" surface for the dynamic (`Val`-based) component calling
//! path that lets each argument opt into a bulk, pre-encoded representation.
//!
//! The dynamic [`Func::call`](crate::component::Func::call) path requires every
//! argument to be materialized as a [`Val`], and a `list<T>` in particular
//! becomes a `Vec<Val>` with one boxed `Val` per element. For a host that
//! already holds a large collection of values whose canonical-ABI image it can
//! produce cheaply (for example an ECS column of `vec2`s), that per-element
//! boxing dominates the cost of the call.
//!
//! This module adds a middle ground between "materialize everything as `Val`"
//! and the fully static [`TypedFunc`](crate::component::TypedFunc) path:
//!
//! * [`Func::prepare_call`](crate::component::Func::prepare_call) validates,
//!   once, how each argument will be supplied ([`ArgSpec`]) against the
//!   function's parameters and returns a reusable [`PreparedCall`].
//! * [`PreparedCall::bind`] begins a single invocation, borrowing the concrete
//!   argument buffers only until [`BoundCall::invoke`] consumes the binding.
//!
//! An argument declared [`ArgSpec::Flat`] is supplied as a borrowed byte slice
//! holding the parameter's canonical-ABI image ([`ArgSource::Flat`]); provided
//! the parameter is a `list<T>` whose element type is
//! [`Type::is_cabi_inline`](crate::component::Type::is_cabi_inline), those bytes
//! are copied into guest memory with a single `memcpy` rather than element by
//! element. Any argument may instead be supplied as a dynamic [`Val`]
//! ([`ArgSpec::Val`] / [`ArgSource::Val`]), so a single call freely mixes the
//! two representations.
//!
//! Results are read back as [`Val`]s, exactly as with
//! [`Func::call`](crate::component::Func::call); a zero-copy result-reading
//! surface is intended as a follow-up.

use crate::component::Func;
use crate::component::func::LowerContext;
use crate::component::types::Type;
use crate::component::values::Val;
use crate::prelude::*;
use crate::{AsContext, AsContextMut, ValRaw};
use core::mem::MaybeUninit;
use wasmtime_environ::component::{InterfaceType, MAX_FLAT_PARAMS, MAX_FLAT_RESULTS, TypeTuple};

/// How a single argument to a [`PreparedCall`] will be supplied, chosen once
/// when the call is prepared.
///
/// This carries no data — it only records the *representation* an argument will
/// use, so that compatibility with the parameter type can be validated a single
/// time in [`Func::prepare_call`](crate::component::Func::prepare_call) rather
/// than on every invocation. The matching data is supplied per call as an
/// [`ArgSource`].
///
/// This enum is `#[non_exhaustive]` because further representations (for example
/// streamed or typed-buffer sources) are anticipated and will be added without a
/// breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArgSpec {
    /// The argument will be supplied as a dynamic [`Val`] and lowered element by
    /// element. Compatible with any parameter type.
    Val,

    /// The argument will be supplied as a pre-encoded canonical-ABI byte image
    /// ([`ArgSource::Flat`]) and copied into guest memory in bulk.
    ///
    /// Only valid when the parameter is a `list<T>` whose element type `T` is
    /// [`Type::is_cabi_inline`](crate::component::Type::is_cabi_inline);
    /// [`Func::prepare_call`](crate::component::Func::prepare_call) returns an
    /// error otherwise.
    Flat,
}

/// The concrete data for a single argument, supplied per invocation on a
/// [`BoundCall`].
///
/// The variant must match the [`ArgSpec`] chosen for the same argument position
/// when the call was prepared. `Flat` borrows the caller's buffer only for the
/// duration of the binding.
///
/// This enum is `#[non_exhaustive]` for the same forward-compatibility reason as
/// [`ArgSpec`].
#[non_exhaustive]
pub enum ArgSource<'a> {
    /// A dynamic value, lowered element by element. Pairs with [`ArgSpec::Val`].
    Val(Val),

    /// A pre-encoded canonical-ABI image of a `list<T>`: the elements laid out
    /// back-to-back exactly as they appear in linear memory, little-endian.
    /// Pairs with [`ArgSpec::Flat`].
    ///
    /// The length must be a whole number of elements; otherwise
    /// [`BoundCall::invoke`] returns an error.
    Flat(&'a [u8]),
}

/// A validated, reusable component call shape.
///
/// Created by [`Func::prepare_call`](crate::component::Func::prepare_call). It
/// holds no borrows of concrete argument data and no lifetime, so it can be kept
/// and reused across many invocations — for example hoisted out of a hot loop.
/// Call [`bind`](PreparedCall::bind) to supply arguments for a single
/// invocation.
#[derive(Debug)]
pub struct PreparedCall {
    func: Func,
    specs: Vec<ArgSpec>,
}

impl PreparedCall {
    /// Begin binding concrete arguments for a single invocation.
    ///
    /// The returned [`BoundCall`] borrows nothing until arguments are added with
    /// [`BoundCall::arg`].
    pub fn bind(&self) -> BoundCall<'_> {
        BoundCall {
            prepared: self,
            sources: Vec::with_capacity(self.specs.len()),
        }
    }
}

/// A single invocation of a [`PreparedCall`], in the process of having its
/// arguments bound.
///
/// The lifetime `'a` ties this binding to the argument buffers it borrows;
/// [`invoke`](BoundCall::invoke) consumes the binding, releasing those borrows
/// before the next statement so the buffers are freely mutable again between
/// calls.
pub struct BoundCall<'a> {
    prepared: &'a PreparedCall,
    sources: Vec<ArgSource<'a>>,
}

impl<'a> BoundCall<'a> {
    /// Supply the next argument, in parameter order.
    ///
    /// The [`ArgSource`] variant must match the [`ArgSpec`] chosen for this
    /// position in [`Func::prepare_call`](crate::component::Func::prepare_call);
    /// a mismatch is reported by [`invoke`](Self::invoke).
    pub fn arg(mut self, source: ArgSource<'a>) -> Self {
        self.sources.push(source);
        self
    }

    /// Invoke the function, writing the lifted results into `results`.
    ///
    /// `results` must have exactly as many elements as the function has results;
    /// each is overwritten with the corresponding lifted [`Val`]. This also runs
    /// the function's `post-return`, if any.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of bound arguments or the size of
    /// `results` does not match the function's signature, if a bound
    /// [`ArgSource`] does not match its prepared [`ArgSpec`], if a `Flat` byte
    /// buffer's length is not a whole number of elements, or if a trap occurs
    /// while executing the function.
    ///
    /// # Panics
    ///
    /// Panics if `store` does not own the underlying function.
    pub fn invoke(self, mut store: impl AsContextMut, results: &mut [Val]) -> Result<()> {
        let mut store = store.as_context_mut();
        store.0.validate_sync_call()?;

        let func = self.prepared.func;
        let specs = &self.prepared.specs;
        let sources = &self.sources;

        if sources.len() != specs.len() {
            bail!(
                "expected {} argument(s), got {}",
                specs.len(),
                sources.len()
            );
        }

        let ty = func.ty(store.as_context());
        if ty.results().len() != results.len() {
            bail!(
                "expected {} result(s), got {}",
                ty.results().len(),
                results.len()
            );
        }
        drop(ty);

        if func.abi_async(store.0) {
            unreachable!(
                "async-lifted exports should have failed validation \
                 when `component-model-async` feature disabled"
            );
        }

        // SAFETY: the representations chosen here mirror `Func::call_impl`:
        // parameters are the maximal flat-parameter array and results the
        // maximal flat-result array, both as `ValRaw`. The lowering closure
        // fills the parameter storage to match the function's actual parameter
        // types (either flattened onto the stack or, for larger signatures,
        // stored behind a single pointer), and the lifting closure interprets
        // the results using the function's actual result types.
        let (_, post_return_arg) = unsafe {
            func.call_raw(
                store.as_context_mut(),
                |cx, params_ty, dst: &mut MaybeUninit<[MaybeUninit<ValRaw>; MAX_FLAT_PARAMS]>| {
                    // SAFETY: `MaybeUninit<array-of-maybe-uninit>` is safe to
                    // treat as initialized because each element remains
                    // individually uninitialized.
                    let dst: &mut [MaybeUninit<ValRaw>] = dst.assume_init_mut();
                    lower_sources(cx, specs, sources, params_ty, dst)
                },
                |cx, results_ty, src: &[ValRaw; MAX_FLAT_RESULTS]| {
                    for (result, slot) in
                        Func::lift_results(cx, results_ty, src, MAX_FLAT_RESULTS)?.zip(results)
                    {
                        *slot = result?;
                    }
                    Ok(())
                },
            )?
        };

        func.post_return_impl(store, post_return_arg)
    }
}

impl Func {
    /// Prepare a reusable, dynamically-typed call to this function in which each
    /// argument may opt into a bulk, pre-encoded representation.
    ///
    /// `specs` describes, in parameter order, how each argument will be supplied
    /// on each invocation (see [`ArgSpec`]). This validates — once — that the
    /// number of specs matches the function's arity and that each spec is
    /// compatible with its parameter type, so that repeated invocations of the
    /// returned [`PreparedCall`] need not re-check the call's shape.
    ///
    /// This is the dynamic-path counterpart to [`Func::typed`]: like
    /// [`Func::call`] it works without static knowledge of the component's
    /// types, but it additionally lets a `list<T>` argument whose element type
    /// is [`Type::is_cabi_inline`](crate::component::Type::is_cabi_inline) be
    /// provided as a borrowed slice of its canonical-ABI bytes
    /// ([`ArgSource::Flat`]) and copied into guest memory in bulk, avoiding the
    /// per-element `Val` allocation of [`Func::call`].
    ///
    /// # Errors
    ///
    /// Returns an error if `specs.len()` does not equal the number of
    /// parameters, or if an [`ArgSpec::Flat`] is paired with a parameter that is
    /// not a `list` of a [`Type::is_cabi_inline`](crate::component::Type::is_cabi_inline)
    /// element type.
    ///
    /// # Panics
    ///
    /// Panics if `store` does not own this function.
    pub fn prepare_call(&self, store: impl AsContext, specs: &[ArgSpec]) -> Result<PreparedCall> {
        let ty = self.ty(store);
        let params: Vec<Type> = ty.params().map(|(_, ty)| ty).collect();

        if specs.len() != params.len() {
            bail!(
                "expected {} argument spec(s), got {}",
                params.len(),
                specs.len()
            );
        }

        for (index, (spec, param)) in specs.iter().zip(&params).enumerate() {
            match spec {
                ArgSpec::Val => {}
                ArgSpec::Flat => match param {
                    Type::List(list) if list.ty().is_cabi_inline() => {}
                    Type::List(_) => bail!(
                        "argument {index}: `ArgSpec::Flat` requires a `list` whose element type \
                         has a fixed canonical-ABI layout (no strings, lists, resources, or types \
                         with invalid bit patterns), which this parameter's element type is not"
                    ),
                    _ => bail!(
                        "argument {index}: `ArgSpec::Flat` requires a `list` parameter, but this \
                         parameter is not a list"
                    ),
                },
            }
        }

        Ok(PreparedCall {
            func: *self,
            specs: specs.to_vec(),
        })
    }
}

/// Lower every argument into the parameter storage `dst`, mirroring
/// `Func::lower_args` but driven by per-argument [`ArgSource`]s instead of a
/// uniform slice of [`Val`]s.
fn lower_sources<T>(
    cx: &mut LowerContext<'_, T>,
    specs: &[ArgSpec],
    sources: &[ArgSource<'_>],
    params_ty: InterfaceType,
    dst: &mut [MaybeUninit<ValRaw>],
) -> Result<()> {
    let params_ty = match params_ty {
        InterfaceType::Tuple(i) => &cx.types[i],
        _ => unreachable!(),
    };
    if params_ty.abi.flat_count(MAX_FLAT_PARAMS).is_some() {
        let dst = &mut dst.iter_mut();
        for (index, ((spec, source), ty)) in specs
            .iter()
            .zip(sources)
            .zip(params_ty.types.iter())
            .enumerate()
        {
            match (spec, source) {
                (ArgSpec::Val, ArgSource::Val(value)) => value.lower(cx, *ty, dst)?,
                (ArgSpec::Flat, ArgSource::Flat(bytes)) => {
                    let element = match ty {
                        InterfaceType::List(i) => cx.types[*i].element,
                        // Guaranteed by `prepare_call`.
                        _ => unreachable!("`ArgSpec::Flat` validated to a list at prepare time"),
                    };
                    let (ptr, len) = lower_flat_list(cx, element, bytes)?;
                    dst.next().unwrap().write(ValRaw::i64(ptr as i64));
                    dst.next().unwrap().write(ValRaw::i64(len as i64));
                }
                _ => bail!(spec_mismatch(index, spec)),
            }
        }
        Ok(())
    } else {
        store_sources(cx, params_ty, specs, sources, dst)
    }
}

/// The indirect counterpart to [`lower_sources`], mirroring `Func::store_args`:
/// when the parameters don't fit in flat storage they are stored into a single
/// allocated region and a pointer to it is passed instead.
fn store_sources<T>(
    cx: &mut LowerContext<'_, T>,
    params_ty: &TypeTuple,
    specs: &[ArgSpec],
    sources: &[ArgSource<'_>],
    dst: &mut [MaybeUninit<ValRaw>],
) -> Result<()> {
    let size = usize::try_from(params_ty.abi.size32).unwrap();
    let ptr = cx.realloc(0, 0, params_ty.abi.align32, size)?;
    let mut offset = ptr;
    for (index, ((spec, source), ty)) in specs
        .iter()
        .zip(sources)
        .zip(params_ty.types.iter())
        .enumerate()
    {
        let field_offset = cx.types.canonical_abi(ty).next_field32_size(&mut offset);
        match (spec, source) {
            (ArgSpec::Val, ArgSource::Val(value)) => value.store(cx, *ty, field_offset)?,
            (ArgSpec::Flat, ArgSource::Flat(bytes)) => {
                let element = match ty {
                    InterfaceType::List(i) => cx.types[*i].element,
                    _ => unreachable!("`ArgSpec::Flat` validated to a list at prepare time"),
                };
                let (list_ptr, len) = lower_flat_list(cx, element, bytes)?;
                // FIXME(#4311): needs memory64 handling
                *cx.get(field_offset + 0) = u32::try_from(list_ptr).unwrap().to_le_bytes();
                *cx.get(field_offset + 4) = u32::try_from(len).unwrap().to_le_bytes();
            }
            _ => bail!(spec_mismatch(index, spec)),
        }
    }

    dst[0].write(ValRaw::i64(ptr as i64));
    Ok(())
}

/// Copy the pre-encoded canonical image of a `list<T>` into guest memory with a
/// single `memcpy`, returning the `(pointer, element_count)` pair for the list.
///
/// The element type is assumed to be [`Type::is_cabi_inline`], as validated at
/// prepare time, so the raw bytes need no per-element interpretation.
fn lower_flat_list<T>(
    cx: &mut LowerContext<'_, T>,
    element_type: InterfaceType,
    bytes: &[u8],
) -> Result<(usize, usize)> {
    let abi = cx.types.canonical_abi(&element_type);
    let elt_size = usize::try_from(abi.size32).unwrap();
    let elt_align = abi.align32;

    let count = if elt_size == 0 {
        if !bytes.is_empty() {
            bail!(
                "flat argument has {} byte(s) but the list element type is zero-sized",
                bytes.len()
            );
        }
        0
    } else {
        if bytes.len() % elt_size != 0 {
            bail!(
                "flat argument byte length {} is not a multiple of the element size {elt_size}",
                bytes.len()
            );
        }
        bytes.len() / elt_size
    };

    let ptr = cx.realloc(0, 0, elt_align, bytes.len())?;
    cx.as_slice_mut()[ptr..][..bytes.len()].copy_from_slice(bytes);
    Ok((ptr, count))
}

#[cold]
fn spec_mismatch(index: usize, spec: &ArgSpec) -> String {
    format!("argument {index}: the bound `ArgSource` does not match the prepared `{spec:?}` spec")
}
