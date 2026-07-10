//! A "prepared call" surface for the dynamic (`Val`-based) component calling
//! path that lets each argument choose its lowering strategy.
//!
//! The dynamic [`Func::call`](crate::component::Func::call) path requires
//! every argument to be materialized as a [`Val`], and a `list<T>` in
//! particular becomes a `Vec<Val>` with one boxed `Val` per element. For a
//! host that already holds a large collection of values whose canonical-ABI
//! image it can produce cheaply (for example an ECS column of `vec2`s), that
//! per-element boxing dominates the cost of the call.
//!
//! This module adds a middle ground between "materialize everything as `Val`"
//! and the fully static [`TypedFunc`](crate::component::TypedFunc) path:
//!
//! * [`Func::prepare_call`](crate::component::Func::prepare_call) validates,
//!   once, how each argument will be supplied ([`ValSpec`]) against the
//!   function's parameters and returns a reusable [`PreparedCall`].
//! * [`PreparedCall::bind`] begins a single invocation, borrowing the
//!   concrete argument sources only until [`BoundCall::invoke`] consumes the
//!   binding.
//!
//! Crucially this is *not* a separate lowering implementation: the argument
//! vocabulary is [`ValSource`], the same enum the engine behind
//! [`Func::call`](crate::component::Func::call) itself consumes — `Func::call`
//! is exactly a prepared call whose every argument uses the
//! [`ValSource::Val`] strategy. A [`ValSource::ListFlat`] argument instead
//! supplies a `list<T>` as a [`ValidatedCabiBytes`] run of pre-validated
//! canonical element images, which the engine copies into guest memory with
//! one `memcpy` — no per-element lowering, and no per-element validation
//! either, because validation already happened (exactly once) when the proof
//! was minted. A single call freely mixes strategies per argument.

use crate::component::Func;
use crate::component::func::{LiftContext, ValSource};
use crate::component::types::Type;
use crate::component::values::Val;
use crate::prelude::*;
use crate::{AsContext, AsContextMut, ValRaw};
use core::mem::MaybeUninit;
use wasmtime_environ::component::{InterfaceType, MAX_FLAT_PARAMS, MAX_FLAT_RESULTS};

/// How a single argument to a [`PreparedCall`] will be supplied, chosen once
/// when the call is prepared.
///
/// This carries no data — it only records the *strategy* an argument will
/// use, so that compatibility with the parameter type can be validated a
/// single time in [`Func::prepare_call`](crate::component::Func::prepare_call)
/// rather than on every invocation. The matching data is supplied per call as
/// a [`ValSource`].
///
/// This enum is `#[non_exhaustive]` because further strategies are
/// anticipated and will be added without a breaking change, mirroring
/// [`ValSource`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValSpec {
    /// The argument will be supplied as a dynamic [`Val`]
    /// ([`ValSource::Val`]) and lowered element by element. Compatible with
    /// any parameter type.
    Val,

    /// The argument will be supplied as a run of pre-validated canonical-ABI
    /// element images ([`ValSource::ListFlat`]) and copied into guest memory
    /// in bulk.
    ///
    /// Only valid when the parameter is a `list<T>` where `T` is
    /// [`is_cabi_inline`](Type::is_cabi_inline);
    /// [`Func::prepare_call`](crate::component::Func::prepare_call) returns
    /// an error otherwise. Note that `T` does *not* need
    /// [`are_all_bit_patterns_valid`](Type::are_all_bit_patterns_valid): for
    /// types like `enum` that have invalid bit patterns, the validation
    /// sweep was already paid when the [`ValidatedCabiBytes`] proof was
    /// constructed, so the per-call lowering is a plain `memcpy` either way.
    ///
    /// [`ValidatedCabiBytes`]: crate::component::ValidatedCabiBytes
    ListFlat,
}

/// A validated, reusable component call shape.
///
/// Created by [`Func::prepare_call`](crate::component::Func::prepare_call).
/// It holds no borrows of concrete argument data and no lifetime, so it can
/// be kept and reused across many invocations — for example hoisted out of a
/// hot loop. Call [`bind`](PreparedCall::bind) to supply arguments for a
/// single invocation.
#[derive(Debug)]
pub struct PreparedCall {
    func: Func,
    specs: Vec<ValSpec>,
    params: Vec<Type>,
}

impl PreparedCall {
    /// Begin binding concrete arguments for a single invocation.
    ///
    /// The returned [`BoundCall`] borrows nothing until arguments are added
    /// with [`BoundCall::arg`].
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
/// The lifetime `'a` ties this binding to the argument data it borrows;
/// [`invoke`](BoundCall::invoke) consumes the binding, releasing those
/// borrows before the next statement so the underlying buffers are freely
/// mutable again between calls.
pub struct BoundCall<'a> {
    prepared: &'a PreparedCall,
    sources: Vec<ValSource<'a>>,
}

impl<'a> BoundCall<'a> {
    /// Supply the next argument, in parameter order.
    ///
    /// The [`ValSource`] variant must match the [`ValSpec`] chosen for this
    /// position in
    /// [`Func::prepare_call`](crate::component::Func::prepare_call); a
    /// mismatch is reported by [`invoke`](Self::invoke).
    pub fn arg(mut self, source: ValSource<'a>) -> Self {
        self.sources.push(source);
        self
    }

    /// Supply the next argument as a dynamic [`Val`]; sugar for
    /// [`arg`](Self::arg) with [`ValSource::Val`].
    pub fn arg_val(self, value: &'a Val) -> Self {
        self.arg(ValSource::Val(value))
    }

    /// Invoke the function, writing the lifted results into `results`.
    ///
    /// `results` must have exactly as many elements as the function has
    /// results; each is overwritten with the corresponding lifted [`Val`].
    /// This also runs the function's `post-return`, if any.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of bound arguments or the size of
    /// `results` does not match the function's signature, if a bound
    /// [`ValSource`] does not match its prepared [`ValSpec`], if a
    /// [`ValSource::ListFlat`] proof's element type is not structurally equal
    /// to the parameter's element type, or if a trap occurs while executing
    /// the function.
    ///
    /// # Panics
    ///
    /// Panics if `store` does not own the underlying function.
    pub fn invoke(self, store: impl AsContextMut, results: &mut [Val]) -> Result<()> {
        self.run(store, |cx, results_ty, src| {
            let tuple = match results_ty {
                InterfaceType::Tuple(i) => &cx.types[i],
                _ => unreachable!(),
            };
            if tuple.types.len() != results.len() {
                bail!(
                    "expected {} result(s), got {}",
                    tuple.types.len(),
                    results.len()
                );
            }
            for (result, slot) in
                Func::lift_results(cx, results_ty, src, MAX_FLAT_RESULTS)?.zip(results)
            {
                *slot = result?;
            }
            Ok(())
        })
    }

    /// Shared invocation machinery: validate the binding, lower the arguments
    /// through the engine, run the guest, lift the results via `lift`, then
    /// run `post-return`.
    fn run<R>(
        self,
        mut store: impl AsContextMut,
        lift: impl FnOnce(&mut LiftContext<'_>, InterfaceType, &[ValRaw; MAX_FLAT_RESULTS]) -> Result<R>,
    ) -> Result<R> {
        let mut store = store.as_context_mut();

        let func = self.prepared.func;
        let specs = &self.prepared.specs;
        let params = &self.prepared.params;
        let sources = &self.sources;

        if sources.len() != specs.len() {
            bail!(
                "expected {} argument(s), got {}",
                specs.len(),
                sources.len()
            );
        }

        // Validate that each bound source matches its prepared spec — and for
        // `ListFlat`, that the proof was minted for a type structurally equal
        // to the parameter's element type — *before* entering the guest, so a
        // mismatch is a clean error rather than surfacing mid-lowering after
        // guest memory has been allocated. Note the structural (not
        // identity) equality: a proof minted against one component's
        // reflected type is accepted by any other component's structurally
        // equal type, which is what lets validated bytes flow between
        // instances.
        for (index, ((spec, source), param)) in specs.iter().zip(sources).zip(params).enumerate() {
            match (spec, source) {
                (ValSpec::Val, ValSource::Val(_)) => {}
                (ValSpec::ListFlat, ValSource::ListFlat(vb)) => {
                    let elem = param.unwrap_list().ty();
                    if vb.ty() != &elem {
                        bail!(
                            "argument {index}: buffer holds validated `{}` images but the \
                             parameter's element type is `{}`",
                            vb.ty().desc(),
                            elem.desc(),
                        );
                    }
                }
                (spec, source) => bail!(
                    "argument {index}: bound source `{}` does not match the prepared \
                     `ValSpec::{spec:?}`",
                    source.desc(),
                ),
            }
        }

        if func.abi_async(store.0) {
            unreachable!(
                "async-lifted exports should have failed validation \
                 when `component-model-async` feature disabled"
            );
        }

        // SAFETY: the representations chosen here mirror `Func::call_impl`:
        // parameters are the maximal flat-parameter array and results the
        // maximal flat-result array, both as `ValRaw`. The lowering closure
        // fills the parameter storage to match the function's actual
        // parameter types (either flattened onto the stack or, for larger
        // signatures, stored behind a single pointer), and `lift` interprets
        // the results using the function's actual result types.
        let (value, post_return_arg) = unsafe {
            func.call_raw(
                store.as_context_mut(),
                |cx, params_ty, dst: &mut MaybeUninit<[MaybeUninit<ValRaw>; MAX_FLAT_PARAMS]>| {
                    // SAFETY: `MaybeUninit<array-of-maybe-uninit>` is safe to
                    // treat as initialized because each element remains
                    // individually uninitialized.
                    let dst: &mut [MaybeUninit<ValRaw>] = dst.assume_init_mut();
                    Func::lower_args(cx, sources, params_ty, dst)
                },
                lift,
            )?
        };

        func.post_return_impl(store, post_return_arg)?;
        Ok(value)
    }
}

impl Func {
    /// Prepare a reusable, dynamically-typed call to this function in which
    /// each argument chooses its lowering strategy.
    ///
    /// `specs` describes, in parameter order, how each argument will be
    /// supplied on each invocation (see [`ValSpec`]). This validates — once —
    /// that the number of specs matches the function's arity and that each
    /// spec is compatible with its parameter type, so that repeated
    /// invocations of the returned [`PreparedCall`] need not re-check the
    /// call's shape.
    ///
    /// This is not a third calling path beside [`Func::call`] and
    /// [`Func::typed`]: `Func::call` itself runs on the same engine and is
    /// equivalent to a prepared call whose every argument is
    /// [`ValSpec::Val`]. What `prepare_call` adds is the ability to supply a
    /// `list<T>` argument with an inline element type as pre-validated
    /// canonical bytes
    /// ([`ValidatedCabiBytes`](crate::component::ValidatedCabiBytes) via
    /// [`ValSource::ListFlat`]), which crosses into guest memory as a single
    /// `memcpy` instead of one lowered `Val` per element.
    ///
    /// # Errors
    ///
    /// Returns an error if `specs.len()` does not equal the number of
    /// parameters, or if a [`ValSpec::ListFlat`] is paired with a parameter
    /// that is not a `list` whose element type is
    /// [`is_cabi_inline`](Type::is_cabi_inline).
    ///
    /// # Panics
    ///
    /// Panics if `store` does not own this function.
    pub fn prepare_call(&self, store: impl AsContext, specs: &[ValSpec]) -> Result<PreparedCall> {
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
                ValSpec::Val => {}
                ValSpec::ListFlat => match param {
                    Type::List(list) if list.ty().is_cabi_inline() => {}
                    Type::List(_) => bail!(
                        "argument {index}: `ValSpec::ListFlat` requires a `list` whose element \
                         type has a fixed inline canonical-ABI layout (no strings, lists, or \
                         resources), which this parameter's element type does not"
                    ),
                    _ => bail!(
                        "argument {index}: `ValSpec::ListFlat` requires a `list` parameter, but \
                         this parameter is not a list"
                    ),
                },
            }
        }

        Ok(PreparedCall {
            func: *self,
            specs: specs.to_vec(),
            params,
        })
    }
}
