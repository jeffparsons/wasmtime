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
use crate::component::func::{LiftContext, ValSource, ValidatedCabiBytes, ValidatedCabiBytesBuf};
use crate::component::types::{
    Type, interface_type_all_bit_patterns_valid, interface_type_is_cabi_inline,
};
use crate::component::values::Val;
use crate::prelude::*;
use crate::{AsContext, AsContextMut, ValRaw};
use core::mem::MaybeUninit;
use wasmtime_environ::component::{
    InterfaceType, MAX_FLAT_PARAMS, MAX_FLAT_RESULTS, TypeTupleIndex,
};

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

    /// Invoke the function and read its results in place, without copying
    /// them out of guest memory.
    ///
    /// Runs the guest and then hands a borrowed [`Results`] accessor to `f`.
    /// Within `f`, each result can be read as a zero-copy proof-carrying byte
    /// view ([`Results::view`] / [`Results::view_checked`]), an owned
    /// validated copy ([`Results::copy`]), or a dynamic [`Val`]
    /// ([`Results::val`]), chosen independently per result. The borrows
    /// handed to `f` are confined to it — they cannot escape — and the
    /// guest's `post-return` runs after `f` returns.
    ///
    /// This is the zero-copy counterpart to [`invoke`](Self::invoke): use it
    /// when the host wants to read guest-produced inline data (for example a
    /// `list<T>`) in place rather than materializing it as [`Val`]s.
    ///
    /// # Errors
    ///
    /// Returns an error for the same argument/spec problems as
    /// [`invoke`](Self::invoke), if a trap occurs, or if `f` itself returns
    /// an error.
    ///
    /// # Panics
    ///
    /// Panics if `store` does not own the underlying function.
    pub fn invoke_scoped<R>(
        self,
        store: impl AsContextMut,
        f: impl FnOnce(&mut Results<'_, '_>) -> Result<R>,
    ) -> Result<R> {
        self.run(store, |cx, results_ty, src| {
            let mut results = Results::new(cx, results_ty, src)?;
            f(&mut results)
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

/// A borrowed accessor over a call's results, handed to the closure passed to
/// [`BoundCall::invoke_scoped`].
///
/// Each result can be read as a zero-copy proof-carrying byte view
/// ([`view`](Self::view) / [`view_checked`](Self::view_checked)), an owned
/// validated copy ([`copy`](Self::copy)), or a dynamic [`Val`]
/// ([`val`](Self::val)), chosen independently per result. Views borrow guest
/// memory and are valid only until the accessor's closure returns; the borrow
/// checker prevents them from escaping it.
pub struct Results<'a, 'b> {
    cx: &'a mut LiftContext<'b>,
    /// The results tuple type.
    results_ty: TypeTupleIndex,
    /// The flattened core result values. When the results are returned
    /// indirectly this is instead a single pointer to the results block,
    /// which [`Results::new`] resolves into `indirect`.
    src: &'a [ValRaw],
    /// For an indirectly-returned result tuple, the base pointer of the tuple
    /// in guest memory and the byte offset of each result within it. `None`
    /// when the results are returned flat (0 or 1 core values).
    indirect: Option<(usize, Vec<usize>)>,
}

impl<'a, 'b> Results<'a, 'b> {
    fn new(
        cx: &'a mut LiftContext<'b>,
        results_ty: InterfaceType,
        src: &'a [ValRaw],
    ) -> Result<Self> {
        let results_ty = match results_ty {
            InterfaceType::Tuple(i) => i,
            _ => unreachable!(),
        };
        let tuple = &cx.types[results_ty];
        let indirect = if tuple.abi.flat_count(MAX_FLAT_RESULTS).is_some() {
            None
        } else {
            // FIXME(#4311): needs to read an i64 for memory64
            let ptr = usize::try_from(src[0].get_u32())?;
            if ptr % usize::try_from(tuple.abi.align32)? != 0 {
                bail!("return pointer not aligned");
            }
            let size = usize::try_from(tuple.abi.size32).unwrap();
            cx.memory()
                .get(ptr..)
                .and_then(|b| b.get(..size))
                .ok_or_else(|| crate::format_err!("pointer out of bounds of memory"))?;
            let mut offset = 0;
            let offsets = tuple
                .types
                .iter()
                .map(|ty| cx.types.canonical_abi(ty).next_field32_size(&mut offset))
                .collect();
            Some((ptr, offsets))
        };
        Ok(Results {
            cx,
            results_ty,
            src,
            indirect,
        })
    }

    /// The number of results.
    pub fn len(&self) -> usize {
        self.cx.types[self.results_ty].types.len()
    }

    /// Returns `true` if the function has no results.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn result_type(&self, index: usize) -> Result<InterfaceType> {
        self.cx.types[self.results_ty]
            .types
            .get(index)
            .copied()
            .ok_or_else(|| crate::format_err!("result index {index} out of bounds"))
    }

    /// Extract the raw element bytes of a `list` result `index` from guest
    /// memory, with bounds and alignment hardening, along with the element's
    /// interface type.
    fn list_bytes(&self, index: usize) -> Result<(&'b [u8], InterfaceType)> {
        let ty = self.result_type(index)?;
        let element = match ty {
            InterfaceType::List(i) => self.cx.types[i].element,
            _ => bail!("result {index} is not a list; only list results can be viewed as bytes"),
        };
        // A list result is always returned indirectly (its pointer/length
        // pair is two core values, exceeding MAX_FLAT_RESULTS), so `indirect`
        // is `Some`.
        let (base, offsets) = self
            .indirect
            .as_ref()
            .expect("a list result is always returned indirectly");
        let field = base + offsets[index];
        let memory = self.cx.memory();
        // FIXME(#4311): needs memory64 handling
        let ptr = usize::try_from(u32::from_le_bytes(memory[field..][..4].try_into().unwrap()))?;
        let len = usize::try_from(u32::from_le_bytes(
            memory[field + 4..][..4].try_into().unwrap(),
        ))?;
        let abi = self.cx.types.canonical_abi(&element);
        // Reject a misaligned list pointer, matching the `Val` lift path
        // (`load_list`). Callers read the returned bytes as `&[T]`, so an
        // adversarial guest returning a misaligned (but in-bounds) pointer
        // must not be accepted here either.
        if ptr % usize::try_from(abi.align32)? != 0 {
            bail!("result {index}: list pointer is not aligned");
        }
        let elt_size = usize::try_from(abi.size32).unwrap();
        let byte_len = len
            .checked_mul(elt_size)
            .ok_or_else(|| crate::format_err!("list size overflow"))?;
        let bytes = memory
            .get(ptr..)
            .and_then(|b| b.get(..byte_len))
            .ok_or_else(|| crate::format_err!("list out of bounds of memory"))?;
        Ok((bytes, element))
    }

    /// Mint a proof-carrying wrapper over a list result's element bytes.
    ///
    /// Guest memory is *not* pre-validated by the runtime — a zero-copy view
    /// lifts nothing — so this is where the [`ValidatedCabiBytes`] contract
    /// is established: O(1) for bit-pattern-total element types, a linear
    /// sweep otherwise.
    fn mint_proof(&self, index: usize) -> Result<ValidatedCabiBytes<'b>> {
        let (bytes, element) = self.list_bytes(index)?;
        let element = Type::from(&element, &self.cx.instance_type());
        ValidatedCabiBytes::checked(bytes, &element)
            .with_context(|| format!("result {index}: guest-produced list failed validation"))
    }

    /// Read result `index` as a zero-copy, proof-carrying view of its
    /// canonical-ABI element bytes.
    ///
    /// Only valid when the result is a `list<T>` whose element type has
    /// [`are_all_bit_patterns_valid`](Type::are_all_bit_patterns_valid) — for
    /// those types every bit pattern is a value, so the proof is free and
    /// this is O(1). For a `list` of an inline type that *does* need
    /// validation (`bool`, `enum`, …) use
    /// [`view_checked`](Self::view_checked), which pays a linear sweep; for
    /// anything else use [`val`](Self::val).
    ///
    /// The returned [`ValidatedCabiBytes`] can be fed directly to another
    /// call's [`ValSource::ListFlat`] argument (in a different store) with no
    /// re-encoding and no re-validation — the guest→guest conduit.
    pub fn view(&self, index: usize) -> Result<ValidatedCabiBytes<'b>> {
        let (_, element) = self.list_bytes(index)?;
        if !interface_type_all_bit_patterns_valid(self.cx.types, &element) {
            bail!(
                "result {index} is a list whose element type has bit patterns that need \
                 validation; use `view_checked` (a linear sweep) or `val` instead"
            );
        }
        self.mint_proof(index)
    }

    /// Read result `index` as a zero-copy, proof-carrying view of its
    /// canonical-ABI element bytes, running a validation sweep over them.
    ///
    /// The checked counterpart of [`view`](Self::view) for `list`s of inline
    /// element types with invalid bit patterns (`bool`, `char`, `enum`,
    /// `flags`, discriminated unions): the runtime sweeps the guest-produced
    /// bytes once (rejecting e.g. an out-of-range discriminant an adversarial
    /// guest left in memory) and returns the same proof-carrying
    /// [`ValidatedCabiBytes`]. Keeping this sweep inside Wasmtime means
    /// embedders never hand-roll canonical-ABI validation to get zero-copy
    /// reads of such lists.
    pub fn view_checked(&self, index: usize) -> Result<ValidatedCabiBytes<'b>> {
        let (_, element) = self.list_bytes(index)?;
        if !interface_type_is_cabi_inline(self.cx.types, &element) {
            bail!(
                "result {index} is a list whose element type is not inline; read it with \
                 `val` instead"
            );
        }
        self.mint_proof(index)
    }

    /// Read result `index` as an owned, validated copy of its canonical-ABI
    /// element bytes.
    ///
    /// Equivalent to [`view_checked`](Self::view_checked) followed by
    /// [`ValidatedCabiBytes::to_owned`], and subject to the same
    /// restrictions: the proof (and any needed validation sweep) is
    /// established before the bytes are copied out, so the returned buffer
    /// can be stored and replayed into later calls with no further checks.
    pub fn copy(&self, index: usize) -> Result<ValidatedCabiBytesBuf> {
        Ok(self.view_checked(index)?.to_owned())
    }

    /// Read result `index` as a dynamic [`Val`]. Works for any result type.
    pub fn val(&mut self, index: usize) -> Result<Val> {
        let ty = self.result_type(index)?;
        match &self.indirect {
            Some((base, offsets)) => {
                let field = base + offsets[index];
                let size = usize::try_from(self.cx.types.canonical_abi(&ty).size32).unwrap();
                // Copy the field's bytes out so the shared borrow of memory
                // ends before `Val::load` takes `cx` mutably.
                let bytes = self.cx.memory()[field..][..size].to_vec();
                Val::load(self.cx, ty, &bytes)
            }
            None => {
                // Flat results: at most MAX_FLAT_RESULTS core values total,
                // so lifting reads directly from `src`.
                let mut flat = self.src.iter();
                Val::lift(self.cx, ty, &mut flat)
            }
        }
    }
}
