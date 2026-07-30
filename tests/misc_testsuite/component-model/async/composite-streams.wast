;;! component_model_async = true
;;! component_model_more_async_builtins = true
;;! reference_types = true

;; These tests exercise inter-component stream copies whose payload is a
;; composite type, i.e. the types which `flat_stream_element_info` in
;; `wasmtime-cranelift` classifies with `ComponentTypes::is_bitwise_copyable`.
;;
;; Each test has the same shape as `sync-streams.wast`: $C.get writes to a
;; stream that $D reads, and $D writes to a stream that $C.set reads, so both
;; rendezvous orders are covered. The payload bytes are checked exactly on the
;; reading side, which is the property that must hold whether the copy is done
;; with a `memcpy` or by lifting and lowering each element.

;; A `record` of two `float32`s: every bit pattern is a valid value and the
;; layout has no padding, so this is copied verbatim.
(component
  (component $C
    (core module $Memory (memory (export "mem") 1))
    (core instance $memory (instantiate $Memory))
    (core module $CM
      (import "" "mem" (memory 1))
      (import "" "task.return0" (func $task.return0))
      (import "" "task.return1" (func $task.return1 (param i32)))
      (import "" "stream.new" (func $stream.new (result i64)))
      (import "" "stream.read" (func $stream.read (param i32 i32 i32) (result i32)))
      (import "" "stream.write" (func $stream.write (param i32 i32 i32) (result i32)))
      (import "" "stream.drop-readable" (func $stream.drop-readable (param i32)))
      (import "" "stream.drop-writable" (func $stream.drop-writable (param i32)))

      (func (export "get") (result i32)
        (local $ret i32) (local $ret64 i64)
        (local $tx i32) (local $rx i32)
        (local $bufp i32)

        ;; ($rx, $tx) = stream.new
        (local.set $ret64 (call $stream.new))
        (local.set $rx (i32.wrap_i64 (local.get $ret64)))
        (local.set $tx (i32.wrap_i64 (i64.shr_u (local.get $ret64) (i64.const 32))))

        ;; return $rx
        (call $task.return1 (local.get $rx))

        ;; two elements: {1.5, -2.5} and {3.5, -4.5}
        (local.set $bufp (i32.const 0x40))
        (i32.store offset=0 (local.get $bufp) (i32.const 0x3fc00000))
        (i32.store offset=4 (local.get $bufp) (i32.const 0xc0200000))
        (i32.store offset=8 (local.get $bufp) (i32.const 0x40600000))
        (i32.store offset=12 (local.get $bufp) (i32.const 0xc0900000))

        ;; (stream.write $tx $bufp 2) will block and, because called
        ;; synchronously, switch to the caller who will read and rendezvous
        (local.set $ret (call $stream.write (local.get $tx) (local.get $bufp) (i32.const 2)))
        (if (i32.ne (i32.const 0x21 (; DROPPED=1 | (2<<4) ;)) (local.get $ret))
          (then unreachable))

        (call $stream.drop-writable (local.get $tx))
        (return (i32.const 0 (; EXIT ;)))
      )
      (func (export "get_cb") (param i32 i32 i32) (result i32)
        unreachable
      )

      (func (export "set") (param $rx i32) (result i32)
        (local $ret i32)
        (local $bufp i32)

        ;; return immediately so that the caller can just call synchronously
        (call $task.return0)

        ;; (stream.read $rx $bufp 2) will block and, because called
        ;; synchronously, switch to the caller who will write and rendezvous
        (local.set $bufp (i32.const 0x80))
        (local.set $ret (call $stream.read (local.get $rx) (local.get $bufp) (i32.const 2)))
        (if (i32.ne (i32.const 0x21 (; DROPPED=1 | (2<<4) ;)) (local.get $ret))
          (then unreachable))

        ;; two elements: {10.0, -11.0} and {12.0, -13.0}
        (if (i32.ne (i32.const 0x41200000) (i32.load offset=0 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0xc1300000) (i32.load offset=4 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0x41400000) (i32.load offset=8 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0xc1500000) (i32.load offset=12 (local.get $bufp)))
          (then unreachable))

        (call $stream.drop-readable (local.get $rx))
        (return (i32.const 0 (; EXIT ;)))
      )
      (func (export "set_cb") (param i32 i32 i32) (result i32)
        unreachable
      )
    )
    (type $R' (record (field "x" float32) (field "y" float32)))
    (export $R "r" (type $R'))
    (type $ST (stream $R))
    (canon task.return (memory (core memory $memory "mem")) (core func $task.return0))
    (canon task.return (result $ST) (memory (core memory $memory "mem")) (core func $task.return1))
    (canon stream.new $ST (core func $stream.new))
    (canon stream.read $ST (memory (core memory $memory "mem")) (core func $stream.read))
    (canon stream.write $ST (memory (core memory $memory "mem")) (core func $stream.write))
    (canon stream.drop-readable $ST (core func $stream.drop-readable))
    (canon stream.drop-writable $ST (core func $stream.drop-writable))
    (core instance $cm (instantiate $CM (with "" (instance
      (export "mem" (memory $memory "mem"))
      (export "task.return0" (func $task.return0))
      (export "task.return1" (func $task.return1))
      (export "stream.new" (func $stream.new))
      (export "stream.read" (func $stream.read))
      (export "stream.write" (func $stream.write))
      (export "stream.drop-readable" (func $stream.drop-readable))
      (export "stream.drop-writable" (func $stream.drop-writable))
    ))))
    (func (export "get") async (result $ST) (canon lift
      (core func $cm "get")
      async (memory (core memory $memory "mem")) (callback (core func $cm "get_cb"))
    ))
    (func (export "set") async (param "in" $ST) (canon lift
      (core func $cm "set")
      async (memory (core memory $memory "mem")) (callback (core func $cm "set_cb"))
    ))
  )
  (component $D
    (type $R' (record (field "x" float32) (field "y" float32)))
    (import "r" (type $R (eq $R')))
    (type $ST (stream $R))
    (import "get" (func $get async (result $ST)))
    (import "set" (func $set async (param "in" $ST)))

    (core module $Memory (memory (export "mem") 1))
    (core instance $memory (instantiate $Memory))
    (core module $DM
      (import "" "mem" (memory 1))
      (import "" "stream.new" (func $stream.new (result i64)))
      (import "" "stream.read" (func $stream.read (param i32 i32 i32) (result i32)))
      (import "" "stream.write" (func $stream.write (param i32 i32 i32) (result i32)))
      (import "" "stream.drop-readable" (func $stream.drop-readable (param i32)))
      (import "" "stream.drop-writable" (func $stream.drop-writable (param i32)))
      (import "" "get" (func $get (result i32)))
      (import "" "set" (func $set (param i32)))

      (func (export "run") (result i32)
        (local $ret i32) (local $ret64 i64)
        (local $rx i32) (local $tx i32)
        (local $bufp i32)

        ;; $rx = $C.get()
        (local.set $rx (call $get))

        ;; (stream.read $rx $bufp 2) will succeed without blocking
        (local.set $bufp (i32.const 0x40))
        (local.set $ret (call $stream.read (local.get $rx) (local.get $bufp) (i32.const 2)))
        (if (i32.ne (i32.const 0x20 (; COMPLETED=0 | (2<<4) ;)) (local.get $ret))
          (then unreachable))

        ;; two elements: {1.5, -2.5} and {3.5, -4.5}
        (if (i32.ne (i32.const 0x3fc00000) (i32.load offset=0 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0xc0200000) (i32.load offset=4 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0x40600000) (i32.load offset=8 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0xc0900000) (i32.load offset=12 (local.get $bufp)))
          (then unreachable))

        (call $stream.drop-readable (local.get $rx))

        ;; ($rx, $tx) = stream.new
        ;; $C.set($rx)
        (local.set $ret64 (call $stream.new))
        (local.set $rx (i32.wrap_i64 (local.get $ret64)))
        (local.set $tx (i32.wrap_i64 (i64.shr_u (local.get $ret64) (i64.const 32))))
        (call $set (local.get $rx))

        ;; two elements: {10.0, -11.0} and {12.0, -13.0}
        (local.set $bufp (i32.const 0x80))
        (i32.store offset=0 (local.get $bufp) (i32.const 0x41200000))
        (i32.store offset=4 (local.get $bufp) (i32.const 0xc1300000))
        (i32.store offset=8 (local.get $bufp) (i32.const 0x41400000))
        (i32.store offset=12 (local.get $bufp) (i32.const 0xc1500000))

        ;; (stream.write $tx $bufp 2) will succeed without blocking
        (local.set $ret (call $stream.write (local.get $tx) (local.get $bufp) (i32.const 2)))
        (if (i32.ne (i32.const 0x20 (; COMPLETED=0 | (2<<4) ;)) (local.get $ret))
          (then unreachable))

        (call $stream.drop-writable (local.get $tx))
        (i32.const 42)
      )
    )
    (canon stream.new $ST (core func $stream.new))
    (canon stream.read $ST async (memory (core memory $memory "mem")) (core func $stream.read))
    (canon stream.write $ST async (memory (core memory $memory "mem")) (core func $stream.write))
    (canon stream.drop-readable $ST (core func $stream.drop-readable))
    (canon stream.drop-writable $ST (core func $stream.drop-writable))
    (canon lower (func $get) (core func $get'))
    (canon lower (func $set) (core func $set'))
    (core instance $dm (instantiate $DM (with "" (instance
      (export "mem" (memory $memory "mem"))
      (export "stream.new" (func $stream.new))
      (export "stream.read" (func $stream.read))
      (export "stream.write" (func $stream.write))
      (export "stream.drop-readable" (func $stream.drop-readable))
      (export "stream.drop-writable" (func $stream.drop-writable))
      (export "get" (func $get'))
      (export "set" (func $set'))
    ))))
    (func (export "run") async (result u32) (canon lift (core func $dm "run")))
  )

  (instance $c (instantiate $C))
  (alias export $c "r" (type $r))
  (instance $d (instantiate $D
    (with "r" (type $r))
    (with "get" (func $c "get"))
    (with "set" (func $c "set"))
  ))
  (func (export "run") (alias export $d "run"))
)
(assert_return (invoke "run") (u32.const 42))

;; A `tuple<u8, u32>`: every bit pattern is a valid value too, but the layout
;; has three padding bytes after the `u8`, so this must *not* be copied
;; verbatim -- doing so would hand the reader whatever the writer happened to
;; leave in the gap. Each side leaves a sentinel in the padding of its own
;; buffer and the reader checks that its own sentinel survived.
(component
  (component $C
    (core module $Memory (memory (export "mem") 1))
    (core instance $memory (instantiate $Memory))
    (core module $CM
      (import "" "mem" (memory 1))
      (import "" "task.return0" (func $task.return0))
      (import "" "task.return1" (func $task.return1 (param i32)))
      (import "" "stream.new" (func $stream.new (result i64)))
      (import "" "stream.read" (func $stream.read (param i32 i32 i32) (result i32)))
      (import "" "stream.write" (func $stream.write (param i32 i32 i32) (result i32)))
      (import "" "stream.drop-readable" (func $stream.drop-readable (param i32)))
      (import "" "stream.drop-writable" (func $stream.drop-writable (param i32)))

      (func (export "get") (result i32)
        (local $ret i32) (local $ret64 i64)
        (local $tx i32) (local $rx i32)
        (local $bufp i32)

        ;; ($rx, $tx) = stream.new
        (local.set $ret64 (call $stream.new))
        (local.set $rx (i32.wrap_i64 (local.get $ret64)))
        (local.set $tx (i32.wrap_i64 (i64.shr_u (local.get $ret64) (i64.const 32))))

        ;; return $rx
        (call $task.return1 (local.get $rx))

        ;; two elements, with 0x11 left in the padding of each
        (local.set $bufp (i32.const 0x40))
        (i32.store offset=0 (local.get $bufp) (i32.const 0x111111aa))
        (i32.store offset=4 (local.get $bufp) (i32.const 0xdeadbeef))
        (i32.store offset=8 (local.get $bufp) (i32.const 0x111111bb))
        (i32.store offset=12 (local.get $bufp) (i32.const 0xfeedface))

        (local.set $ret (call $stream.write (local.get $tx) (local.get $bufp) (i32.const 2)))
        (if (i32.ne (i32.const 0x21 (; DROPPED=1 | (2<<4) ;)) (local.get $ret))
          (then unreachable))

        (call $stream.drop-writable (local.get $tx))
        (return (i32.const 0 (; EXIT ;)))
      )
      (func (export "get_cb") (param i32 i32 i32) (result i32)
        unreachable
      )

      (func (export "set") (param $rx i32) (result i32)
        (local $ret i32)
        (local $bufp i32)

        ;; return immediately so that the caller can just call synchronously
        (call $task.return0)

        ;; fill the read buffer with this component's own sentinel
        (local.set $bufp (i32.const 0x80))
        (i32.store offset=0 (local.get $bufp) (i32.const 0x33333333))
        (i32.store offset=8 (local.get $bufp) (i32.const 0x33333333))

        (local.set $ret (call $stream.read (local.get $rx) (local.get $bufp) (i32.const 2)))
        (if (i32.ne (i32.const 0x21 (; DROPPED=1 | (2<<4) ;)) (local.get $ret))
          (then unreachable))

        ;; the payload arrived...
        (if (i32.ne (i32.const 0xcc) (i32.load8_u offset=0 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0x01234567) (i32.load offset=4 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0xdd) (i32.load8_u offset=8 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0x89abcdef) (i32.load offset=12 (local.get $bufp)))
          (then unreachable))

        ;; ...and the writer's padding did not come with it
        (if (i32.ne (i32.const 0x333333) (i32.shr_u (i32.load offset=0 (local.get $bufp)) (i32.const 8)))
          (then unreachable))
        (if (i32.ne (i32.const 0x333333) (i32.shr_u (i32.load offset=8 (local.get $bufp)) (i32.const 8)))
          (then unreachable))

        (call $stream.drop-readable (local.get $rx))
        (return (i32.const 0 (; EXIT ;)))
      )
      (func (export "set_cb") (param i32 i32 i32) (result i32)
        unreachable
      )
    )
    (type $T (tuple u8 u32))
    (type $ST (stream $T))
    (canon task.return (memory (core memory $memory "mem")) (core func $task.return0))
    (canon task.return (result $ST) (memory (core memory $memory "mem")) (core func $task.return1))
    (canon stream.new $ST (core func $stream.new))
    (canon stream.read $ST (memory (core memory $memory "mem")) (core func $stream.read))
    (canon stream.write $ST (memory (core memory $memory "mem")) (core func $stream.write))
    (canon stream.drop-readable $ST (core func $stream.drop-readable))
    (canon stream.drop-writable $ST (core func $stream.drop-writable))
    (core instance $cm (instantiate $CM (with "" (instance
      (export "mem" (memory $memory "mem"))
      (export "task.return0" (func $task.return0))
      (export "task.return1" (func $task.return1))
      (export "stream.new" (func $stream.new))
      (export "stream.read" (func $stream.read))
      (export "stream.write" (func $stream.write))
      (export "stream.drop-readable" (func $stream.drop-readable))
      (export "stream.drop-writable" (func $stream.drop-writable))
    ))))
    (func (export "get") async (result $ST) (canon lift
      (core func $cm "get")
      async (memory (core memory $memory "mem")) (callback (core func $cm "get_cb"))
    ))
    (func (export "set") async (param "in" $ST) (canon lift
      (core func $cm "set")
      async (memory (core memory $memory "mem")) (callback (core func $cm "set_cb"))
    ))
  )
  (component $D
    (type $T (tuple u8 u32))
    (type $ST (stream $T))
    (import "get" (func $get async (result $ST)))
    (import "set" (func $set async (param "in" $ST)))

    (core module $Memory (memory (export "mem") 1))
    (core instance $memory (instantiate $Memory))
    (core module $DM
      (import "" "mem" (memory 1))
      (import "" "stream.new" (func $stream.new (result i64)))
      (import "" "stream.read" (func $stream.read (param i32 i32 i32) (result i32)))
      (import "" "stream.write" (func $stream.write (param i32 i32 i32) (result i32)))
      (import "" "stream.drop-readable" (func $stream.drop-readable (param i32)))
      (import "" "stream.drop-writable" (func $stream.drop-writable (param i32)))
      (import "" "get" (func $get (result i32)))
      (import "" "set" (func $set (param i32)))

      (func (export "run") (result i32)
        (local $ret i32) (local $ret64 i64)
        (local $rx i32) (local $tx i32)
        (local $bufp i32)

        ;; $rx = $C.get()
        (local.set $rx (call $get))

        ;; fill the read buffer with this component's own sentinel
        (local.set $bufp (i32.const 0x40))
        (i32.store offset=0 (local.get $bufp) (i32.const 0x22222222))
        (i32.store offset=8 (local.get $bufp) (i32.const 0x22222222))

        (local.set $ret (call $stream.read (local.get $rx) (local.get $bufp) (i32.const 2)))
        (if (i32.ne (i32.const 0x20 (; COMPLETED=0 | (2<<4) ;)) (local.get $ret))
          (then unreachable))

        ;; the payload arrived...
        (if (i32.ne (i32.const 0xaa) (i32.load8_u offset=0 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0xdeadbeef) (i32.load offset=4 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0xbb) (i32.load8_u offset=8 (local.get $bufp)))
          (then unreachable))
        (if (i32.ne (i32.const 0xfeedface) (i32.load offset=12 (local.get $bufp)))
          (then unreachable))

        ;; ...and the writer's padding did not come with it
        (if (i32.ne (i32.const 0x222222) (i32.shr_u (i32.load offset=0 (local.get $bufp)) (i32.const 8)))
          (then unreachable))
        (if (i32.ne (i32.const 0x222222) (i32.shr_u (i32.load offset=8 (local.get $bufp)) (i32.const 8)))
          (then unreachable))

        (call $stream.drop-readable (local.get $rx))

        ;; ($rx, $tx) = stream.new
        ;; $C.set($rx)
        (local.set $ret64 (call $stream.new))
        (local.set $rx (i32.wrap_i64 (local.get $ret64)))
        (local.set $tx (i32.wrap_i64 (i64.shr_u (local.get $ret64) (i64.const 32))))
        (call $set (local.get $rx))

        ;; two elements, with 0x44 left in the padding of each
        (local.set $bufp (i32.const 0x80))
        (i32.store offset=0 (local.get $bufp) (i32.const 0x444444cc))
        (i32.store offset=4 (local.get $bufp) (i32.const 0x01234567))
        (i32.store offset=8 (local.get $bufp) (i32.const 0x444444dd))
        (i32.store offset=12 (local.get $bufp) (i32.const 0x89abcdef))

        (local.set $ret (call $stream.write (local.get $tx) (local.get $bufp) (i32.const 2)))
        (if (i32.ne (i32.const 0x20 (; COMPLETED=0 | (2<<4) ;)) (local.get $ret))
          (then unreachable))

        (call $stream.drop-writable (local.get $tx))
        (i32.const 42)
      )
    )
    (canon stream.new $ST (core func $stream.new))
    (canon stream.read $ST async (memory (core memory $memory "mem")) (core func $stream.read))
    (canon stream.write $ST async (memory (core memory $memory "mem")) (core func $stream.write))
    (canon stream.drop-readable $ST (core func $stream.drop-readable))
    (canon stream.drop-writable $ST (core func $stream.drop-writable))
    (canon lower (func $get) (core func $get'))
    (canon lower (func $set) (core func $set'))
    (core instance $dm (instantiate $DM (with "" (instance
      (export "mem" (memory $memory "mem"))
      (export "stream.new" (func $stream.new))
      (export "stream.read" (func $stream.read))
      (export "stream.write" (func $stream.write))
      (export "stream.drop-readable" (func $stream.drop-readable))
      (export "stream.drop-writable" (func $stream.drop-writable))
      (export "get" (func $get'))
      (export "set" (func $set'))
    ))))
    (func (export "run") async (result u32) (canon lift (core func $dm "run")))
  )

  (instance $c (instantiate $C))
  (instance $d (instantiate $D
    (with "get" (func $c "get"))
    (with "set" (func $c "set"))
  ))
  (func (export "run") (alias export $d "run"))
)
(assert_return (invoke "run") (u32.const 42))
