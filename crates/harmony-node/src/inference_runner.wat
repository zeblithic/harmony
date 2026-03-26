(module
  ;; Host function imports (harmony namespace)
  (import "harmony" "model_load" (func $model_load (param i32 i32) (result i32)))
  (import "harmony" "tokenize" (func $tokenize (param i32 i32 i32 i32) (result i32)))
  (import "harmony" "detokenize" (func $detokenize (param i32 i32 i32 i32) (result i32)))
  (import "harmony" "forward" (func $forward (param i32 i32) (result i32)))
  (import "harmony" "sample" (func $sample (param i32 i32) (result i32)))

  ;; 2 pages = 128KB
  (memory (export "memory") 2)

  (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
    (local $result i32)
    (local $prompt_len i32)
    (local $prompt_ptr i32)
    (local $token_bytes i32)
    (local $params_ptr i32)
    (local $max_tokens i32)
    (local $token_id i32)
    (local $gen_count i32)
    (local $gen_ptr i32)
    (local $output_bytes i32)
    (local $out_ptr i32)

    ;; 1. Load model
    (local.set $result
      (call $model_load (local.get $input_ptr) (i32.add (local.get $input_ptr) (i32.const 32))))
    (if (i32.ne (local.get $result) (i32.const 0))
      (then (return (i32.const -1))))

    ;; 2. Read prompt_len and prompt_ptr
    (local.set $prompt_len
      (i32.load (i32.add (local.get $input_ptr) (i32.const 64))))
    (local.set $prompt_ptr
      (i32.add (local.get $input_ptr) (i32.const 68)))

    ;; Guard: full input tail (prompt + 20B params + 4B max_tokens) must not
    ;; reach the single-token scratch slot at 32764.
    ;; End of input = prompt_ptr + prompt_len + 24.
    (if (i32.gt_u
          (i32.add (i32.add (local.get $prompt_ptr) (local.get $prompt_len)) (i32.const 24))
          (i32.const 32764))
      (then (return (i32.const -1))))

    ;; 3. Tokenize prompt
    (local.set $token_bytes
      (call $tokenize
        (local.get $prompt_ptr) (local.get $prompt_len)
        (i32.const 32768) (i32.const 8192)))
    (if (i32.lt_s (local.get $token_bytes) (i32.const 0))
      (then (return (i32.const -1))))

    ;; 4. Forward pass (prefill)
    (local.set $result
      (call $forward (i32.const 32768) (local.get $token_bytes)))
    (if (i32.ne (local.get $result) (i32.const 0))
      (then (return (i32.const -1))))

    ;; 5. Read params and max_tokens
    (local.set $params_ptr
      (i32.add (local.get $prompt_ptr) (local.get $prompt_len)))
    (local.set $max_tokens
      (i32.load (i32.add (local.get $params_ptr) (i32.const 20))))

    ;; 6. Autoregressive loop
    (local.set $gen_count (i32.const 0))
    (local.set $gen_ptr (i32.const 40960))

    (block $done
      (loop $gen_loop
        (local.set $token_id
          (call $sample (local.get $params_ptr) (i32.const 20)))
        (br_if $done (i32.lt_s (local.get $token_id) (i32.const 0)))

        (i32.store
          (i32.add (local.get $gen_ptr) (i32.shl (local.get $gen_count) (i32.const 2)))
          (local.get $token_id))
        (local.set $gen_count (i32.add (local.get $gen_count) (i32.const 1)))

        (br_if $done (i32.ge_u (local.get $gen_count) (local.get $max_tokens)))

        (i32.store (i32.const 32764) (local.get $token_id))
        (local.set $result (call $forward (i32.const 32764) (i32.const 4)))
        (br_if $done (i32.ne (local.get $result) (i32.const 0)))

        (br $gen_loop)))

    ;; 7. Detokenize
    (if (i32.gt_u (local.get $gen_count) (i32.const 0))
      (then
        (local.set $output_bytes
          (call $detokenize
            (local.get $gen_ptr)
            (i32.shl (local.get $gen_count) (i32.const 2))
            (i32.const 49152) (i32.const 16384)))
        (if (i32.lt_s (local.get $output_bytes) (i32.const 0))
          (then (return (i32.const -1)))))
      (else
        (local.set $output_bytes (i32.const 0))))

    ;; 8. Copy to output area
    (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
    (if (i32.gt_u (local.get $output_bytes) (i32.const 0))
      (then
        (memory.copy
          (local.get $out_ptr)
          (i32.const 49152)
          (local.get $output_bytes))))

    ;; 9. Return output byte count
    (local.get $output_bytes))
)
