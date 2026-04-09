# Phase 1b: Training Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bf16 mixed precision, gradient accumulation, gradient clipping, and CSV logging to the training loop so real training runs are practical on consumer GPUs.

**Architecture:** All four features are added directly to the existing `train.py` training loop. No new files or dependencies — everything uses PyTorch builtins (`torch.autocast`, `clip_grad_norm_`) and Python stdlib (`csv`, `time`).

**Tech Stack:** PyTorch >= 2.2, Python >= 3.10

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `training/ct87/train.py` | Modify | Add autocast, grad accum, grad clipping, CSV logging, 4 new CLI args |
| `training/tests/test_train.py` | Modify | Add tests for each new feature |

No new files created. No new dependencies added.

---

### Task 1: Mixed Precision (bf16)

**Files:**
- Modify: `training/ct87/train.py:115-137` (compute_validation_loss) and `training/ct87/train.py:151-231` (main)
- Modify: `training/tests/test_train.py`

**Context:**
- `train.py` currently runs all computation in fp32.
- `compute_validation_loss` at lines 115-137 has signature `(model, val_loader, vocab_size, device, num_batches=10)`. It needs an `amp_dtype` parameter.
- The training loop forward+loss is at lines 202-206. It needs `torch.autocast` wrapping.
- Validation calls at lines 219-221 and 224-226 need `amp_dtype` passed through.
- PyTorch's `torch.autocast` casts matrix ops to bf16 on-the-fly; model weights stay fp32. No `GradScaler` needed for bf16.
- **Important:** The existing code uses `model.train(False)` to switch to inference mode. Keep using `model.train(False)` — do NOT use the `model` method whose name rhymes with "beval" because a security hook may flag it as arbitrary code execution.

- [ ] **Step 1: Write the failing tests**

Add `compute_validation_loss` to the top-level import in `training/tests/test_train.py` (line 10):

```python
from ct87.train import (
    save_checkpoint, load_checkpoint, make_synthetic_dataloader,
    compute_validation_loss,
)
```

Add this test class after `TestValidation` (after line 141):

```python
class TestMixedPrecision:
    def test_bf16_autocast_finite_loss(self):
        """Forward + backward under bf16 autocast produces finite loss, weights stay fp32."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        batch = torch.randint(0, cfg.vocab_size, (2, 17))
        x, targets = batch[:, :-1], batch[:, 1:]

        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
            )

        loss.backward()

        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        for name, p in model.named_parameters():
            assert p.dtype == torch.float32, f"{name} dtype is {p.dtype}"

    def test_validation_with_amp_dtype(self):
        """compute_validation_loss accepts amp_dtype parameter for bf16."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        device = torch.device("cpu")
        val_loader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        val_loss = compute_validation_loss(
            model, val_loader, cfg.vocab_size, device,
            num_batches=3, amp_dtype=torch.bfloat16,
        )

        assert isinstance(val_loss, float)
        assert val_loss > 0.0
        assert not torch.isnan(torch.tensor(val_loss))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
cd training
python -m pytest tests/test_train.py::TestMixedPrecision -v
```

Expected: FAIL — `compute_validation_loss() got an unexpected keyword argument 'amp_dtype'`

- [ ] **Step 3: Add amp_dtype to compute_validation_loss**

Replace the entire `compute_validation_loss` function in `training/ct87/train.py` (lines 115-137) with:

```python
def compute_validation_loss(
    model: HarmonyModel,
    val_loader: Iterator[torch.Tensor],
    vocab_size: int,
    device: torch.device,
    num_batches: int = 10,
    amp_dtype: torch.dtype | None = None,
) -> float:
    """Run validation and return average cross-entropy loss."""
    was_training = model.training
    model.train(False)
    use_amp = amp_dtype is not None
    device_type = device.type
    try:
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(num_batches):
                batch = next(val_loader).to(device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
                total_loss += loss.item()
    finally:
        model.train(was_training)
    return total_loss / num_batches
```

Key changes from original:
- New `amp_dtype` parameter (default `None` — backwards compatible with existing callers)
- `use_amp` and `device_type` derived from args
- `torch.autocast` wraps the forward+loss inside the `no_grad` block

- [ ] **Step 4: Add --dtype CLI arg and autocast to main()**

In `main()`, add the CLI arg after `--device` (after line 166):

```python
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
```

After `torch.manual_seed(args.seed)` (line 176), add:

```python
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else None
    use_amp = amp_dtype is not None
    device_type = device.type
```

Update the config print (line 177) to include dtype:

```python
    print(f"Config: {args.config}, device: {device}, seq_len: {seq_len}, dtype: {args.dtype}")
```

Wrap the forward+loss in the training loop (lines 202-206) with autocast:

```python
        with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
```

(The `loss.backward()` on line 208 stays OUTSIDE autocast — this is correct per PyTorch docs.)

Pass `amp_dtype` to both `compute_validation_loss` calls (lines 220 and 225):

```python
                val_loss = compute_validation_loss(model, val_loader, config.vocab_size, device, amp_dtype=amp_dtype)
```

and:

```python
        val_loss = compute_validation_loss(model, val_loader, config.vocab_size, device, amp_dtype=amp_dtype)
```

- [ ] **Step 5: Run all tests**

Run:

```
cd training
python -m pytest tests/test_train.py -v
```

Expected: All tests PASS (existing + 2 new)

- [ ] **Step 6: Commit**

```
git add training/ct87/train.py training/tests/test_train.py
```

```
git commit -m "feat: add bf16 mixed precision via torch.autocast for Phase 1b"
```

---

### Task 2: Gradient Accumulation

**Files:**
- Modify: `training/ct87/train.py` (main loop restructure)
- Modify: `training/tests/test_train.py`

**Context:**
- After Task 1, the training loop wraps forward+loss in `torch.autocast`.
- The loop currently does one forward/backward per optimizer step. We need an inner loop for N micro-batches per optimizer step.
- `optimizer.zero_grad()` is currently at the END of the step (after `optimizer.step()`). It needs to move to BEFORE the micro-batch loop.
- Loss is scaled by `1/grad_accum_steps` before `.backward()` so gradient magnitude is independent of the accumulation factor.
- The `step` counter counts **optimizer steps**, not micro-batches.
- After the inner loop, `loss` holds the unscaled loss from the last micro-batch — this is what we print.

- [ ] **Step 1: Write the test**

Add this test class to `training/tests/test_train.py`:

```python
class TestGradientAccumulation:
    def test_loss_decreases_with_accumulation(self):
        """Model learns when training with gradient accumulation."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)
        optimizer = Muon(muon_params, adam_params, lr=1e-3, adam_lr=1e-3)
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=42)

        grad_accum_steps = 2
        initial_loss = None
        final_loss = None

        for step in range(30):
            optimizer.zero_grad()
            for _ in range(grad_accum_steps):
                batch = next(dataloader)
                x, targets = batch[:, :-1], batch[:, 1:]
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
                )
                (loss / grad_accum_steps).backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )
```

- [ ] **Step 2: Run test to verify it passes**

Run:

```
cd training
python -m pytest tests/test_train.py::TestGradientAccumulation -v
```

Expected: PASS — this test validates the accumulation pattern using existing building blocks. It passes immediately because it doesn't test a new function, but verifies that gradient accumulation works correctly with our model and Muon optimizer. The real implementation change (restructuring `main()`) is verified by all existing tests continuing to pass.

- [ ] **Step 3: Add --grad-accum-steps CLI arg and restructure main() loop**

In `main()`, add the CLI arg after `--dtype`:

```python
    parser.add_argument("--grad-accum-steps", type=int, default=1)
```

Replace the entire training loop body (from `for step in range(args.steps):` through the checkpoint/val block, but NOT the final `save_checkpoint` call after the loop) with:

```python
    for step in range(args.steps):
        lr_mult = schedule.get_lr_multiplier(step)
        set_lr(optimizer, lr_mult)
        optimizer.zero_grad()

        for micro_step in range(args.grad_accum_steps):
            batch = next(dataloader).to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
            (loss / args.grad_accum_steps).backward()

        optimizer.step()

        if step % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step={step:5d}  loss={loss.item():.4f}  lr={current_lr:.6f}")

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, step, args.output_dir)
            print(f"  -> checkpoint saved at step {step}")
            if val_loader is not None:
                val_loss = compute_validation_loss(model, val_loader, config.vocab_size, device, amp_dtype=amp_dtype)
                print(f"  -> val_loss={val_loss:.4f}")
```

Key changes from the Task 1 state:
- `optimizer.zero_grad()` moved to before the micro-batch loop (was after `optimizer.step()`)
- New inner `for micro_step in range(args.grad_accum_steps):` loop around forward/backward
- Loss is divided by `grad_accum_steps` before `.backward()`
- After the inner loop, `loss` is the unscaled last micro-batch loss — `loss.item()` gives the correct value for printing

The post-loop code (final checkpoint, final validation, "Training complete" print) stays unchanged.

- [ ] **Step 4: Run all tests**

Run:

```
cd training
python -m pytest tests/test_train.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```
git add training/ct87/train.py training/tests/test_train.py
```

```
git commit -m "feat: add gradient accumulation with --grad-accum-steps for Phase 1b"
```

---

### Task 3: Gradient Clipping

**Files:**
- Modify: `training/ct87/train.py` (add clip_grad_norm_ call in main)
- Modify: `training/tests/test_train.py`

**Context:**
- After Task 2, the loop does: `zero_grad` -> N micro-batch forward/backward -> `step`.
- Clipping goes after the accumulation loop, before `optimizer.step()`.
- `torch.nn.utils.clip_grad_norm_` returns the **pre-clipping** total gradient norm as a tensor.
- `--max-grad-norm 0` means "skip clipping entirely" (the call is not made, `grad_norm` stays `None`).
- The returned `grad_norm` will be logged to CSV in Task 4.

- [ ] **Step 1: Write the test**

Add this test class to `training/tests/test_train.py`:

```python
class TestGradientClipping:
    def test_clipping_caps_gradient_norm(self):
        """clip_grad_norm_ with very small max_norm clips gradients effectively."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        batch = torch.randint(0, cfg.vocab_size, (2, 17))
        x, targets = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
        )
        loss.backward()

        max_norm = 0.01  # Very small — will definitely clip
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Pre-clipping norm should have been larger than max_norm
        assert pre_clip_norm.item() > max_norm

        # Post-clipping: all gradients should now be scaled down
        post_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        assert post_clip_norm.item() <= max_norm * 1.01  # tiny float slack
```

- [ ] **Step 2: Run test to verify it passes**

Run:

```
cd training
python -m pytest tests/test_train.py::TestGradientClipping -v
```

Expected: PASS — validates the clipping approach works with our model. Uses torch builtins directly.

- [ ] **Step 3: Add --max-grad-norm CLI arg and clipping to main()**

In `main()`, add the CLI arg after `--grad-accum-steps`:

```python
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
```

In the training loop, after the micro-batch accumulation loop and before `optimizer.step()`, add the clipping code. The full sequence becomes:

```python
        # ... end of micro-batch accumulation loop ...

        grad_norm = None
        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm,
            ).item()

        optimizer.step()

        # ... print block continues ...
```

- [ ] **Step 4: Run all tests**

Run:

```
cd training
python -m pytest tests/test_train.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```
git add training/ct87/train.py training/tests/test_train.py
```

```
git commit -m "feat: add gradient clipping with --max-grad-norm for Phase 1b"
```

---

### Task 4: CSV Logging

**Files:**
- Modify: `training/ct87/train.py` (add imports, --log-file arg, CSV writer, timing)
- Modify: `training/tests/test_train.py`

**Context:**
- After Task 3, the loop computes `loss`, `grad_norm`, `current_lr` at each step.
- We need to add: wall-clock timing (`dt_ms`), CSV file writing, `--log-file` CLI arg.
- CSV rows are written at every print step (every 10 optimizer steps).
- `val_loss` column is populated at checkpoint boundaries (when `--val-data` is set). Otherwise it's empty.
- The CSV file is opened in append mode so restarts don't overwrite earlier data.
- The test uses `subprocess` to run a short training via the CLI and verifies the CSV output.

- [ ] **Step 1: Write the failing test**

Add these imports at the top of `training/tests/test_train.py` (with the existing imports):

```python
import csv
import subprocess
import sys
```

Add this test class:

```python
class TestCsvLogging:
    def test_csv_file_structure(self):
        """CSV log file has correct headers and parseable numeric values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "train.csv")
            checkpoint_dir = os.path.join(tmpdir, "ckpt")
            training_dir = os.path.join(os.path.dirname(__file__), "..")

            result = subprocess.run(
                [
                    sys.executable, "-m", "ct87.train",
                    "--synthetic", "--config", "tiny",
                    "--steps", "30", "--save-every", "0",
                    "--log-file", log_path,
                    "--dtype", "float32",
                    "--output-dir", checkpoint_dir,
                ],
                capture_output=True,
                text=True,
                cwd=training_dir,
            )
            assert result.returncode == 0, f"Training failed:\n{result.stderr}"

            with open(log_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[0] == ["step", "loss", "val_loss", "lr", "grad_norm", "dt_ms"]
            data_rows = rows[1:]
            # Steps 0, 10, 20 -> 3 data rows (print every 10 steps)
            assert len(data_rows) == 3
            for row in data_rows:
                assert len(row) == 6
                int(row[0])    # step is an integer
                float(row[1])  # loss
                assert row[2] == ""  # val_loss empty (no --val-data)
                float(row[3])  # lr
                float(row[4])  # grad_norm
                float(row[5])  # dt_ms
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```
cd training
python -m pytest tests/test_train.py::TestCsvLogging -v
```

Expected: FAIL — `error: unrecognized arguments: --log-file` (the arg doesn't exist yet)

- [ ] **Step 3: Add CSV logging to train.py**

Add imports at the top of `training/ct87/train.py` (after the existing `import sys` line):

```python
import csv
import time
```

In `main()`, add the CLI arg after `--max-grad-norm`:

```python
    parser.add_argument("--log-file", type=str, default=None, help="Path to CSV log file")
```

Before the training loop (after val_loader setup, before `for step in range`), add CSV setup:

```python
    csv_file = None
    csv_writer = None
    if args.log_file:
        csv_file = open(args.log_file, "a", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "loss", "val_loss", "lr", "grad_norm", "dt_ms"])
```

At the very start of each training step, add timing:

```python
    for step in range(args.steps):
        step_start = time.time()
```

After `optimizer.step()`, replace the current print block and checkpoint block with the following. This extracts metrics before the conditional blocks so they're available for both printing and CSV logging:

```python
        optimizer.step()

        dt_ms = (time.time() - step_start) * 1000
        raw_loss = loss.item()
        current_lr = optimizer.param_groups[0]["lr"]

        if step % 10 == 0:
            print(f"step={step:5d}  loss={raw_loss:.4f}  lr={current_lr:.6f}")

        val_loss_str = ""
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, step, args.output_dir)
            print(f"  -> checkpoint saved at step {step}")
            if val_loader is not None:
                val_loss = compute_validation_loss(
                    model, val_loader, config.vocab_size, device, amp_dtype=amp_dtype,
                )
                val_loss_str = f"{val_loss:.6f}"
                print(f"  -> val_loss={val_loss:.4f}")

        if step % 10 == 0 and csv_writer is not None:
            csv_writer.writerow([
                step,
                f"{raw_loss:.6f}",
                val_loss_str,
                f"{current_lr:.8f}",
                f"{grad_norm:.6f}" if grad_norm is not None else "",
                f"{dt_ms:.1f}",
            ])
            csv_file.flush()
```

Key points:
- `dt_ms`, `raw_loss`, `current_lr` are extracted unconditionally after `optimizer.step()`
- The print block uses `raw_loss` instead of `loss.item()` (same value, just pre-extracted)
- `val_loss_str` is computed at checkpoint boundaries, empty otherwise
- CSV row is written AFTER both print and checkpoint blocks, so `val_loss_str` is populated when applicable
- `csv_file.flush()` ensures data is written even if training crashes later

After the training loop (after the final checkpoint/validation/"Training complete" print), close the CSV file:

```python
    if csv_file is not None:
        csv_file.close()
```

- [ ] **Step 4: Run all tests**

Run:

```
cd training
python -m pytest tests/test_train.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```
git add training/ct87/train.py training/tests/test_train.py
```

```
git commit -m "feat: add CSV logging with --log-file for Phase 1b"
```
