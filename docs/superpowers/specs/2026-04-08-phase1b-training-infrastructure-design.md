# Phase 1b: Training Infrastructure Design

> Second sub-project of Phase 1 (Actual Training). Adds mixed precision,
> gradient accumulation, gradient clipping, and CSV logging to the training
> loop so real training runs are practical on consumer GPUs.

**Dependencies:** Phase 1a (data pipeline — provides tokenized datasets and
validation loss loop)

## Goal

Make the existing training loop production-ready for multi-hour runs on
consumer GPUs (RTX 3070 8GB, RTX 5080 16GB, RTX 4090 24GB, Apple M5 32GB
unified). Add bf16 mixed precision to halve memory and double throughput,
gradient accumulation to simulate larger batch sizes, gradient clipping to
prevent training instability, and CSV logging to record loss curves for
offline analysis.

## Mixed Precision (bf16)

**Strategy:** PyTorch `torch.autocast` wraps the forward pass and loss
computation. Matrix multiplications run in bf16; reductions (loss) stay
in fp32 automatically. No `GradScaler` needed for bf16. Model weights
remain fp32 — autocast casts them on-the-fly.

**CLI arg:** `--dtype`, choices `float32` and `bfloat16`, default `bfloat16`.

The forward pass changes from:

```python
logits = model(input_ids)
loss = F.cross_entropy(...)
```

to:

```python
with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
    logits = model(input_ids)
    loss = F.cross_entropy(...)
```

The validation path (`compute_validation_loss`) gets the same autocast
wrapper for consistency.

### Device Compatibility

| GPU | bf16 Support |
|---|---|
| RTX 4090 (Ampere successor) | Native tensor cores |
| RTX 5080 (Blackwell) | Native tensor cores |
| RTX 3070 (Ampere) | Native tensor cores |
| Apple M5 (MPS) | Supported via MPS backend |
| CPU | Supported (no acceleration) |

## Gradient Accumulation

**CLI arg:** `--grad-accum-steps`, default 1 (no accumulation). Effective
batch size = `batch_size * grad_accum_steps`.

Loss is scaled by `1 / grad_accum_steps` before `.backward()` so gradient
magnitude is independent of the accumulation factor.

```python
for step in range(total_steps):
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        batch = next(dataloader).to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        with torch.autocast(...):
            logits = model(input_ids)
            loss = F.cross_entropy(...) / grad_accum_steps
        loss.backward()
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
```

### Step Counting Convention

The `step` counter counts **optimizer steps**, not micro-batches.
`--steps 1000` with `--grad-accum-steps 4` does 4,000 forward/backward
passes but 1,000 optimizer updates. Learning rate schedules, checkpointing,
logging, and all step-based arguments operate on optimizer steps.

### Loss Reporting

The last micro-batch loss (before the `1/grad_accum_steps` scaling) is
printed and logged. This is a diagnostic signal, not the exact accumulated
average — but it tracks training progress accurately enough without the
complexity of a running mean across micro-batches.

## Gradient Clipping

**CLI arg:** `--max-grad-norm`, default 1.0. Applied after all micro-batches
accumulate, right before `optimizer.step()`.

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
```

The returned `grad_norm` is the pre-clipping total norm — logged to CSV as
a training health signal. If `grad_norm` consistently exceeds
`max_grad_norm`, clipping is actively preventing instability.

Set `--max-grad-norm 0` to disable clipping entirely (the call is skipped).

## CSV Logging

**CLI arg:** `--log-file`, default `None` (stdout-only, backwards compatible).

### Columns

| Column | Description |
|---|---|
| `step` | Optimizer step number |
| `loss` | Last micro-batch loss (unscaled) |
| `val_loss` | Validation loss (empty when not computed) |
| `lr` | Current learning rate |
| `grad_norm` | Pre-clipping gradient norm |
| `dt_ms` | Wall-clock milliseconds for this step |

### Behavior

- Header written once when file is created.
- A row is written at every print step (every 10 optimizer steps).
- Validation loss fills the `val_loss` column at checkpoint boundaries.
- File opened in append mode so restarts don't overwrite earlier data.
- `dt_ms` is computed from `time.time()` delta — useful for spotting
  thermal throttling across multi-hour runs.

## New CLI Args Summary

| Arg | Default | Description |
|---|---|---|
| `--dtype` | `bfloat16` | Training precision (`float32` or `bfloat16`) |
| `--grad-accum-steps` | 1 | Micro-batches per optimizer step |
| `--max-grad-norm` | 1.0 | Max gradient norm for clipping (0 to disable) |
| `--log-file` | None | Path to CSV log file (omit for stdout-only) |

## Files Changed

| File | Action | ~Lines |
|---|---|---|
| `training/ct87/train.py` | Modify | +80 |
| `training/tests/test_train.py` | Modify | +60 |

## Dependencies

No new dependencies. All features use PyTorch builtins (`torch.autocast`,
`torch.nn.utils.clip_grad_norm_`) and Python stdlib (`csv`, `time`).

## Testing

### New Tests (`training/tests/test_train.py`)

- **Mixed precision:** Verify training step runs under autocast bf16 on CPU,
  produces finite loss, and model weights are still fp32 after the step.
- **Gradient accumulation:** Run N steps with `grad_accum_steps=1` and
  N/K steps with `grad_accum_steps=K`, verify both produce finite loss
  and the accumulated path calls `optimizer.step()` fewer times.
- **Gradient clipping:** Verify `clip_grad_norm_` is effective — run a step
  with very small `max_grad_norm`, verify returned grad_norm >= max_grad_norm
  (clipping activated). Verify `max_grad_norm=0` skips clipping.
- **CSV logging:** Run a short training loop with `--log-file`, verify the
  CSV exists, has correct headers, correct number of rows, and all numeric
  columns parse as floats.

### Existing Tests

All existing tests remain unchanged and must continue to pass. The default
`--dtype bfloat16` does not affect existing tests because they use the
Python API directly (not the CLI), and autocast is only enabled in `main()`.

## What Does NOT Change

- Model architecture (`model.py`)
- Optimizer / LR schedule (`optim.py`)
- Data pipeline (`prepare_data.py`)
- Checkpoint format (safetensors weights + optimizer .pt)
- Existing CLI args and their defaults
- Synthetic data path
- Validation loss computation (same logic, just wrapped in autocast)

## Scope Boundary

**In scope:**
- `torch.autocast` bf16 with `--dtype` CLI arg
- Gradient accumulation with `--grad-accum-steps` CLI arg
- Gradient clipping with `--max-grad-norm` CLI arg
- CSV logging with `--log-file` CLI arg
- Tests for each feature

**Out of scope:**
- Wandb / TensorBoard (easy to add later, deferred)
- Multi-GPU / distributed training
- Automatic batch size tuning
- Checkpoint resume (picking up from a saved optimizer step)
- Actually running a training session (Phase 1c)
- fp16 with GradScaler (bf16 is sufficient for all target hardware)
