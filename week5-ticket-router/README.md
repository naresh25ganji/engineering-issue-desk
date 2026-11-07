# Week 5 support ticket router

Completed LoRA training of `Qwen/Qwen3-1.7B-Base` on a Colab T4, merged the adapter into an unquantized standalone model, and verified a fresh reload.

The original end-token generation protocol failed on all 117 tuned-model tickets. An explicit `label_stop` repair stops actual generation when its entire new text exactly matches an allowed label. With that decoder, the adapter and merged model each scored **71.79% accuracy, 0.6895 macro F1**, with no invalid outputs; five illustrative smoke checks passed. The repair was chosen after observing the failures, so these are **exploratory follow-up results**. Full results and the original failures are preserved in [RESULTS.md](RESULTS.md). The regression suite passed 73 tests.

[Colab, project bundle, and saved adapter links](LAUNCH.md).

## Reproduce

From the repository root:

```bash
cd week5-ticket-router
uv sync
uv run python -m router_lab prepare --output artifacts/prepared
uv run python -m router_lab baselines --data artifacts/prepared --output artifacts/baselines
uv run python scripts/package_project.py
```

The course CSV has 585 tickets across seven queues: 468 development and 117 frozen outer held-out tickets. Training uses a separate 10% inner validation split from the development data. The outer tickets are excluded from training and checkpoint selection. Source attribution, versions, and fingerprints are in [SOURCE.md](SOURCE.md).

1. Open [the notebook in Colab](LAUNCH.md), select a T4 GPU, and upload `artifacts/week5-ticket-router.zip` in its first code cell.
2. Run the data checks, set `RUN_GPU_STEPS = True`, and run setup, training, original evaluation, merge, and fresh-process checks.
3. Explicitly set `RUN_LABEL_STOP_REPAIR = True` in the optional follow-up section to reproduce the separate matched base/adapter/merged repair evaluation.
4. Run the final evidence and adapter downloads. Save the notebook with its real outputs.

Both flags are **False in the source notebook**. Each experiment uses a new output directory; existing reports are preserved. No API key is required for the public model. For local model dependencies use `uv sync --extra inference`; run checks with `uv run --extra inference pytest -q`.

The notebook uses visible LLaMA-Factory CLI commands in place of the handout's LLaMA Board interface. It keeps the training configuration and logs inspectable without a public web UI tunnel.

## Completed training configuration

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3-1.7B-Base` |
| Model revision | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| LLaMA-Factory commit | `702005ce5237ae10f3c572c3f7741ecfd1422fc9` |
| Method | NF4 4-bit base, LoRA rank 8 / alpha 16 |
| Precision | FP16 on Tesla T4 |
| Learning rate / epochs | `1e-4` / 3 |
| Batch / accumulation / cutoff | 1 / 8 / 512 tokens |
| Checkpoint selection | Lowest inner-validation loss |

Full merged weights remain only in the Colab runtime. The [saved adapter restore bundle](LAUNCH.md) can regenerate them with the pinned base model. Saving the notebook alone does not preserve weights.

## Submit

Use a concise Word write-up imported as a Google Doc in Drive, with actual run screenshots and the original failure plus exploratory repair results disclosed. The custom-project route permits a GitHub repository and Loom. The handout's Builder deadline is November 8, 2026 at 11:59 PM Pacific. [Submission form](https://forms.gle/cmB8mLtpjNCNsjrZ6).
