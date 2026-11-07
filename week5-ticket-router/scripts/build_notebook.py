"""Build the reviewable Colab notebook from small, runnable project commands."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda text: cells.append(nb.v4.new_markdown_cell(text))
code = lambda text: cells.append(nb.v4.new_code_cell(text))

md("""# Week 5 support ticket router

Train a LoRA adapter for **Qwen3 1.7B Base**, merge it into a standalone model, and compare actual routing results on 117 frozen held-out tickets.

The project fixes the source notebook's per-class label order, preserves invalid outputs, and adds a comparison with the same prompt and decoder for base and tuned models. Source CSV labels have not received independent human signoff. No fine-tuning result is supplied in this notebook.

Use **Runtime → Change runtime type → GPU**. The configuration targets a T4 with FP16; available free GPUs vary. Training runs through notebook-visible LLaMA Factory CLI commands instead of a public LLaMA Board tunnel. The underlying LoRA training and saved configuration remain inspectable.
""")
md("""## 1 Load the project

Upload `week5-ticket-router.zip` from the implementation package. This cell never replaces an existing project folder. For a clean new experiment, use a fresh runtime or a separate directory.
""")
code('''from pathlib import Path
import os, sys, json, subprocess, zipfile

def stream(*command, cwd=None):
    command = list(map(str, command))
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        return_code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return subprocess.CompletedProcess(command, return_code)

if os.environ.get("WEEK5_OFFLINE_TEST") == "1":
    PROJECT = Path.cwd()
elif Path("/content").is_dir():
    PROJECT = Path("/content/week5-ticket-router")
    if not PROJECT.exists():
        from google.colab import files
        uploaded = files.upload()
        candidates = [name for name in uploaded if name.endswith(".zip")]
        if len(candidates) != 1:
            raise ValueError("Upload exactly one week5-ticket-router.zip file.")
        with zipfile.ZipFile(candidates[0]) as archive:
            for member in archive.infolist():
                target = (Path("/content") / member.filename).resolve()
                if not target.is_relative_to(PROJECT.resolve()) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive contains a path outside the project or a symbolic link.")
            archive.extractall("/content")
else:
    PROJECT = Path.cwd() if (Path.cwd() / "router_lab").is_dir() else Path.cwd().parent
if not (PROJECT / "router_lab/config.py").is_file():
    raise FileNotFoundError("The extracted project files are missing.")
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print("Project:", PROJECT)
if os.environ.get("WEEK5_OFFLINE_TEST") != "1":
    stream(sys.executable, "-m", "pip", "install", "-e", PROJECT)

def run(*args):
    stream(sys.executable, "-m", "router_lab", *args, cwd=PROJECT)
''')
md("""## 2 Freeze and check the data

The 468 development tickets provide the training data. The 117 outer held-out tickets never enter training or checkpoint selection. IDs, source hashes, exact split membership, prompt and template are saved. Normalized duplicate ticket texts and unknown labels fail validation.
""")
code('''from router_lab.experiment import load_prepared
DATA = PROJECT / "artifacts/prepared"
if not (DATA / "manifest.json").exists():
    run("prepare", "--output", DATA)
manifest, development, heldout = load_prepared(DATA)
assert (len(development), len(heldout)) == (468, 117)
print("Development:", len(development), "| Outer held-out:", len(heldout))
print(json.dumps(manifest["label_counts"], indent=2))
print("Source SHA-256:", manifest["source_sha256"])
''')
md("""## 3 Run CPU baselines

These are routing baselines fitted only on the development tickets. They are separate from the required unchanged-Qwen comparison. Each prediction and failed ticket is exported.
""")
code('''CPU = PROJECT / "artifacts/baselines"
if not (CPU / "summary.json").exists():
    run("baselines", "--data", DATA, "--output", CPU)
print(json.dumps(json.loads((CPU / "summary.json").read_text()), indent=2))
from IPython.display import display, Image
display(Image(filename=str(CPU / "tfidf_logistic/confusion_matrix.png")))
''')
md("""## 4 Enable the GPU experiment

Change `RUN_GPU_STEPS` to `True` only after selecting a GPU runtime. The following cells install the pinned framework, download the public 1.7B checkpoint, train, and evaluate. No API key is needed. Execution time and model quality have not been measured yet.

The source notebook's destructive directory reset is removed. Each GPU experiment gets its own output directory. Set the flag back to `False` to browse existing reports without starting another run.
""")
code('''RUN_GPU_STEPS = False
from datetime import datetime, timezone
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN = PROJECT / "outputs" / RUN_ID
if RUN_GPU_STEPS:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime before continuing.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Evidence and model directory:", RUN)
else:
    print("GPU installation, training, and model inference are disabled.")
''')
code('''if RUN_GPU_STEPS:
    stream(sys.executable, "scripts/setup_colab.py", cwd=PROJECT)
else:
    print("GPU setup not executed.")
''')
md("""## 5 Audit tokens and create the training configuration

The tokenizer audit includes the full prompt, target label, and end marker. Training refuses a cutoff that truncates examples. This first experiment uses rank 8, alpha 16, learning rate 1e-4, three epochs, batch 1 and accumulation 8, with 4-bit NF4 and FP16 computation.

LLaMA Factory draws a separate 10% inner validation split from the 468 development examples for checkpoint selection. It does not see the 117 outer held-out tickets. Save actual settings and results; these are starter settings rather than selected values from a completed experiment.
""")
code('''AUDIT = RUN / "token-audit.json"
TRAIN_CONFIG = RUN / "train.yaml"
ADAPTER = RUN / "adapter"
MERGED = RUN / "merged"
if RUN_GPU_STEPS:
    run("audit-tokens", "--data", DATA, "--output", AUDIT)
    run("train-config", "--data", DATA, "--audit", AUDIT,
        "--output", TRAIN_CONFIG, "--adapter-output", ADAPTER)
    run("train", "--config", TRAIN_CONFIG)  # validation only
else:
    print("Token audit and GPU training configuration not executed.")
''')
md("""## 6 Train the LoRA adapter

This is the cell that starts real training. Keep its live log and the saved YAML, execution record, loss curve, and adapter. Do not select changes using the outer held-out results. Another attempt should use a new run directory.
""")
code('''if RUN_GPU_STEPS:
    run("train", "--config", TRAIN_CONFIG, "--execute")
else:
    print("Training not executed. No adapter or quality result is claimed.")
''')
code('''if RUN_GPU_STEPS:
    import matplotlib.pyplot as plt
    state = ADAPTER / "trainer_state.json"
    if not state.exists():
        candidates = list(ADAPTER.glob("checkpoint-*/trainer_state.json"))
        state = max(candidates, key=lambda path: int(path.parent.name.split("-")[-1])) if candidates else state
    if not state.exists():
        raise FileNotFoundError("No training state exists; inspect the saved training log.")
    history = json.loads(state.read_text())["log_history"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for field, label in (("loss", "Training loss"), ("eval_loss", "Inner validation loss")):
        rows = [row for row in history if field in row]
        if rows:
            ax.plot([row["step"] for row in rows], [row[field] for row in rows], marker=".", label=label)
    ax.set(xlabel="Training step", ylabel="Loss", title="Actual LoRA training and inner validation loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(RUN / "training_loss.png", dpi=150)
    plt.show()
    print("Training record:", ADAPTER.parent / (ADAPTER.name + "-execution.json"))
else:
    print("No training curve available before a real run.")
''')
md("""## 7 Evaluate the unchanged base and trained adapter

The primary comparison uses identical label-name generation, prompts, stop tokens, FP16 precision, and held-out ticket IDs. Invalid text, generation cutoffs, exceptions, and missing predictions count as incorrect. There is no silent `Support general` fallback or claimed calibrated confidence.

Each subprocess releases its model before the next stage. Model loading is measured separately; per-ticket latency covers tokenization and inference with device synchronization. It does not include browser or network time.
""")
code('''if RUN_GPU_STEPS:
    run("evaluate", "--variant", "base", "--data", DATA,
        "--output", RUN / "base", "--dtype", "float16", "--device", "cuda:0")
    run("evaluate", "--variant", "adapter", "--adapter", ADAPTER, "--data", DATA,
        "--output", RUN / "adapter-eval", "--dtype", "float16", "--device", "cuda:0")
    run("compare", "--base", RUN / "base/report.json", "--tuned", RUN / "adapter-eval/report.json",
        "--output", RUN / "base-vs-adapter")
else:
    print("Base and adapter inference not executed.")
''')
md("""## 8 Merge and verify standalone inference

Merge with a freshly loaded **unquantized** base model, not the 4-bit training model. The helper saves a standalone checkpoint, releases the original objects, and verifies a local-only fresh reload. The next cell starts a separate process for five sanity checks and full held-out inference. Keep failed checks visible.
""")
code('''if RUN_GPU_STEPS:
    run("merge", "--adapter", ADAPTER, "--output", MERGED)
    run("smoke", "--model", MERGED, "--output", RUN / "smoke", "--device", "cuda:0")
    run("evaluate", "--variant", "merged", "--model", MERGED, "--data", DATA,
        "--output", RUN / "merged-eval", "--dtype", "float16", "--device", "cuda:0")
    run("compare", "--base", RUN / "base/report.json", "--tuned", RUN / "merged-eval/report.json",
        "--output", RUN / "matched-comparison")
    adapter_predictions = json.loads((RUN / "adapter-eval/report.json").read_text())["predictions"]
    merged_predictions = json.loads((RUN / "merged-eval/report.json").read_text())["predictions"]
    disagreements = [a["id"] for a, b in zip(adapter_predictions, merged_predictions)
                     if a["id"] != b["id"] or a["prediction"] != b["prediction"]]
    print("Adapter/merged prediction disagreements:", disagreements)
else:
    print("Merge, smoke checks, and standalone evaluation not executed.")
''')
md("""## 9 Preserve the handout comparison separately

The source handout uses a constrained A–G first-token baseline and a label-generating tuned model. Run that additional baseline below. Its delta includes differences in prompt and decoder, so report the matched comparison above as the control for the training effect. An optional `label_loglikelihood` decoder scores all complete labels under a shared prompt; it is not a calibrated probability.
""")
code('''if RUN_GPU_STEPS:
    run("evaluate", "--variant", "base", "--decoding", "letter_choice", "--data", DATA,
        "--output", RUN / "letter-base", "--dtype", "float16", "--device", "cuda:0")
    run("compare", "--base", RUN / "letter-base/report.json", "--tuned", RUN / "merged-eval/report.json",
        "--output", RUN / "handout-comparison", "--allow-protocol-difference")
else:
    print("Handout-style comparison not executed.")
''')
md("""## 10 Inspect and save the actual results

Inspect the paired chart, precision/recall/F1 by queue, confusion matrix, invalid counts, regressions, and actual ticket examples. The report exports raw model outputs and preserves interrupted cases in its denominators. Save the notebook **with its real outputs** and take a screenshot of the successful training and comparison cells.
""")
code('''if RUN_GPU_STEPS:
    result = json.loads((RUN / "matched-comparison/comparison.json").read_text())
    print(json.dumps({key: result[key] for key in ("protocol_matched", "delta", "case_fixes", "case_regressions")}, indent=2))
    display(Image(filename=str(RUN / "matched-comparison/comparison.png")))
    display(Image(filename=str(RUN / "merged-eval/confusion_matrix.png")))
    tuned = json.loads((RUN / "merged-eval/report.json").read_text())
    for row in tuned["metrics"]["per_class"]:
        print(row)
    print("Invalid:", tuned["metrics"]["invalid_outputs"], "Errors:", tuned["metrics"]["error_cases"])
    print("Measured timing and tokens:", json.dumps(tuned["metrics"]["telemetry"], indent=2))
    print("Dollar cost: not measured. Training and serving time are separate from per-ticket inference.")
else:
    print("Only the CPU baseline results above have been measured in this execution.")
''')
md("""## Optional follow-up: stop at a complete routing label

This engineering format repair was chosen **after observing failure of the original generation protocol**: the adapter could emit a category and then continue without an end token. Its results are an **exploratory follow-up**, not an independent estimate selected before seeing the held-out failures.

Set `RUN_LABEL_STOP_REPAIR = True` to apply the same `label_stop` decoder to the base, adapter, and merged checkpoints. It stops actual greedy generation as soon as the entire generated text, after stripping whitespace, exactly equals one allowed label. It records `complete_label` rather than inventing an end token. Other text remains invalid. The original generation cells and failure reports remain above.

The separate follow-up comparison, raw predictions, and five illustrative checks are included by the final evidence download cell. These checks are not additional held-out data.
""")
code('''RUN_LABEL_STOP_REPAIR = False
if RUN_GPU_STEPS and RUN_LABEL_STOP_REPAIR:
    run("evaluate", "--variant", "base", "--decoding", "label_stop", "--data", DATA,
        "--output", RUN / "label-stop-base", "--dtype", "float16", "--device", "cuda:0")
    run("evaluate", "--variant", "adapter", "--adapter", ADAPTER,
        "--decoding", "label_stop", "--data", DATA,
        "--output", RUN / "label-stop-adapter", "--dtype", "float16", "--device", "cuda:0")
    run("evaluate", "--variant", "merged", "--model", MERGED,
        "--decoding", "label_stop", "--data", DATA,
        "--output", RUN / "label-stop-merged", "--dtype", "float16", "--device", "cuda:0")
    run("compare", "--base", RUN / "label-stop-base/report.json",
        "--tuned", RUN / "label-stop-merged/report.json", "--output", RUN / "label-stop-comparison")

    from router_lab.modeling import LocalRouter
    from router_lab.experiment import new_output, write_json
    smoke_out = new_output(RUN / "label-stop-smoke")
    smoke_router = LocalRouter(str(MERGED), device="cuda:0", dtype="float16", decoding="label_stop")
    examples = [
        ("Please create a new employee directory account.", "Active Directory"),
        ("I cannot access the shared network folder; access is denied.", "Fileservice"),
        ("Outlook asks for my password repeatedly and Teams will not sync.", "O365"),
        ("Retire this end-of-life server and remove its monitoring.", "EOL"),
        ("Adobe Acrobat installation fails with error 1603.", "Software"),
    ]
    try:
        smoke_rows = [{"text": text, "expected": label, **smoke_router.predict(text)} for text, label in examples]
    finally:
        del smoke_router
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    smoke = {"kind": "five_example_smoke", "decoding": "label_stop",
             "correct": sum(row["prediction"] == row["expected"] for row in smoke_rows),
             "total": len(smoke_rows), "rows": smoke_rows,
             "note": "Exploratory posthoc format repair. Illustrative sanity checks, separate from the frozen held-out evaluation. Failures remain visible."}
    write_json(smoke_out / "smoke.json", smoke)

    print("Exploratory follow-up: decoder repair chosen after original generation failures.")
    for name in ("base", "adapter", "merged"):
        measured = json.loads((RUN / ("label-stop-" + name) / "report.json").read_text())["metrics"]
        print(name, json.dumps({key: measured[key] for key in ("accuracy", "macro_f1", "invalid_outputs", "error_cases")}, indent=2))
    followup = json.loads((RUN / "label-stop-comparison/comparison.json").read_text())
    print(json.dumps({key: followup[key] for key in ("protocol_matched", "delta", "case_fixes", "case_regressions")}, indent=2))
    print("Illustrative label-stop smoke checks:", smoke["correct"], "/", smoke["total"])
    from IPython.display import display, Image
    display(Image(filename=str(RUN / "label-stop-comparison/comparison.png")))
else:
    print("Optional label-stop follow-up disabled. Original generation results are preserved.")
''')
md("""## 11 Download evidence and preserve the adapter

The evidence archive below contains reports, predictions, configuration, and logs. It excludes multi-gigabyte merged weights and the adapter. Download the adapter separately; preserve the model folder in private Drive storage if you need it after the runtime ends. Saving this notebook alone does not save model weights.

For submission, use the usual Word source → Google Doc in Drive, with an actual successful-run screenshot and measured results. The custom option permits a GitHub repository with all project assets and a Loom. The handout gives **November 8, 2026, 11:59 PM Pacific** for Builder consideration.
""")
code('''if RUN_GPU_STEPS:
    evidence_zip = PROJECT / (RUN_ID + "-evidence.zip")
    with zipfile.ZipFile(evidence_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RUN.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".yaml", ".log", ".html", ".png"} and not path.is_relative_to(MERGED) and not path.is_relative_to(ADAPTER):
                archive.write(path, path.relative_to(RUN))
    import shutil
    adapter_zip = Path(shutil.make_archive(str(PROJECT / (RUN_ID + "-adapter")), "zip", ADAPTER))
    print("Evidence:", evidence_zip, "Adapter:", adapter_zip)
    if Path("/content").is_dir():
        from google.colab import files
        files.download(str(evidence_zip))
        files.download(str(adapter_zip))
else:
    print("No trained-model archive exists before GPU execution.")
''')

notebook = nb.v4.new_notebook(cells=cells)
notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"},
                     "colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU"}
nb.validate(notebook)
target = ROOT / "notebooks/Week5_Support_Ticket_Router.ipynb"
nb.write(notebook, target)
print(target)
