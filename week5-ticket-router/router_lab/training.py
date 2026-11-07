"""Explicit GPU training, token-budget checks, and standalone model provenance."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .config import FACTORY_REVISION, MODEL_ID, MODEL_REVISION, TEMPLATE, format_prompt, messages_for
from .experiment import load_prepared, read_json, sha_file, write_json


def audit_tokens(data_dir, output_path, *, tokenizer=None):
    """Download only the pinned tokenizer and measure complete training examples."""
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    _, train, heldout = load_prepared(data_dir)
    lengths = {}
    for name, rows in (("development", train), ("heldout", heldout)):
        lengths[name] = [{"id": row.id, "tokens": len(tokenizer.encode(format_prompt(messages_for(row.text)) + row.label + "<|im_end|>\n", add_special_tokens=False))} for row in rows]
    report = {"model": MODEL_ID, "model_revision": MODEL_REVISION,
              "dataset_manifest_sha256": sha_file(Path(data_dir) / "manifest.json"),
              "max_development_tokens": max(row["tokens"] for row in lengths["development"]),
              "max_heldout_tokens": max(row["tokens"] for row in lengths["heldout"]),
              "lengths": lengths, "note": "Complete prompt, target label, and chat terminator. Held-out lengths are diagnostics and do not enter training."}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    return report


def write_training_config(data_dir, config_path, output_dir, *, audit_path, cutoff_len=512, epochs=3.0):
    import yaml
    load_prepared(data_dir)
    audit = read_json(audit_path)
    if audit["dataset_manifest_sha256"] != sha_file(Path(data_dir) / "manifest.json") or audit["model_revision"] != MODEL_REVISION:
        raise ValueError("Token audit belongs to another dataset or tokenizer revision.")
    if cutoff_len < audit["max_development_tokens"]:
        raise ValueError(f"cutoff_len={cutoff_len} would truncate training examples; need at least {audit['max_development_tokens']} tokens.")
    if not 1 <= epochs <= 10 or not 128 <= cutoff_len <= 4096:
        raise ValueError("Use 1–10 epochs and a token cutoff between 128 and 4096.")
    config = {
        "model_name_or_path": MODEL_ID, "model_revision": MODEL_REVISION,
        "trust_remote_code": False,
        "stage": "sft", "do_train": True, "do_eval": True,
        "finetuning_type": "lora", "lora_rank": 8, "lora_alpha": 16,
        "lora_target": "all", "lora_dropout": 0.0,
        "quantization_bit": 4, "quantization_method": "bnb",
        "quantization_type": "nf4", "double_quantization": True,
        "dataset": "support_tickets", "dataset_dir": str(Path(data_dir).resolve()),
        "template": TEMPLATE, "cutoff_len": cutoff_len,
        "val_size": 0.1, "preprocessing_num_workers": 2,
        "dataloader_num_workers": 0, "packing": False,
        "output_dir": str(Path(output_dir).resolve()),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
        "per_device_eval_batch_size": 1, "learning_rate": 1e-4,
        "num_train_epochs": float(epochs), "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1, "gradient_checkpointing": True,
        "fp16": True, "bf16": False, "fp16_full_eval": True,
        "flash_attn": "sdpa", "seed": 42, "data_seed": 42,
        "logging_steps": 5, "save_strategy": "epoch", "eval_strategy": "epoch",
        "save_total_limit": 2, "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss", "greater_is_better": False,
        "plot_loss": True, "report_to": "none",
    }
    destination = Path(config_path)
    if destination.exists():
        raise FileExistsError(f"Use a new training configuration: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(destination.with_suffix(".manifest.json"), {
        "factory_revision": FACTORY_REVISION, "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset_manifest_sha256": sha_file(Path(data_dir) / "manifest.json"),
        "token_audit_sha256": sha_file(audit_path), "config_sha256": sha_file(destination),
        "development_examples": len(audit["lengths"]["development"]),
        "inner_validation_fraction": 0.1,
        "heldout_policy": "Outer held-out tickets are never used for training or checkpoint selection",
    })
    return config


def train(config_path, *, execute=False):
    """Run the pinned LLaMA Factory CLI only on an explicitly requested CUDA run."""
    import yaml
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    manifest = read_json(config_path.with_suffix(".manifest.json"))
    if sha_file(config_path) != manifest["config_sha256"]:
        raise ValueError("Training YAML changed; regenerate its audited manifest.")
    prepared, _, _ = load_prepared(config["dataset_dir"])
    if sha_file(Path(config["dataset_dir"]) / "manifest.json") != manifest["dataset_manifest_sha256"]:
        raise ValueError("Training data changed after configuration creation.")
    if not execute:
        return {"status": "validated_only", "configuration": config, "manifest": manifest, "model_calls": 0}
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this 4-bit LoRA training configuration. Run the notebook in a GPU runtime.")
    import llamafactory
    factory_root = Path(llamafactory.__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=factory_root, text=True).strip()
    if revision != FACTORY_REVISION:
        raise ValueError("Installed LLaMA Factory does not match the pinned training commit.")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new adapter directory; this command does not overwrite or silently resume training.")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.parent / (output.name + "-execution.json")
    if metadata_path.exists():
        raise FileExistsError("This training execution already has a saved record; use a new run directory.")
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "llamafactory"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "config": config, "config_manifest": manifest, "versions": versions,
              "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "development_count": prepared["train_count"], "outer_heldout_count": prepared["heldout_count"]}
    write_json(metadata_path, record)
    command = [sys.executable, "-m", "llamafactory.cli", "train", str(config_path)]
    with (output.parent / (output.name + "-training.log")).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            record["status"] = "interrupted"
            write_json(metadata_path, record)
            raise
    record.update({"status": "completed" if code == 0 else "failed", "return_code": code,
                   "finished_at": datetime.now(timezone.utc).isoformat()})
    write_json(metadata_path, record)
    if code:
        raise RuntimeError(f"LLaMA Factory training failed with exit code {code}; inspect the saved log.")
    if not (output / "adapter_config.json").exists():
        raise RuntimeError("Training exited without an adapter_config.json; inspect the saved log.")
    record["adapter_files"] = {path.name: sha_file(path) for path in output.iterdir()
                               if path.is_file() and (path.suffix == ".safetensors" or path.name == "adapter_config.json")}
    if not any(name.endswith(".safetensors") for name in record["adapter_files"]):
        raise RuntimeError("No trained adapter safetensors were produced.")
    write_json(metadata_path, record)
    return record


def validate_adapter(adapter_path, dataset_sha=None):
    adapter_path = Path(adapter_path).resolve()
    config = read_json(adapter_path / "adapter_config.json")
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError("Adapter declares a different base checkpoint.")
    if config.get("revision") not in (None, MODEL_REVISION):
        raise ValueError("Adapter revision differs from the pinned base checkpoint.")
    execution_path = adapter_path.parent / (adapter_path.name + "-execution.json")
    execution = read_json(execution_path)
    if execution.get("status") != "completed" or execution["config"].get("model_revision") != MODEL_REVISION:
        raise ValueError("Adapter requires a completed training record for the pinned model revision.")
    if dataset_sha is not None and execution["config_manifest"]["dataset_manifest_sha256"] != dataset_sha:
        raise ValueError("Adapter training used a different prepared dataset.")
    files = execution.get("adapter_files", {})
    if not files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Adapter execution record has no saved weight fingerprints.")
    actual_files = {path.name for path in adapter_path.iterdir() if path.is_file()
                    and (path.suffix == ".safetensors" or path.name == "adapter_config.json")}
    if set(files) != actual_files:
        raise ValueError("Adapter has missing or unexpected checkpoint files.")
    for name, expected_hash in files.items():
        if Path(name).name != name or sha_file(adapter_path / name) != expected_hash:
            raise ValueError("Adapter files changed after training.")
    return execution


def merge(adapter_path, output_dir):
    from .modeling import merge_adapter
    adapter_path = Path(adapter_path).resolve()
    execution = validate_adapter(adapter_path)
    execution_path = adapter_path.parent / (adapter_path.name + "-execution.json")
    receipt = merge_adapter(MODEL_ID, adapter_path, output_dir, revision=MODEL_REVISION, device="cpu", dtype="float16")
    proof = {"base_model": MODEL_ID, "base_revision": MODEL_REVISION,
             "adapter_config_sha256": sha_file(adapter_path / "adapter_config.json"),
             "training_execution_sha256": sha_file(execution_path),
             "dataset_manifest_sha256": execution["config_manifest"]["dataset_manifest_sha256"],
             "fresh_reload_verified": True, "merge_receipt": receipt,
             "model_files": {path.name: sha_file(path) for path in Path(output_dir).iterdir()
                             if path.is_file() and path.suffix in (".json", ".safetensors")},
             "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(Path(output_dir) / "router_model.json", proof)
    return proof
