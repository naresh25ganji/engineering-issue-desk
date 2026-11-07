"""Small explicit commands used by both the local project and Colab notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MODEL_ID, SYSTEM_PROMPT, TEMPLATE
from .data import prepare_dataset
from .experiment import compare_reports, new_output, read_json, run_baselines, run_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Validate and export frozen data without model calls.")
    p.add_argument("--csv", default="data/support_tickets.csv")
    p.add_argument("--output", default="artifacts/prepared")
    p = commands.add_parser("baselines", help="Fit majority and TF-IDF/logistic CPU baselines.")
    p.add_argument("--data", default="artifacts/prepared")
    p.add_argument("--output", default="artifacts/baselines")
    p = commands.add_parser("audit-tokens", help="Check full example lengths with the pinned tokenizer only.")
    p.add_argument("--data", default="artifacts/prepared")
    p.add_argument("--output", default="artifacts/token-audit.json")
    p = commands.add_parser("train-config", help="Create a reproducible, audited FP16/T4 LoRA configuration.")
    p.add_argument("--data", default="artifacts/prepared")
    p.add_argument("--audit", default="artifacts/token-audit.json")
    p.add_argument("--output", default="configs/train.yaml")
    p.add_argument("--adapter-output", default="outputs/run1/adapter")
    p.add_argument("--cutoff", type=int, default=512)
    p.add_argument("--epochs", type=float, default=3.0)
    p = commands.add_parser("train", help="Validate training configuration; --execute starts GPU training.")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--execute", action="store_true")
    p = commands.add_parser("evaluate", help="Run actual local checkpoint inference.")
    p.add_argument("--data", default="artifacts/prepared")
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=("base", "adapter", "merged"), required=True)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--adapter")
    p.add_argument("--decoding", choices=("generate", "label_stop", "label_loglikelihood", "letter_choice"), default="generate")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p = commands.add_parser("merge", help="Merge unquantized base and adapter, then verify a fresh reload.")
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("compare", help="Compare saved model results on identical held-out cases.")
    p.add_argument("--base", required=True)
    p.add_argument("--tuned", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-protocol-difference", action="store_true")
    p = commands.add_parser("smoke", help="Five fresh-process sanity checks on a merged checkpoint.")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare_dataset(args.csv, args.output, system_prompt=SYSTEM_PROMPT, template=TEMPLATE)
        value = {key: value[key] for key in ("source_sha256", "train_count", "heldout_count", "label_counts", "heldout_policy")}
    elif args.command == "baselines":
        value = run_baselines(args.data, args.output)
    elif args.command == "audit-tokens":
        from .training import audit_tokens
        value = audit_tokens(args.data, args.output)
        value = {key: value[key] for key in ("max_development_tokens", "max_heldout_tokens", "model_revision")}
    elif args.command == "train-config":
        from .training import write_training_config
        value = write_training_config(args.data, args.output, args.adapter_output, audit_path=args.audit, cutoff_len=args.cutoff, epochs=args.epochs)
    elif args.command == "train":
        from .training import train
        value = train(args.config, execute=args.execute)
    elif args.command == "evaluate":
        value = run_model(args.data, args.output, variant=args.variant, model_path=args.model, adapter_path=args.adapter,
                          decoding=args.decoding, limit=args.limit, device=args.device, dtype=args.dtype)
        value = {"variant": value["variant"], "scope": value["manifest"]["scope"], "accuracy": value["metrics"]["accuracy"], "macro_f1": value["metrics"]["macro_f1"], "invalid_outputs": value["metrics"]["invalid_outputs"], "errors": value["metrics"]["error_cases"]}
    elif args.command == "merge":
        from .training import merge
        value = merge(args.adapter, args.output)
    elif args.command == "compare":
        value = compare_reports(args.base, args.tuned, args.output, allow_protocol_difference=args.allow_protocol_difference)
        value = {key: value[key] for key in ("protocol_matched", "interpretation", "delta", "case_fixes", "case_regressions")}
    else:
        from .modeling import LocalRouter
        proof = read_json(Path(args.model) / "router_model.json")
        if not proof.get("fresh_reload_verified"):
            raise ValueError("Run merge verification first.")
        out = new_output(args.output)
        router = LocalRouter(args.model, device=args.device, dtype="float16", decoding="generate")
        examples = [
            ("Please create a new employee directory account.", "Active Directory"),
            ("I cannot access the shared network folder; access is denied.", "Fileservice"),
            ("Outlook asks for my password repeatedly and Teams will not sync.", "O365"),
            ("Retire this end-of-life server and remove its monitoring.", "EOL"),
            ("Adobe Acrobat installation fails with error 1603.", "Software"),
        ]
        rows = [{"text": text, "expected": label, **router.predict(text)} for text, label in examples]
        value = {"kind": "five_example_smoke", "correct": sum(row["prediction"] == row["expected"] for row in rows), "total": len(rows), "rows": rows,
                 "note": "Illustrative sanity checks, separate from the frozen held-out evaluation. Failures remain visible."}
        write_json(out / "smoke.json", value)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
