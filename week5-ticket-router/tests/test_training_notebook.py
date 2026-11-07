"""Integration checks for truncation guards, GPU boundaries and the portable lab."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest
import yaml

from router_lab.config import MODEL_REVISION, SYSTEM_PROMPT, TEMPLATE
from router_lab.data import prepare_dataset
from router_lab.training import audit_tokens, train, write_training_config

ROOT = Path(__file__).resolve().parents[1]


class LengthTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(len(text.split())))


def prepared(tmp_path):
    destination = tmp_path / "prepared"
    prepare_dataset(ROOT / "data/support_tickets.csv", destination,
                    system_prompt=SYSTEM_PROMPT, template=TEMPLATE)
    audit_path = tmp_path / "tokens.json"
    audit_tokens(destination, audit_path, tokenizer=LengthTokenizer())
    return destination, audit_path


def test_training_uses_inner_split_and_never_outer_heldout(tmp_path):
    data, audit = prepared(tmp_path)
    path = tmp_path / "train.yaml"
    config = write_training_config(data, path, tmp_path / "adapter", audit_path=audit)
    result = train(path)
    assert result["status"] == "validated_only" and result["model_calls"] == 0
    registration = json.loads((data / "dataset_info.json").read_text())
    assert registration[config["dataset"]]["file_name"] == "train_sharegpt.json"
    assert config["val_size"] == .1
    assert config["model_revision"] == MODEL_REVISION
    assert config["fp16"] is True and config["bf16"] is False
    assert not (tmp_path / "adapter").exists()


def test_cutoff_guard_checks_full_training_prompt_and_target(tmp_path):
    data, audit = prepared(tmp_path)
    content = json.loads(audit.read_text())
    content["max_development_tokens"] = 600
    audit.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="truncate"):
        write_training_config(data, tmp_path / "train.yaml", tmp_path / "adapter", audit_path=audit)


def test_tampered_training_yaml_is_rejected_before_execution(tmp_path):
    data, audit = prepared(tmp_path)
    path = tmp_path / "train.yaml"
    write_training_config(data, path, tmp_path / "adapter", audit_path=audit)
    content = yaml.safe_load(path.read_text())
    content["dataset"] = "heldout"
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="changed"):
        train(path, execute=True)


def test_cuda_guard_does_not_attempt_training_on_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    data, audit = prepared(tmp_path)
    path = tmp_path / "train.yaml"
    write_training_config(data, path, tmp_path / "adapter", audit_path=audit)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU required"):
        train(path, execute=True)
    assert not (tmp_path / "adapter-execution.json").exists()


def test_notebook_cells_compile_and_no_gpu_work_enabled_by_default():
    notebook = nbformat.read(ROOT / "notebooks/Week5_Support_Ticket_Router.ipynb", as_version=4)
    nbformat.validate(notebook)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for index, source in enumerate(sources):
        compile(source, f"cell-{index}", "exec")
    assert sum("RUN_GPU_STEPS = False" in source for source in sources) == 1
    assert sum("RUN_LABEL_STOP_REPAIR = False" in source for source in sources) == 1
    for source in sources:
        if 'run("evaluate"' in source:
            assert '"--dtype", "float16"' in source
            assert "if RUN_GPU_STEPS" in source
        if '"--execute"' in source:
            assert "if RUN_GPU_STEPS:" in source


def test_label_stop_followup_is_optional_separate_and_downloaded(capsys):
    notebook = nbformat.read(ROOT / "notebooks/Week5_Support_Ticket_Router.ipynb", as_version=4)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    repair = next(source for source in sources if "RUN_LABEL_STOP_REPAIR = False" in source)
    tree = ast.parse(repair)
    guard = tree.body[1]
    assert isinstance(guard, ast.If) and isinstance(guard.test, ast.BoolOp) and isinstance(guard.test.op, ast.And)
    assert [node.id for node in guard.test.values] == ["RUN_GPU_STEPS", "RUN_LABEL_STOP_REPAIR"]
    evaluations = [node for node in ast.walk(guard) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "run"
                   and isinstance(node.args[0], ast.Constant) and node.args[0].value == "evaluate"]
    assert len(evaluations) == 3
    for call in evaluations:
        constants = [node.value for node in call.args if isinstance(node, ast.Constant)]
        assert constants[constants.index("--decoding") + 1] == "label_stop"
        assert constants[constants.index("--dtype") + 1] == "float16"
        assert constants[constants.index("--device") + 1] == "cuda:0"
    for path in ("label-stop-base", "label-stop-adapter", "label-stop-merged", "label-stop-comparison", "label-stop-smoke"):
        assert path in repair
    assert '"kind": "five_example_smoke", "decoding": "label_stop"' in repair
    examples = next(node.value for node in guard.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "examples" for target in node.targets))
    assert len(examples.elts) == 5
    assert sources.index(repair) < next(index for index, source in enumerate(sources) if "evidence_zip =" in source)
    assert 'RUN / "base"' not in repair and 'RUN / "merged-eval"' not in repair
    # Its default must be safe even when the main GPU experiment is enabled.
    exec(compile(tree, "optional-label-stop-cell", "exec"), {"RUN_GPU_STEPS": True})
    assert "follow-up disabled" in capsys.readouterr().out


def test_notebook_streams_bootstrap_cli_and_setup_output(tmp_path, capsys):
    notebook = nbformat.read(ROOT / "notebooks/Week5_Support_Ticket_Router.ipynb", as_version=4)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    bootstrap = ast.parse(sources[0])
    helper = next(node for node in bootstrap.body if isinstance(node, ast.FunctionDef) and node.name == "stream")
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "notebook-stream", "exec"), namespace)
    result = namespace["stream"](sys.executable, "-c", "import sys; print('child stdout', flush=True); print('child stderr', file=sys.stderr, flush=True)", cwd=tmp_path)
    assert result.returncode == 0
    output = capsys.readouterr().out
    assert "child stdout" in output and "child stderr" in output
    with pytest.raises(subprocess.CalledProcessError) as failure:
        namespace["stream"](sys.executable, "-c", "raise SystemExit(7)", cwd=tmp_path)
    assert failure.value.returncode == 7
    stream_calls = [node for node in ast.walk(bootstrap)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "stream"]
    assert len(stream_calls) == 2  # bootstrap pip install and the router_lab run() wrapper
    assert 'stream(sys.executable, "-m", "pip"' in sources[0]
    assert 'stream(sys.executable, "-m", "router_lab"' in sources[0]
    setup = next(source for source in sources if '"scripts/setup_colab.py"' in source)
    assert 'stream(sys.executable, "scripts/setup_colab.py", cwd=PROJECT)' in setup
    assert not any("subprocess.run(" in source for source in sources)


def test_packaged_project_contains_no_environment_or_model_weights(tmp_path):
    import importlib.util
    import zipfile
    spec = importlib.util.spec_from_file_location("package_project", ROOT / "scripts/package_project.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    receipt = package.build(tmp_path / "project.zip")
    with zipfile.ZipFile(receipt["path"]) as archive:
        names = archive.namelist()
        assert "week5-ticket-router/notebooks/Week5_Support_Ticket_Router.ipynb" in names
        assert "week5-ticket-router/data/support_tickets.csv" in names
        assert not any("/.venv/" in name or "/.cache/" in name or name.endswith(".safetensors") or "/.env" in name for name in names)
        manifest = json.loads(archive.read("week5-ticket-router/PACKAGE-MANIFEST.json"))
        import hashlib
        for row in manifest["files"]:
            assert hashlib.sha256(archive.read("week5-ticket-router/" + row["path"])).hexdigest() == row["sha256"]
