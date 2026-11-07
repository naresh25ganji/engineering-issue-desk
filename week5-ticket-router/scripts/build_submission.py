"""Build a concise submission from completed Colab evidence, without inventing results.

Run with the bundled Codex Python runtime after the artifact-operation marker.
Sanitize the resulting DOCX title, render it, and inspect every page before import.
This script intentionally refuses incomplete or unmatched evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT


NOTEBOOK_URL = "https://colab.research.google.com/drive/1hp3rD-MzprA8XunuC4u74Hb7Tr2M_zvB"
COURSE_URL = "https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value):
    return f"{value:.2%}"


def add_link(paragraph, label, url):
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True))
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1557A0")
    properties.append(color)
    run.append(properties)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


def set_cell_properties(cell, *, header=False):
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    properties = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        border = OxmlElement("w:" + side)
        for name, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            border.set(qn("w:" + name), value)
        borders.append(border)
    properties.append(borders)
    margins = OxmlElement("w:tcMar")
    for side in ("top", "left", "bottom", "right"):
        margin = OxmlElement("w:" + side)
        margin.set(qn("w:w"), "85")
        margin.set(qn("w:type"), "dxa")
        margins.append(margin)
    properties.append(margins)
    if header:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "E8EEF4")
        properties.append(shading)
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        for run in paragraph.runs:
            run.font.size = Pt(10)
            run.bold = header
            run.font.color.rgb = RGBColor(0, 0, 0)


def evidence(directory: Path, *, decoder="generate"):
    original_reports = {
        name: load_json(directory / folder / "report.json")
        for name, folder in (("base", "base"), ("adapter", "adapter-eval"), ("merged", "merged-eval"))
    }
    if decoder == "label_stop":
        reports = {
            name: load_json(directory / folder / "report.json")
            for name, folder in (("base", "label-stop-base"), ("adapter", "label-stop-adapter"), ("merged", "label-stop-merged"))
        }
        matched = load_json(directory / "label-stop-comparison/comparison.json")
        smoke = load_json(directory / "label-stop-smoke/smoke.json")
    elif decoder == "generate":
        reports = original_reports
        matched = load_json(directory / "matched-comparison/comparison.json")
        smoke = load_json(directory / "smoke/smoke.json")
    else:
        raise ValueError("Submission decoder must be generate or label_stop.")
    handout = load_json(directory / "handout-comparison/comparison.json")
    execution = load_json(directory / "adapter-execution.json")
    curve = directory / "training_loss.png"
    if not curve.is_file():
        raise ValueError("The actual training loss image is missing.")
    if execution.get("status") != "completed" or not execution.get("adapter_files"):
        raise ValueError("A completed training execution with saved adapter weights is required.")
    if matched.get("protocol_matched") is not True or handout.get("protocol_matched") is not False:
        raise ValueError("Require a matched control and an explicitly unmatched handout comparison.")
    reference = reports["base"]["manifest"]
    for variant, report in reports.items():
        if report.get("kind") != "model_evaluation" or report.get("variant") != variant:
            raise ValueError(f"Unexpected {variant} evaluation report.")
        if report["metrics"]["scheduled_cases"] != 117 or report["manifest"]["scope"] != "full held-out evaluation":
            raise ValueError("Submission requires the full 117-ticket evaluation.")
        if report["manifest"]["decoding"] != decoder:
            raise ValueError("Selected report uses a different decoder.")
        for key in ("dataset_manifest_sha256", "heldout_ids", "base_model", "base_revision"):
            if report["manifest"][key] != reference[key]:
                raise ValueError(f"Evaluation evidence differs on {key}.")
    if matched["metrics"]["baseline"] != reports["base"]["metrics"] or matched["metrics"]["tuned"] != reports["merged"]["metrics"]:
        raise ValueError("Matched comparison does not describe the supplied base and merged reports.")
    if handout["metrics"]["tuned"] != original_reports["merged"]["metrics"]:
        raise ValueError("Handout comparison must describe the original EOS-terminated merged report.")
    if decoder == "label_stop":
        for variant, original in original_reports.items():
            if original.get("variant") != variant or original["manifest"]["decoding"] != "generate":
                raise ValueError("Original EOS-terminated evidence is required alongside the repair.")
            if original["metrics"]["scheduled_cases"] != 117:
                raise ValueError("Original failure accounting must cover all 117 cases.")
            for key in ("dataset_manifest_sha256", "heldout_ids", "base_model", "base_revision"):
                if original["manifest"][key] != reference[key]:
                    raise ValueError(f"Original and repaired runs differ on {key}.")
    if smoke.get("kind") != "five_example_smoke" or smoke.get("total") != 5:
        raise ValueError("The five-example standalone smoke report is missing.")
    return reports, matched, handout, execution, smoke, curve, original_reports


def build(directory: Path, destination: Path, *, screenshot=None, image_width=6.4, evidence_url=None, decoder="generate"):
    reports, matched, handout, execution, smoke, curve, original_reports = evidence(directory, decoder=decoder)
    image = Path(screenshot).resolve() if screenshot else curve
    if not image.is_file():
        raise FileNotFoundError(image)
    config = execution["config"]
    development = execution["development_count"]
    inner = math.ceil(development * config["val_size"])
    if (development - inner, inner, execution["outer_heldout_count"]) != (421, 47, 117):
        raise ValueError("The observed split differs from the intended 421/47/117 run.")
    base, adapter, merged = (reports[name]["metrics"] for name in ("base", "adapter", "merged"))
    disagreements = sum(
        left["prediction"] != right["prediction"]
        for left, right in zip(adapter["case_results"], merged["case_results"])
    )
    finished = datetime.fromisoformat(execution["finished_at"])

    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.65)
    section.left_margin = section.right_margin = Inches(0.7)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Arial", Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.05
    for name in ("Title", "Subtitle", "Heading 1", "Heading 2"):
        style = document.styles[name]
        style.font.name = "Arial"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.underline = False
    document.styles["Title"].font.size = Pt(21)
    document.styles["Title"].paragraph_format.space_after = Pt(4)
    document.styles["Subtitle"].font.size = Pt(10)
    document.styles["Subtitle"].paragraph_format.space_after = Pt(8)
    document.styles["Heading 1"].font.size = Pt(12)
    document.styles["Heading 1"].paragraph_format.space_before = Pt(8)
    document.styles["Heading 1"].paragraph_format.space_after = Pt(4)
    document.add_paragraph("Week 5 Support Ticket Router Submission", "Title")
    document.add_paragraph(f"Naresh Ganji  |  Run completed {finished:%B %d %Y} UTC", "Subtitle")
    delta = matched["delta"]["accuracy"] * 100
    change = "improved" if delta > 0 else "decreased" if delta < 0 else "was unchanged"
    comparison = f"{abs(delta):.2f} percentage points" if delta else ""
    opening = f"I fine-tuned {config['model_name_or_path']} with LoRA and verified a standalone merged checkpoint. "
    if decoder == "label_stop":
        opening += (
            f"An exploratory follow-up using complete-label stopping scored {percent(merged['accuracy'])} merged accuracy "
            f"on 117 previously examined held-out tickets, versus {percent(base['accuracy'])} for the unchanged base "
            "with the same prompt and decoder."
        )
    else:
        opening += (
            f"On 117 held-out support tickets, merged accuracy was {percent(merged['accuracy'])}; "
            f"it {change}{' by ' + comparison if comparison else ''} against the unchanged base using the same prompt and decoder."
        )
    document.add_paragraph(opening)
    if decoder == "label_stop":
        initial = original_reports["merged"]["metrics"]
        document.add_paragraph(
            f"The initial EOS-terminated run produced {initial['invalid_outputs']}/{initial['scheduled_cases']} invalid tuned outputs; "
            "label stopping was added afterward as an engineering repair, so these follow-up results are exploratory."
        )
    document.add_paragraph("Training", "Heading 1")
    document.add_paragraph(
        f"The 585 course tickets were split into 421 training, 47 inner-validation and 117 outer held-out examples. "
        f"The held-out tickets were excluded from training and checkpoint selection. "
        f"The run used {execution['gpu']}, 4-bit NF4, FP16, LoRA rank {config['lora_rank']} and alpha {config['lora_alpha']}, "
        f"learning rate {config['learning_rate']:g}, {config['num_train_epochs']:g} epochs, "
        f"batch {config['per_device_train_batch_size']} with accumulation {config['gradient_accumulation_steps']}, "
        f"and a {config['cutoff_len']}-token limit."
    )
    document.add_paragraph("Exploratory results with complete label stopping" if decoder == "label_stop" else "Held out results", "Heading 1")
    table = document.add_table(rows=1, cols=5)
    table.autofit = False
    widths = (2.7, 1.0, 1.0, 0.7, 0.7)
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    first_header = "Matched prompt and label stopping" if decoder == "label_stop" else "Same prompt and decoder"
    for cell, width, text in zip(table.rows[0].cells, widths, (first_header, "Accuracy", "Macro F1", "Invalid", "Errors")):
        cell.width = Inches(width)
        cell.text = text
        set_cell_properties(cell, header=True)
    repeat = OxmlElement("w:tblHeader")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for label, scores in (("Unchanged base", base), ("Base with trained adapter", adapter), ("Standalone merged model", merged)):
        cells = table.add_row().cells
        for index, (cell, width, text) in enumerate(zip(cells, widths, (label, percent(scores["accuracy"]), f"{scores['macro_f1']:.4f}", str(scores["invalid_outputs"]), str(scores["error_cases"])))):
            cell.width = Inches(width)
            cell.text = text
            set_cell_properties(cell)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(
        f"The original handout-style comparison scored {percent(handout['metrics']['baseline']['accuracy'])} accuracy "
        f"and {handout['metrics']['baseline']['macro_f1']:.4f} macro F1 for the A–G base, versus "
        f"{percent(handout['metrics']['tuned']['accuracy'])} and {handout['metrics']['tuned']['macro_f1']:.4f} "
        "for the EOS-terminated merged model. Those decoders differ; that comparison combines training and protocol effects."
    )
    document.add_paragraph("Verification and limits", "Heading 1")
    limits = (
        f"The merged model passed a fresh local-only reload and scored {smoke['correct']}/{smoke['total']} "
        f"on illustrative smoke checks. Adapter and merged predictions differed on {disagreements}/117 tickets. "
        "Invalid outputs, errors and missing predictions count as incorrect. "
        "The source labels and semantic near-duplicates have not received independent human review. "
        "Dollar cost was not measured."
    )
    if decoder == "label_stop":
        limits += " An independent test set is needed to confirm the follow-up scores."
    document.add_paragraph(limits)
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    shape = paragraph.add_run().add_picture(str(image), width=Inches(image_width))
    shape._inline.docPr.set("descr", "Actual Colab notebook output" if screenshot else "Actual training and inner-validation loss curve")
    caption = document.add_paragraph("Actual Colab run" if screenshot else "Actual training and inner validation loss")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_after = Pt(4)
    for run in caption.runs:
        run.font.size = Pt(9)
    links = document.add_paragraph()
    add_link(links, "Executed notebook", NOTEBOOK_URL)
    links.add_run("  |  ")
    add_link(links, "Course source", COURSE_URL)
    if evidence_url:
        links.add_run("  |  ")
        add_link(links, "Run evidence", evidence_url)
    document.core_properties.title = "Week 5 Support Ticket Router Submission"
    document.core_properties.author = "Naresh Ganji"
    document.core_properties.subject = "Actual LoRA training merge and routing evaluation"
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)
    return {"path": str(destination.resolve()), "image": str(image), "decoder": decoder,
            "accuracy": merged["accuracy"], "macro_f1": merged["macro_f1"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--screenshot", type=Path, help="Optional actual successful notebook screenshot; otherwise use the saved loss curve.")
    parser.add_argument("--image-width", type=float, default=6.4)
    parser.add_argument("--evidence-url")
    parser.add_argument("--decoder", choices=("generate", "label_stop"), default="generate")
    options = parser.parse_args()
    if not 4 <= options.image_width <= 7.1:
        parser.error("--image-width must be between 4 and 7.1 inches")
    print(json.dumps(build(options.evidence.resolve(), options.output.resolve(), screenshot=options.screenshot,
                           image_width=options.image_width, evidence_url=options.evidence_url, decoder=options.decoder), indent=2))
