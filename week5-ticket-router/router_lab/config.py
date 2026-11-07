"""One label vocabulary and prompt shared by training and every model evaluation."""
from __future__ import annotations

import hashlib
import json


LABELS = (
    "Support general", "Fileservice", "O365", "EOL", "Software",
    "Active Directory", "Computer-Services",
)
MODEL_ID = "Qwen/Qwen3-1.7B-Base"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
FACTORY_REVISION = "702005ce5237ae10f3c572c3f7741ecfd1422fc9"
TEMPLATE = "qwen3_nothink"
EOS_TEXT = "<|im_end|>"
PROMPT_VERSION = "ticket-label-v1"
SYSTEM_PROMPT = (
    "You are an IT helpdesk ticket routing assistant. "
    "Given a support ticket, respond with exactly one of the following categories: "
    + ", ".join(LABELS) + ". Return only the exact category name, with no explanation."
)


def messages_for(text: str) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket text must be a nonempty string.")
    if "<|im_start|>" in text or EOS_TEXT in text:
        raise ValueError("Ticket text contains reserved chat-template delimiters.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Support Ticket: " + text},
    ]


def format_prompt(messages: list[dict[str, str]]) -> str:
    """Match the pinned LLaMA Factory qwen3_nothink template exactly."""
    if [item.get("role") for item in messages] != ["system", "user"]:
        raise ValueError("Expected one system message and one user message.")
    return "".join(
        "<|im_start|>" + item["role"] + "\n" + item["content"] + "<|im_end|>\n"
        for item in messages
    ) + "<|im_start|>assistant\n"


def prompt_metadata() -> dict:
    value = {
        "system_prompt": SYSTEM_PROMPT, "user_prefix": "Support Ticket: ",
        "template": TEMPLATE, "labels": list(LABELS), "version": PROMPT_VERSION,
        "assistant_prefix": "<|im_start|>assistant\n", "eos": EOS_TEXT,
    }
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value

