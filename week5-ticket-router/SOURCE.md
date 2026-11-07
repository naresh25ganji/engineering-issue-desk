# Sources and attribution

This lab is an adaptation of The Gen Academy's Week 5 support-ticket routing exercise. The source handout is `Week 5 Project Handout (Oct 2026).docx` supplied by the participant. It specifies Qwen3 1.7B Base, LLaMA-Factory LoRA training, adapter merging, smoke tests, and a baseline comparison.

## Course assets

- Repository: [The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router).
- Reviewed source commit: [`05f7ea9643495741bda4e272bd1977105c3ad529`](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router/tree/05f7ea9643495741bda4e272bd1977105c3ad529).
- Original notebook: [Finetune_Support_Ticket_Classifier_Qwen3.ipynb](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router/blob/05f7ea9643495741bda4e272bd1977105c3ad529/Finetune_Support_Ticket_Classifier_Qwen3.ipynb).
- Original data: [support_tickets.csv](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router/blob/05f7ea9643495741bda4e272bd1977105c3ad529/support_tickets.csv).
- CSV SHA-256: `c55c98b462f1af9b2f5e4ad8a7460a6b598e55bcfa42da1b57dc9e921e82783f`.

The CSV has 585 rows and the columns `category_truth` and `text`. It contains seven support categories. No license file was found in the reviewed source repository. Attribution is retained; this project does not assert a new license for the course dataset or imply that its labels have received independent human review.

The implementation follows the source workflow and dataset. It adds explicit version pins, preparation records, CPU baselines, case-level evaluation, and matched base/adapter/merged comparisons. Notebook-visible CLI training replaces the source's public LLaMA Board workflow for use with free Colab.

## Model and tools

- [Qwen3 1.7B Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base): model source and Apache 2.0 license. The selected revision is `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Qwen3 requires a compatible Transformers version; the model card identifies versions older than 4.51 as unsupported.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/702005ce5237ae10f3c572c3f7741ecfd1422fc9): pinned training framework.
- [SFT configuration](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html): reference for LoRA training arguments and CLI use.
- [Fine-tuning arguments](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html): LoRA rank and alpha definitions.
- [LoRA merging](https://llamafactory.readthedocs.io/en/latest/getting_started/merge_lora.html): merge with an unquantized base and the matching template.
- [Bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes): training low-rank parameters with a quantized base.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): free runtime limitations and resource availability.

Course examples and their recorded outputs are reference material. Only this project's completed execution artifacts may be reported as its training or evaluation results.
