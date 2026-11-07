# Completed Week 5 run

Run `20261106T040208Z` completed three-epoch LoRA training on a Tesla T4. The adapter was merged into a fresh unquantized base; a local-only reload verified the saved tokenizer and finite model outputs.

## Exploratory follow-up with `label_stop`

| Model | Correct / 117 | Accuracy | Macro F1 | Invalid / errors |
| --- | --- | --- | --- | --- |
| Unchanged base | 28 | 23.93% | 0.0986 | 7 / 0 |
| Trained adapter | 84 | 71.79% | 0.6895 | 0 / 0 |
| Standalone merged | 84 | 71.79% | 0.6895 | 0 / 0 |

These runs share the prompt, FP16 precision, decoder, and frozen ticket IDs. The merged model passed **5/5 illustrative smoke checks**. There are still 33 incorrect held-out routes.

**The decoder repair was chosen after observing the original held-out generation failures. These scores are exploratory, not an independent confirmation of generalization.** `label_stop` stops actual greedy generation immediately when the entire generated text, after stripping whitespace, equals an allowed label. It records `complete_label`; it does not claim that the model generated an end token or trim a completed bad output afterward.

## Original results retained

The original EOS-terminated base scored **23.93% accuracy**, macro F1 **0.0986**, with seven invalid outputs. Both the original adapter and merged runs scored **0%**, with **117/117 invalid outputs** each: category text was followed by unwanted continuation instead of EOS. Their raw outputs and failures remain in the evidence.

A diagnostic on one development ticket confirmed matching tokenizer vocabularies, matching Factory prompt IDs, and correct EOS supervision (`151645`). Both NF4 and FP16 inference still failed to terminate. This diagnostic does not establish the underlying training cause.

The separate handout A–G constrained-choice base scored **31.62% accuracy**, macro F1 **0.1255**. It uses a different inference protocol from label generation. Its saved handout comparison retains the original tuned model's 0% result.

## Verification and limits

- Frozen data: 468 development / 117 outer held-out tickets; outer tickets excluded from training and checkpoint selection.
- CPU baselines: majority **25.64%** accuracy; TF-IDF/logistic **67.52%** accuracy and **0.5737** macro F1.
- Regression suite: **73 tests passed**.
- Source labels, semantic near-duplicates, and ticket-family independence have not received independent human review. Exact normalized duplicates are rejected.
- Dollar cost was not measured; these results do not establish deployment cost or production reliability.

Evidence is saved under `artifacts/colab/20261106T040208Z`, including original and follow-up reports, raw predictions, `merge-proof.json`, training configuration, loss plot, diagnostics, and smoke checks. [Launch and adapter restore links](LAUNCH.md). Full merged weights remain only in the Colab runtime; the saved adapter can regenerate them.
