# Post-run code inspection

These notes were prepared by Codex after inspecting the exported live reports and repository code. They are an additional AI-assisted engineering inspection, not human-scored accuracy or an alteration of the frozen results. Human fields in `report.json` remain null. All three cases remain unapproved.

| Case | Recorded workflow | Inspection notes |
| --- | --- | --- |
| W3-01: outcome labels | Saved at human review; one retained finding | The selected quote is a returned summary dictionary and disclosure string. It supports the label's intended meaning, but does not by itself demonstrate every counting/filtering condition claimed in the finding. Inspect the full implementation before approving. The reviewer removed the cross-process claim rather than consulting primary threading documentation. |
| W3-02: freshness | Failed during independent review; draft retained | Exact citation integrity passed before the reviewer failed. The first finding omits the higher-priority `unknown` branch when any timestamp is invalid. Its explanation needs that condition before approval. No completed evidence-review verdict or human decision is present. |
| W3-03: hosted embeddings | Saved at human review; two retained findings | The empty-key claim depends on `_configuration()`, whose implementation was outside the supplied file; the code calls this helper and handles `GenerationUnavailable`. A bounded excerpt cannot establish that the helper accepts an empty key. The second finding speculates about future error-body leakage even though the selected code records sanitized status/type information and omits raw provider bodies. The AI reviewer accepted both claims; neither is an established defect. |

The three-case run demonstrates measured workflow behavior and exposes quality limitations. It does not establish two correct reports out of three. Completion, registered literal quotations, AI acceptance, and human usefulness must remain separate measures.

Next useful evaluation work is to provide the helper implementation or read-only repository context, require claims to disclose omitted prerequisites, obtain actual human judgments, and compare a different reviewer model on the same frozen cases. These are future experiments, not completed integrations or results.
