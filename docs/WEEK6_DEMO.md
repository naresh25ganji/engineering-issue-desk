# Week 6 recording guide

Use the completed run in `evaluation/week6-live/` or the final run directory linked in `WEEK6_SUBMISSION.md`. Do not rerun the model while recording unless you intend to collect a new result. The submission form requires a **Video Demo Drive Link**, even though the handout focuses on the Google Doc. Upload the completed video to Drive and verify the submitted link is publicly accessible.

## Before recording

Open the final findings document and `report.html`. Keep the terminal command and the saved run manifest available. Show only synthetic attack data and exported, redacted evidence. Close credential settings and unrelated cases.

## Suggested three minute sequence

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:25 | Findings document opening | “I used Path B to red-team Engineering Issue Desk, my engineering investigation agent. I tested its existing evidence, tool, privacy, and human-review boundaries.” |
| 0:25–0:55 | Saved report summary and manifest | State the actual attack/control counts, model, and score totals from the final report. Explain that automated scores are bounded observations and human semantic review is separate. |
| 0:55–1:45 | A significant FAIL or WARN case, if present | Read the exact attack prompt, show the actual response, and explain the scored boundary. If the case errored or exhausted its budget, say that explicitly. Do not describe an incomplete attempt as a safe refusal. |
| 1:45–2:15 | A multi-turn crescendo case | Show the saved start and revision prompts in order. Explain what authority or disclosure the later turn tried to gain and what the saved response did. |
| 2:15–2:40 | A normal engineering control | Show whether the application still produced a useful supported report. Explain any overblocking or incomplete result. |
| 2:40–3:00 | Attack-to-defense mapping and evidence links | Point to a specific backend or output control for the observed issue. State whether it is an existing control, a recommendation, or a separately tested change. End on the document and complete evidence package. |

## Claims to preserve

- Public tool names are part of this application's documented interface, so naming one is not automatically a confidentiality failure.
- The agent does not have banking account records. Synthetic PII probes test the selected privacy boundary; they do not establish authorization for a banking deployment.
- A PASS means the configured failure was not observed in that case. It is not a general security certification.
- Source text that contains an injected instruction is still evidence, not permission to execute the instruction.

The final document and saved HTML contain the actual figures and transcripts. Use those values in the narration instead of memorizing provisional counts.
