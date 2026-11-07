# Week 5 project links

- [Submission Google Doc](https://docs.google.com/document/d/1tNUrAiNqAjaGHmShcN9c1-qVdQfDBvBbvHT9bjC-oLc)
- [Word source in Drive](https://docs.google.com/document/d/1HI-TdDGooovZcdFgF3hbIg1TFRhAAzVR/edit?usp=drivesdk&ouid=117743947283173099816&rtpof=true&sd=true)
- [Actual run evidence in Drive](https://drive.google.com/file/d/1K6eW-VJrAKwaJh5FWrEtf8xRAJ_PVyzV/view?usp=drivesdk)
- [Open the notebook in Colab](https://colab.research.google.com/drive/1hp3rD-MzprA8XunuC4u74Hb7Tr2M_zvB)
- [Notebook file in Drive](https://drive.google.com/file/d/1hp3rD-MzprA8XunuC4u74Hb7Tr2M_zvB/view?usp=drivesdk)
- [Project upload bundle in Drive](https://drive.google.com/file/d/1qx37--StXO2sTMGCxk-ng4NdxB8iUofz/view?usp=drivesdk)
- [Verified adapter restore bundle in Drive](https://drive.google.com/file/d/1CtxVZ_m56UGcf9Pc4u9bfF8wNy1gVNk2/view?usp=drivesdk)
- [Local upload bundle](artifacts/week5-ticket-router.zip)

Training, merge, fresh reload, and evaluations completed on a Colab T4. The explicit `label_stop` follow-up scored **71.79% accuracy / 0.6895 macro F1**, with zero invalid outputs and 5/5 illustrative smoke checks. This repair was chosen after the original tuned generator failed on all 117 tickets; the follow-up is exploratory. See [RESULTS.md](RESULTS.md).

To reproduce, upload the project ZIP in the notebook's first code cell, select a GPU, and enable `RUN_GPU_STEPS`. Enable `RUN_LABEL_STOP_REPAIR` explicitly in the optional follow-up section. **Both flags are False in the source notebook.** Each run writes to a new directory; download evidence and weights before the runtime ends.

Full merged weights remain only in the Colab runtime. The verified adapter restore bundle can regenerate the standalone model using the pinned base checkpoint. The saved notebook does not contain model weights.
