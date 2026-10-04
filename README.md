# AIPI 591: Lab 6: Human Preference Collection

### Author: Matana Pornluanprasert

A Streamlit app that collects human preference data for training language models. The user enters a prompt, the app samples two responses from the same local model, and the user picks the better one (or marks a tie, or both bad). The responses stay fixed once generated, every comparison is saved to a database with its generation settings, and the data is exported both as full records and as `{prompt, chosen, rejected}` pairs for DPO training.

---

## Preference Collector App

`app.py` runs the models locally with Hugging Face `transformers` (on the GPU if available).

1. **Generate:** enter a prompt and sample two responses, A and B.
2. **Compare:** the responses are locked, and the prompt can't be edited until the pair is labeled or discarded.
3. **Label:** A is better, B is better, Tie, or Both are bad, with an optional confidence level and reason.
4. **Manage and export:** a table of all comparisons (select rows to delete them), a preview of any entry, and download buttons for the exports.

| Setting | Scope | Default |
| ------- | ----- | ------- |
| Model | Shared | `gpt2`, `SmolLM2-135M-Instruct`, `gemma-3-270m-it`, or `Qwen2.5-0.5B-Instruct` |
| Seed | Per response | A: 1, B: 2 |
| Temperature | Per response | A: 0.7, B: 1.0 |
| Top-p | Shared | 0.95 |
| Max new tokens | Shared | 90 |

---

## Design Choices

* **Same model, different seeds and temperatures:** both responses come from the same model and prompt, so the preference compares samples, not models. Fixed seeds make every response reproducible. If A and B have the same seed and temperature, the sidebar warns that they will be identical.
* **Responses locked once generated:** the pair is kept in session state, so Streamlit reruns never regenerate it. This stops the annotator from re-rolling until a favorite appears.
* **Training-format model input:** responses are sampled from `prompt + "\n"`, the same format a `prompt + "\n" + response` training loop sees. Instruction-tuned models get their chat template applied first, and the exported DPO prompt is this exact model input.
* **"Both are bad" kept separate from "Tie":** both carry no ordering signal for DPO, but they mean different things for data analysis. Ties are kept in the full export and only excluded from the DPO pairs.
* **Exports per model:** each export only includes the model selected in the sidebar, so pairs from different models are never mixed in one training file.

---

## Collected Data

The `collected_data/` folder contains a sample of data collected from a user with this app. It is available in three formats, described under [Export Format](#export-format):

* `human_pref_pairs.jsonl`: full labeled comparisons
* `dpo_pairs.jsonl`: `{prompt, chosen, rejected}` pairs
* `comparisons_raw_*.csv`: every record with full metadata

---

## Export Format

**`human_pref_pairs.jsonl`:** one labeled comparison per line, ties included.

```json
{"example_id": "1791117869826", "timestamp": "2026-10-04T12:50:45Z",
 "prompt": "...", "response_A": "...", "response_B": "...",
 "preference": "tie", "reason": "...",
 "gen": {"model": "HuggingFaceTB/SmolLM2-135M-Instruct",
         "A": {"seed": 1, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 90},
         "B": {"seed": 2, "temperature": 1.0, "top_p": 0.95, "max_new_tokens": 90}}}
```

`preference` is `A`, `B`, `tie`, or `both_bad`. `example_id` is the generation time in epoch milliseconds, and `timestamp` is when the label was saved (UTC).

**`dpo_pairs.jsonl`:** `{"prompt", "chosen", "rejected"}` per line, for DPO or reward-model training. `prompt` is the exact model input (chat template applied). Ties, "both bad", and empty or identical pairs are excluded.

**`comparisons_raw_*.csv`:** every record with full metadata: exact model input, confidence, decision time, and per-response latency, token counts, finish reason, and device.

---

## Files

```
app.py               Streamlit app: generation, labeling, data table, exports
storage.py           SQLite storage and export formats
requirements.txt     Python dependencies
collected_data/      Sample of collected data, in three formats
  human_pref_pairs.jsonl
  dpo_pairs.jsonl
  comparisons_raw_20261005_010735.csv
```

`preferences.db` is created on first run and is not committed. The exports in `collected_data/` are the versioned copy of the data.

---

## Ethics Statement

This project is for research and education in learning from human feedback. Prompts should not contain personal data. The responses come from very small models, can be wrong or unsafe, and are not professional advice. The labels reflect their annotator's judgment, so a model trained on them learns that person's preferences, not a general standard. Data like this should not be used to build a real advice system without expert review and many more annotators.

---

## Requirements and How to Run

**Requirements** (Python 3.11):

```
streamlit==1.65.0
transformers==5.18.0
huggingface_hub==1.33.0
pandas==3.0.6
truststore==0.10.4
torch==2.14.0
```

`truststore` makes model downloads work on networks that inspect HTTPS (antivirus or corporate proxies) by using the Windows certificate store.

**Install dependencies:**

```
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch (the default Windows build is CPU-only):

```
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
```

**Run the app**:

```
streamlit run app.py
```

Each model is downloaded from Hugging Face the first time it is used.
