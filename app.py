"""Human preference collection app for small local LMs (GPT-2, Qwen2.5-0.5B-Instruct).

Run with:  streamlit run app.py
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import truststore

import storage

# Verify HTTPS (Hugging Face downloads) against the OS certificate store, which
# includes any proxy/antivirus root certificates that certifi lacks.
truststore.inject_into_ssl()

DB_PATH = os.getenv("PREFERENCE_DB", str(Path(__file__).parent / "preferences.db"))
MODELS = (
    "gpt2",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "google/gemma-3-270m-it",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
EXPORT_FILENAME = "human_pref_pairs.jsonl"
DPO_EXPORT_FILENAME = "dpo_pairs.jsonl"
PREFERENCE_LABELS = {
    "A": "A is better",
    "B": "B is better",
    "tie": "Tie",
    "both_bad": "Both are bad",
}


def now_iso():
    """Current UTC time as an ISO 8601 string with second precision, e.g. 2026-10-04T12:50:45Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Generation

@st.cache_resource(show_spinner="Loading model...", max_entries=1)
def load_model(name):
    """Load a Hugging Face causal LM and its tokenizer onto the GPU if available.

    Cached across reruns; max_entries=1 keeps only the last-used model in memory.
    Returns (tokenizer, model, device, eos_ids), where eos_ids lists every token
    that ends a response.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    # Qwen's generation config stops on both <|im_end|> and <|endoftext|>.
    eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = list(eos_ids) if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    return tokenizer, model, device, eos_ids


def format_model_prompt(model_name, prompt):
    """Return the prompt text the model is conditioned on, minus the trailing newline.

    Responses are sampled from model_prompt + "\\n", matching a prompt + "\\n" +
    response training format. Chat models get their chat template; base models
    get raw text.
    """
    tokenizer = load_model(model_name)[0]
    if not tokenizer.chat_template:
        return prompt
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    # Some templates (e.g. Gemma) start with the BOS token, which the tokenizer
    # also adds itself; keep only one so the model doesn't see it twice.
    bos = tokenizer.bos_token
    if bos and templated.startswith(bos) and tokenizer("x")["input_ids"][0] == tokenizer.bos_token_id:
        templated = templated[len(bos):]
    return templated.rstrip("\n")


def generate_one(model_name, model_prompt, settings):
    """Sample one response to model_prompt with the given seed, temperature, top_p and max_new_tokens.

    Returns (text, metadata), where metadata holds latency, finish reason ("eos"
    if the model stopped on its own, "length" if it hit max_new_tokens), the
    model's own sampling defaults, token counts, and device.
    """
    import torch
    from transformers import set_seed

    tokenizer, model, device, eos_ids = load_model(model_name)
    inputs = tokenizer(model_prompt + "\n", return_tensors="pt").to(device)
    set_seed(settings["seed"])
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=settings["max_new_tokens"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
    latency = time.perf_counter() - start
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[0, prompt_len:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    gen_cfg = model.generation_config
    return text, {
        "latency_s": round(latency, 3),
        "finish_reason": "eos" if any(t in eos_ids for t in new_tokens) else "length",
        "model_defaults": {"top_k": gen_cfg.top_k, "repetition_penalty": gen_cfg.repetition_penalty},
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": len(new_tokens)},
        "device": device,
        "generated_at": now_iso(),
    }


# Callbacks

def save_preference(preference):
    """Button callback: save the pending comparison with the chosen label, then clear it for the next prompt."""
    p = st.session_state.pending
    if p is None:
        return
    record = {
        "id": p["id"],
        "created_at": now_iso(),
        "generated_at": p["generated_at"],
        "session_id": st.session_state.session_id,
        "annotator": st.session_state.get("annotator") or None,
        "prompt": p["prompt"],
        "model_prompt": p["model_prompt"],
        "response_a": p["response_a"],
        "response_b": p["response_b"],
        "preference": preference,
        "confidence": st.session_state.get("confidence"),
        "note": st.session_state.get("note") or None,
        "decision_seconds": round(time.time() - p["shown_at"], 2),
        "generation_config": p["gen"],
        "meta_a": p["meta_a"],
        "meta_b": p["meta_b"],
    }
    storage.save_comparison(DB_PATH, record)
    st.session_state.last_saved = (record["id"], PREFERENCE_LABELS[preference])
    reset_comparison()


def delete_records(ids):
    """Button callback: delete the selected comparisons and remember how many were removed."""
    st.session_state.last_deleted = storage.delete_comparisons(DB_PATH, ids)
    # A fresh table key clears the row selection, which would otherwise point at shifted rows.
    st.session_state.table_version += 1


def reset_comparison(clear_prompt=True):
    """Unlock the prompt and reset the labeling widgets.

    Also used by the Discard button with clear_prompt=False, which keeps the
    prompt text so it can be regenerated with different settings.
    """
    st.session_state.pending = None
    st.session_state.note = ""
    st.session_state.confidence = "Somewhat sure"
    if clear_prompt:
        st.session_state.prompt_input = ""


# UI

st.set_page_config(page_title="Preference Collector", layout="wide")
storage.init_db(DB_PATH)

ss = st.session_state
ss.setdefault("session_id", str(time.time_ns()))
ss.setdefault("pending", None)
ss.setdefault("last_saved", None)
ss.setdefault("last_deleted", None)
ss.setdefault("table_version", 0)
ss.setdefault("confidence", "Somewhat sure")

with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox("Model", MODELS)

    st.subheader("Per response")
    ca, cb = st.columns(2)
    seed_a = ca.number_input("Seed A", 0, 2**32 - 1, 1, 1)
    seed_b = cb.number_input("Seed B", 0, 2**32 - 1, 2, 1)
    temp_a = ca.number_input("Temperature A", 0.05, 2.0, 0.7, 0.05)
    temp_b = cb.number_input("Temperature B", 0.05, 2.0, 1.0, 0.05)
    if seed_a == seed_b and temp_a == temp_b:
        st.warning("Same seed and temperature: A and B will be identical.")

    st.subheader("Shared")
    top_p = st.slider("Top-p", 0.05, 1.0, 0.95, 0.05)
    max_new_tokens = st.number_input("Max new tokens", 16, 512, 90, 8)

    st.header("Annotator")
    st.text_input("Annotator ID (optional)", key="annotator")
    st.caption(f"DB `{Path(DB_PATH).name}`")

st.title("Pairwise Preference Collector")
st.caption(f"Enter a prompt, compare two samples from `{model_name}`, and record which you prefer.")

if ss.last_saved:
    st.success(f"Saved {ss.last_saved[1]!r} (example_id `{ss.last_saved[0]}`).")
    ss.last_saved = None

pending = ss.pending
st.text_area("Prompt", key="prompt_input", height=140, disabled=pending is not None,
             placeholder="Ask me anything")

if pending is None:
    if st.button("Generate two responses", type="primary",
                 disabled=not ss.prompt_input.strip()):
        shared = {"top_p": round(top_p, 2), "max_new_tokens": int(max_new_tokens)}
        gen = {
            "model": model_name,
            "A": {"seed": int(seed_a), "temperature": round(temp_a, 2), **shared},
            "B": {"seed": int(seed_b), "temperature": round(temp_b, 2), **shared},
        }
        prompt = ss.prompt_input.strip()
        with st.spinner("Generating..."):
            try:
                model_prompt = format_model_prompt(model_name, prompt)
                text_a, meta_a = generate_one(model_name, model_prompt, gen["A"])
                text_b, meta_b = generate_one(model_name, model_prompt, gen["B"])
            except Exception as e:
                st.error(f"Generation failed: {e}")
                if "gated repo" in str(e):
                    st.info(f"`{model_name}` requires accepting its license at "
                            f"https://huggingface.co/{model_name} and logging in with "
                            "`hf auth login` (see README).")
                st.stop()
        ss.pending = {
            "id": str(time.time_ns() // 1_000_000),
            "prompt": prompt,
            "model_prompt": model_prompt,
            "gen": gen,
            "response_a": text_a,
            "response_b": text_b,
            "meta_a": meta_a,
            "meta_b": meta_b,
            "generated_at": now_iso(),
            "shown_at": time.time(),
        }
        st.rerun()
else:
    st.info("Responses are locked for this prompt. Record a preference or discard to start over.")
    if pending["response_a"] == pending["response_b"]:
        st.warning("A and B are identical; this comparison carries no preference signal.")
    if pending["model_prompt"] != pending["prompt"]:
        with st.expander("Exact model input (chat template applied)"):
            st.code(pending["model_prompt"] + "\n", language=None)
    cols = st.columns(2)
    for col, label, text, meta in (
        (cols[0], "A", pending["response_a"], pending["meta_a"]),
        (cols[1], "B", pending["response_b"], pending["meta_b"]),
    ):
        settings = pending["gen"][label]
        with col:
            st.subheader(f"Response {label}")
            st.caption(f"seed {settings['seed']} · temperature {settings['temperature']}")
            with st.container(border=True, height=360):
                st.text(text or "(empty response)")
            st.caption(
                f"{meta['latency_s']}s · {meta['usage']['completion_tokens']} tokens · "
                f"finish: {meta['finish_reason']}"
            )
            with st.expander("Generation metadata"):
                st.json({**settings, **meta})

    st.divider()
    st.subheader("Which response do you prefer?")
    c1, c2 = st.columns([1, 2])
    c1.select_slider("Confidence", ["Unsure", "Somewhat sure", "Very sure"], key="confidence")
    c2.text_input("Reason (optional)", key="note",
                  placeholder="Why? e.g. more on-topic, less repetitive, more cautious...")

    bcols = st.columns(5)
    for bcol, (pref, label) in zip(bcols, PREFERENCE_LABELS.items()):
        bcol.button(label, on_click=save_preference, args=(pref,),
                    width="stretch", type="primary" if pref in ("A", "B") else "secondary")
    bcols[4].button("Discard (don't save)", on_click=reset_comparison, kwargs={"clear_prompt": False},
                    width="stretch")

# Data & export

st.divider()
st.header("Collected data")
records = storage.load_comparisons(DB_PATH)

if ss.last_deleted is not None:
    st.success(f"Deleted {ss.last_deleted} row(s).")
    ss.last_deleted = None

if not records:
    st.write("No comparisons recorded yet.")
else:
    counts = pd.Series([r["preference"] for r in records]).value_counts()
    mcols = st.columns(5)
    mcols[0].metric("Total", len(records))
    for mcol, (pref, label) in zip(mcols[1:], PREFERENCE_LABELS.items()):
        mcol.metric(label, int(counts.get(pref, 0)))

    df = pd.DataFrame(records)
    df["model"] = [r["generation_config"]["model"] for r in records]
    table = df[["id", "created_at", "model", "prompt", "preference", "note", "confidence",
                "annotator", "decision_seconds"]].iloc[::-1].reset_index(drop=True)
    event = st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={"id": "example_id", "created_at": "timestamp", "note": "reason"},
        on_select="rerun", selection_mode="multi-row", key=f"records_table_{ss.table_version}",
    )
    selected_ids = table.loc[event.selection.rows, "id"].tolist()
    with st.popover(f"Delete selected ({len(selected_ids)})", disabled=not selected_ids):
        st.write(f"Permanently delete {len(selected_ids)} row(s)? This cannot be undone.")
        st.button("Confirm delete", type="primary", on_click=delete_records, args=(selected_ids,))
    st.caption("Select rows with the checkboxes on the left of the table to delete them.")

    st.subheader("Export")
    model_records = [r for r in records if r["generation_config"]["model"] == model_name]
    examples = storage.to_example_records(model_records)
    pairs = storage.to_preference_pairs(model_records)
    st.caption(
        f"Exports cover `{model_name}` only (switch the sidebar model to export another model's data). "
        f"`{EXPORT_FILENAME}`: {len(examples)} labeled comparisons, ties included. "
        f"`{DPO_EXPORT_FILENAME}`: {len(pairs)} {{prompt, chosen, rejected}} pairs, "
        "ties and 'both bad' excluded."
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ecols = st.columns(3)
    ecols[0].download_button(
        EXPORT_FILENAME, storage.to_jsonl(examples),
        file_name=EXPORT_FILENAME, mime="application/jsonl",
        disabled=not examples, width="stretch", type="primary",
    )
    ecols[1].download_button(
        DPO_EXPORT_FILENAME, storage.to_jsonl(pairs),
        file_name=DPO_EXPORT_FILENAME, mime="application/jsonl",
        disabled=not pairs, width="stretch",
    )
    ecols[2].download_button(
        "All records with metadata (.csv)", df.to_csv(index=False),
        file_name=f"comparisons_raw_{stamp}.csv", mime="text/csv",
        width="stretch",
    )
    st.subheader(f"Preview `{EXPORT_FILENAME}` entry")
    if examples:
        latest_first = examples[::-1]
        choice = st.selectbox(
            "Entry", range(len(latest_first)),
            format_func=lambda i: (f"{latest_first[i]['timestamp']} · {latest_first[i]['preference']} · "
                                   f"{latest_first[i]['prompt'][:60]}"),
        )
        st.json(latest_first[choice])
    else:
        st.write(f"No `{model_name}` entries yet.")
