"""SQLite persistence and export helpers for pairwise preference data."""

import json
import sqlite3
from contextlib import closing

PREFERENCES = ("A", "B", "tie", "both_bad")
JSON_FIELDS = ("generation_config", "meta_a", "meta_b")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS comparisons (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    generated_at      TEXT,
    session_id        TEXT,
    annotator         TEXT,
    prompt            TEXT NOT NULL,
    model_prompt      TEXT,
    response_a        TEXT NOT NULL,
    response_b        TEXT NOT NULL,
    preference        TEXT NOT NULL CHECK (preference IN {PREFERENCES}),
    confidence        TEXT,
    note              TEXT,
    decision_seconds  REAL,
    generation_config TEXT,
    meta_a            TEXT,
    meta_b            TEXT
)
"""

COLUMNS = (
    "id", "created_at", "generated_at", "session_id", "annotator",
    "prompt", "model_prompt", "response_a", "response_b",
    "preference", "confidence", "note", "decision_seconds",
    "generation_config", "meta_a", "meta_b",
)


def _connect(db_path):
    """Open a SQLite connection whose rows can be read like dicts (by column name)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path):
    """Create the comparisons table if it doesn't exist yet. Safe to call on every run."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(SCHEMA)


def save_comparison(db_path, record):
    """Insert one labeled comparison.

    `record` is a dict keyed by COLUMNS; the JSON_FIELDS values are dicts and are
    stored as JSON text. The primary key makes double-saves fail loudly.
    """
    if record["preference"] not in PREFERENCES:
        raise ValueError(f"Invalid preference: {record['preference']!r}")
    values = [
        json.dumps(record.get(c)) if c in JSON_FIELDS else record.get(c)
        for c in COLUMNS
    ]
    placeholders = ", ".join("?" for _ in COLUMNS)
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            f"INSERT INTO comparisons ({', '.join(COLUMNS)}) VALUES ({placeholders})",
            values,
        )


def delete_comparisons(db_path, ids):
    """Permanently delete the comparisons with the given ids and return how many were removed."""
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.executemany("DELETE FROM comparisons WHERE id = ?", [(i,) for i in ids])
        return cur.rowcount


def load_comparisons(db_path):
    """Return all comparisons, oldest first, as dicts with the JSON fields decoded."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM comparisons ORDER BY created_at"
        ).fetchall()
    records = []
    for row in rows:
        rec = dict(row)
        for field in JSON_FIELDS:
            rec[field] = json.loads(rec[field]) if rec[field] else None
        records.append(rec)
    return records


def to_example_records(records):
    """One row per labeled comparison (ties included), in the human_pref_pairs.jsonl format."""
    return [
        {
            "example_id": rec["id"],
            "timestamp": rec["created_at"],
            "prompt": rec["prompt"],
            "response_A": rec["response_a"],
            "response_B": rec["response_b"],
            "preference": rec["preference"],
            "reason": rec.get("note") or "",
            "gen": rec["generation_config"],
        }
        for rec in records
    ]


def to_preference_pairs(records):
    """Convert labeled comparisons into {prompt, chosen, rejected} rows for DPO.

    Ties and "both bad" carry no ordering signal, so they are skipped, as are
    pairs whose chosen or rejected text is empty or identical.

    The prompt is the exact text the model was conditioned on (e.g. with a chat
    template applied), so training sees the same input as sampling.
    """
    pairs = []
    for rec in records:
        if rec["preference"] == "A":
            chosen, rejected = rec["response_a"], rec["response_b"]
        elif rec["preference"] == "B":
            chosen, rejected = rec["response_b"], rec["response_a"]
        else:
            continue
        if not chosen.strip() or not rejected.strip() or chosen == rejected:
            continue
        pairs.append({
            "prompt": rec.get("model_prompt") or rec["prompt"],
            "chosen": chosen,
            "rejected": rejected,
        })
    return pairs


def to_jsonl(rows):
    """Serialize dicts as JSON Lines (one object per line), keeping non-ASCII text readable."""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
