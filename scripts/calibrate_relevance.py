"""Calibrate the "is this a job posting?" gate (backend/services/relevance.py).

Uses ONLY the EMSCAD train and validation splits plus backend/tests/fixtures/non_job_samples.json. Never opens
the test split. For every posting it scores two variants, because both reach the gate in the app:
  * fields: title, description, requirements, benefits and company profile joined (the form);
  * description only (a pasted posting).
Only texts the gate would judge (at least MIN_WORDS_TO_JUDGE words) with a keyword score at or below
KEYWORD_REJECT_AT_MOST reach the semantic check, so only those are embedded. The report gives, per variant,
the lowest semantic margin (overall and among texts with a strict news cue) and how many postings the current
rule would wrongly reject (must be 0), plus the result for each non-job sample.
Writes reports/relevance_calibration.json.

Usage (repo root, venv active): python scripts/calibrate_relevance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402
from backend.services import relevance as rel  # noqa: E402

PROCESSED = ROOT / "data" / "processed_v2"
SAMPLES = ROOT / "backend" / "tests" / "fixtures" / "non_job_samples.json"
OUT = ROOT / "reports" / "relevance_calibration.json"


def margins(texts: list[str], embed, batch: int = 256) -> np.ndarray:
    """Semantic margin per text (best window), embedding all windows in batches."""
    if not texts:
        return np.array([])
    job, non_job = rel._prototypes(embed)
    owners, windows = [], []
    for i, text in enumerate(texts):
        for w in rel.chunks(text):
            owners.append(i)
            windows.append(w)
    vecs = np.vstack([embed(windows[s:s + batch]) for s in range(0, len(windows), batch)])
    per_window = (vecs @ job.T).max(axis=1) - (vecs @ non_job.T).max(axis=1)
    out = np.full(len(texts), -np.inf)
    np.maximum.at(out, np.array(owners), per_window)
    return out


def score(texts: list[str], embed) -> list[dict]:
    """Per text: words, keyword score, strict news cue, and the margin when the semantic check would run."""
    rows = []
    for t in texts:
        kw = rel.keyword_signal(t)
        rows.append({"words": len(t.split()), "kw": kw["score"],
                     "news": any(c in rel.STRICT_NEWS_CUES for c in kw["news_cues"]), "sem": None})
    needs = [i for i, r in enumerate(rows) if r["words"] >= rel.MIN_WORDS_TO_JUDGE
             and r["kw"] <= rel.KEYWORD_REJECT_AT_MOST]
    for i, m in zip(needs, margins([texts[i] for i in needs], embed)):
        rows[i]["sem"] = float(m)
    return rows


def rejected(row: dict) -> bool:
    if row["sem"] is None:
        return False
    cutoff = rel.NEWS_SEMANTIC_REJECT_BELOW if row["news"] else rel.SEMANTIC_REJECT_BELOW
    return row["sem"] < cutoff


def main() -> int:
    embed = rel.default_embedder()
    report = {"rule": {"MIN_WORDS_TO_JUDGE": rel.MIN_WORDS_TO_JUDGE,
                       "KEYWORD_REJECT_AT_MOST": rel.KEYWORD_REJECT_AT_MOST,
                       "SEMANTIC_REJECT_BELOW": rel.SEMANTIC_REJECT_BELOW,
                       "NEWS_SEMANTIC_REJECT_BELOW": rel.NEWS_SEMANTIC_REJECT_BELOW,
                       "STRICT_NEWS_CUES": list(rel.STRICT_NEWS_CUES), "NOTE_BELOW": rel.NOTE_BELOW,
                       "prototypes": {"job": rel.JOB_PROTOTYPES, "non_job": rel.NON_JOB_PROTOTYPES}},
              "variants": {}, "non_job": []}
    for split in ("train", "val"):  # never "test"
        df = ds.load_split(split, PROCESSED)
        records = df.to_dict("records")
        for variant, fields in (("fields", rel.GATE_FIELDS), ("description", ("description",))):
            rows = score([rel.gate_text({f: r.get(f) for f in fields}) for r in records], embed)
            sems = [r["sem"] for r in rows if r["sem"] is not None]
            news_sems = [r["sem"] for r in rows if r["sem"] is not None and r["news"]]
            report["variants"][f"{split}_{variant}"] = {
                "postings": len(rows),
                "judged": sum(r["words"] >= rel.MIN_WORDS_TO_JUDGE for r in rows),
                "reached_semantic_check": len(sems),
                "lowest_margin": min(sems) if sems else None,
                "with_strict_news_cue": len(news_sems),
                "lowest_margin_with_news_cue": min(news_sems) if news_sems else None,
                "wrongly_rejected": sum(rejected(r) for r in rows),
                "shown_relevance_note": sum(rel.needs_note(not rejected(r), r["sem"]) for r in rows),
            }
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    for s, r in zip(samples, score([s["text"] for s in samples], embed)):
        report["non_job"].append({"id": s["id"], "kind": s["kind"], "keyword": r["kw"], "news_cue": r["news"],
                                  "semantic": None if r["sem"] is None else round(r["sem"], 4),
                                  "rejected": rejected(r),
                                  "relevance_note": rel.needs_note(not rejected(r), r["sem"])})
    report["postings_wrongly_rejected"] = sum(v["wrongly_rejected"] for v in report["variants"].values())
    report["non_job_rejected"] = sum(r["rejected"] for r in report["non_job"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, v in report["variants"].items():
        print(name, v)
    for r in report["non_job"]:
        print("  ", r)
    print(f"postings wrongly rejected: {report['postings_wrongly_rejected']} | "
          f"non-job rejected: {report['non_job_rejected']}/{len(samples)}")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
