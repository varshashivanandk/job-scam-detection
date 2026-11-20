"""The "is this a job posting?" gate (backend/services/relevance.py) and its place in /api/analyze-job.

Rule tests use fake signals. Tests marked with the real embedder skip when all-MiniLM-L6-v2 is not available
locally; the EMSCAD test also skips without the validation split or the DistilBERT model. Test data is never used.
"""
import json
from pathlib import Path

import pytest

from backend.services import relevance as rel

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = {s["id"]: s for s in json.loads((Path(__file__).parent / "fixtures" / "non_job_samples.json")
                                          .read_text(encoding="utf-8"))}
LONG_JOB = ("We are hiring a full-time customer support executive for our Bengaluru office. Responsibilities: answer "
            "customer calls and emails, log tickets and follow up. Requirements: graduate, one year of experience, "
            "good English. Benefits: health insurance and paid leave. Apply with your resume.")
NEWS = SAMPLES["news-2"]["text"]


# --------------------------------------------------------------------------- the fixed rule (no model needed)
def test_job_terms_and_headings_raise_the_keyword_score():
    kw = rel.keyword_signal(LONG_JOB)
    assert kw["score"] > rel.KEYWORD_REJECT_AT_MOST and kw["headings"] >= 2 and not kw["news_cues"]


def test_news_cues_lower_the_keyword_score():
    kw = rel.keyword_signal(NEWS)
    assert "wire" in kw["news_cues"] and kw["score"] <= rel.KEYWORD_REJECT_AT_MOST


def test_job_like_text_passes_without_calling_the_embedder():
    def embed(texts):
        raise AssertionError("embedder must not be called when keywords already show a job posting")

    ok, score, reasons = rel.is_job_posting(LONG_JOB, embed=embed)
    assert ok and score is None and reasons


def test_short_text_is_never_judged():
    ok, score, reasons = rel.is_job_posting("The weather is lovely today in the hills.", embed=lambda t: 1 / 0)
    assert ok and score is None and "Too short to judge" in reasons[0]


def test_a_failing_embedder_lets_the_text_through():
    def broken(texts):
        raise OSError("model files missing")

    ok, score, reasons = rel.is_job_posting(SAMPLES["recipe-1"]["text"], embed=broken)
    assert ok and score is None and "unavailable" in reasons[-1]


@pytest.mark.parametrize("margin, expected", [(rel.SEMANTIC_REJECT_BELOW - 0.01, False),
                                              (rel.SEMANTIC_REJECT_BELOW, True),
                                              (rel.SEMANTIC_REJECT_BELOW + 0.01, True), (0.3, True)])
def test_only_text_clearly_below_the_semantic_threshold_is_rejected(monkeypatch, margin, expected):
    monkeypatch.setattr(rel, "semantic_signal", lambda text, embed: margin)
    ok, score, _ = rel.is_job_posting(SAMPLES["recipe-1"]["text"], embed=object())
    assert ok is expected and score == margin


@pytest.mark.parametrize("margin, expected", [(rel.NEWS_SEMANTIC_REJECT_BELOW - 0.01, False),
                                              (rel.NEWS_SEMANTIC_REJECT_BELOW + 0.01, True)])
def test_a_news_cue_uses_the_stricter_cutoff(monkeypatch, margin, expected):
    monkeypatch.setattr(rel, "semantic_signal", lambda text, embed: margin)
    text = SAMPLES["news-3"]["text"]  # "police said in a statement"
    assert "statement" in rel.keyword_signal(text)["news_cues"]
    assert rel.is_job_posting(text, embed=object())[0] is expected


def test_the_noisy_dateline_cue_does_not_trigger_the_stricter_cutoff(monkeypatch):
    monkeypatch.setattr(rel, "semantic_signal", lambda text, embed: rel.SEMANTIC_REJECT_BELOW + 0.01)
    text = ("SALES EXECUTIVE - Our store sells furniture and home decor across the city, and our showroom is open "
            "every day of the week for walk-in customers and families looking for good value.")
    assert rel.keyword_signal(text)["news_cues"] == ["dateline"]
    assert rel.is_job_posting(text, embed=object())[0] is True


def test_a_text_with_job_terms_is_not_rejected_even_if_it_reads_like_news(monkeypatch):
    monkeypatch.setattr(rel, "semantic_signal", lambda text, embed: -0.9)
    assert rel.is_job_posting(LONG_JOB, embed=object())[0] is True


@pytest.mark.parametrize("ok, score, expected", [(True, None, False), (True, rel.NOTE_BELOW - 0.01, True),
                                                 (True, rel.NOTE_BELOW, False), (True, 0.3, False),
                                                 (False, -0.5, False)])
def test_the_note_is_only_for_borderline_text_that_was_let_through(ok, score, expected):
    assert rel.needs_note(ok, score) is expected


@pytest.mark.parametrize("gate_score, has_note", [(None, False), (-0.01, True), (0.3, False)])
def test_the_route_adds_the_note_without_changing_the_result(monkeypatch, gate_score, has_note):
    pytest.importorskip("flask")
    from flask import Flask

    from backend.routes import job_routes

    class Fixed:
        def predict(self, text):
            return {"model": "DistilBERT", "prediction": 0, "label": "legitimate", "fraud_probability": 0.01,
                    "legitimate_probability": 0.99, "threshold": 0.291163, "fraud_logodds": -4.6}

    monkeypatch.setattr(job_routes, "get_classifier", lambda: Fixed())
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(job_routes.job_bp)
    client = app.test_client()
    text = SAMPLES["encyclopedia-1"]["text"]

    monkeypatch.setattr(job_routes, "get_relevance_fn", lambda: lambda t: (True, None, []))
    plain = client.post("/api/analyze-job", json={"text": text}).get_json()
    monkeypatch.setattr(job_routes, "get_relevance_fn", lambda: lambda t: (True, gate_score, []))
    body = client.post("/api/analyze-job", json={"text": text}).get_json()

    assert ("relevance_note" in body) is has_note
    if has_note:
        assert body["relevance_note"] == rel.RELEVANCE_NOTE
    for key in job_routes.FIXED_REPORT_KEYS:
        assert body[key] == plain[key]


def test_gate_text_puts_job_content_first_and_has_no_training_labels():
    text = rel.gate_text({"company_profile": "<p>Acme</p>", "title": "Clerk", "description": "Enter data.",
                          "requirements": None, "benefits": ""})
    assert text == "Clerk\nEnter data.\nAcme" and "Title:" not in text


def test_non_string_input_is_let_through():
    assert rel.is_job_posting(None)[0] is True


# --------------------------------------------------------------------------- real embedder
@pytest.fixture(scope="module")
def embed():
    try:
        return rel.default_embedder()
    except Exception as exc:  # model files not cached locally
        pytest.skip(f"all-MiniLM-L6-v2 not available locally: {type(exc).__name__}")


@pytest.mark.parametrize("sample_id", ["news-1", "news-2", "news-3", "recipe-1", "poem-1"])
def test_news_recipes_and_poems_are_not_job_postings(embed, sample_id):
    ok, score, reasons = rel.is_job_posting(SAMPLES[sample_id]["text"], embed=embed)
    assert ok is False, reasons
    news = any(c in rel.STRICT_NEWS_CUES for c in rel.keyword_signal(SAMPLES[sample_id]["text"])["news_cues"])
    assert score < (rel.NEWS_SEMANTIC_REJECT_BELOW if news else rel.SEMANTIC_REJECT_BELOW)


# Texts the owner pasted while testing on 2026-11-15 (two of them got through the first version of the gate).
OWNER_EXAMPLES = [
    "Live result updates of Para seat in the 2026 assembly election. Know who is leading, who won, who lost by "
    "what margin and who is the new Para MLA.",
    "List all pairs of states. Mark every pair where one state is final and the other is non-final. For every "
    "unmarked pair, check transitions for each input symbol. If a transition leads to an already marked pair, "
    "mark the current pair. Repeat until no more pairs can be marked.",
    "The weather forecast for Karnataka over the next 5 days indicates warm conditions with temperatures ranging "
    "from 27 °C to 28 °C, accompanied by a gentle breeze.",
]


@pytest.mark.parametrize("text", OWNER_EXAMPLES)
def test_the_owners_examples_are_not_job_postings(embed, text):
    ok, _, reasons = rel.is_job_posting(text, embed=embed)
    assert ok is False, reasons


def test_a_real_job_posting_passes(embed):
    assert rel.is_job_posting(LONG_JOB, embed=embed)[0] is True


@pytest.fixture()
def gated_client(monkeypatch, embed):
    pytest.importorskip("flask")
    from flask import Flask

    from backend.routes import job_routes

    class Spy:
        seen = []

        def predict(self, text):
            Spy.seen.append(text)
            return {"model": "DistilBERT", "prediction": 0, "label": "legitimate", "fraud_probability": 0.01,
                    "legitimate_probability": 0.99, "threshold": 0.291163, "fraud_logodds": -4.6}

    Spy.seen = []
    monkeypatch.setattr(job_routes, "get_classifier", lambda: Spy())
    monkeypatch.setattr(job_routes, "get_relevance_fn",
                        lambda: lambda text: rel.is_job_posting(text, embed=embed))  # real gate, shared embedder
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(job_routes.job_bp)
    return app.test_client(), Spy


@pytest.mark.parametrize("sample_id", ["news-1", "news-2", "news-3", "recipe-1", "poem-1"])
def test_the_route_returns_not_a_job_posting_without_running_the_classifier(gated_client, sample_id):
    client, spy = gated_client
    response = client.post("/api/analyze-job", json={"text": SAMPLES[sample_id]["text"]})
    assert response.status_code == 200
    body = response.get_json()
    assert body["verdict"] == "Not a Job Posting"
    assert body["verdict_reason"] == "This does not look like a job posting. Paste a job advertisement to check it."
    for key in ("risk_score", "risk_category", "display_risk_score", "classifier_risk_score", "classifier_details"):
        assert body[key] is None
    assert "evidence" not in body and spy.seen == []


def test_the_route_still_classifies_a_real_posting(gated_client):
    client, spy = gated_client
    body = client.post("/api/analyze-job", json={"text": LONG_JOB}).get_json()
    assert body["verdict"] == "Likely Legitimate" and len(spy.seen) == 1


# --------------------------------------------------------------------------- 5 EMSCAD validation postings
VAL_CSV = ROOT / "data" / "processed_v2" / "val.csv"


def test_five_validation_postings_reach_the_classifier_unchanged(monkeypatch, embed):
    pytest.importorskip("flask")
    if not VAL_CSV.exists():
        pytest.skip("validation split not available locally")
    from flask import Flask

    from backend.routes import job_routes
    from backend.services import dataset as ds
    from backend.services.job_analysis import build_job_report

    try:
        classifier = job_routes.get_classifier()
    except Exception as exc:
        pytest.skip(f"DistilBERT model not available: {type(exc).__name__}")
    val = ds.load_split("val", VAL_CSV.parent).sort_values(ds.ID_COL)
    picked = list(val[val[ds.LABEL_COL] == 0].head(3).itertuples()) + list(val[val[ds.LABEL_COL] == 1].head(2).itertuples())

    gate_calls = []

    def real_gate(text):
        result = rel.is_job_posting(text, embed=embed)
        gate_calls.append(result[0])
        return result

    monkeypatch.setattr(job_routes, "get_relevance_fn", lambda: real_gate)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(job_routes.job_bp)
    client = app.test_client()

    for row in picked:
        fields = {f: getattr(row, f) if isinstance(getattr(row, f), str) else "" for f in ds.TEXT_FIELDS}
        # What the route did before the gate existed: classify the training-format text and build the report.
        text, profile = job_routes.structured_posting(fields)
        before = build_job_report(classifier.predict(text), text, profile)
        after = client.post("/api/analyze-job", json=fields).get_json()
        for key in ("verdict", "verdict_reason", "risk_score", "display_risk_score", "classifier_risk_score"):
            assert after[key] == before[key], (row.job_id, key)
    assert gate_calls == [True] * 5
