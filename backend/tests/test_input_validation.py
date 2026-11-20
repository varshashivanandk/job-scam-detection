"""Bad input on /api/analyze-job: empty or non-string -> 400; symbols, digits or too little text -> Insufficient
Information, and the classifier is never called. The relevance gate is off here (conftest); see
test_relevance_gate.py for it."""
import logging

import pytest

pytest.importorskip("flask")
from flask import Flask

from backend.routes import job_routes

THRESHOLD = 0.291163


class SpyClassifier:
    def __init__(self):
        self.seen = []

    def predict(self, text):
        self.seen.append(text)
        return {"model": "DistilBERT", "prediction": 0, "label": "legitimate", "fraud_probability": 0.01,
                "legitimate_probability": 0.99, "threshold": THRESHOLD, "fraud_logodds": -4.6}


@pytest.fixture()
def client_and_spy(monkeypatch):
    spy = SpyClassifier()
    monkeypatch.setattr(job_routes, "get_classifier", lambda: spy)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(job_routes.job_bp)
    return app.test_client(), spy


def post(client, body):
    return client.post("/api/analyze-job", json=body)


@pytest.mark.parametrize("body", [{"text": ""}, {"text": None}, {"text": 123}, {"text": 4.5}, {"text": "   \n\t "},
                                  {"text": ["a list"]}, {}, None])
def test_empty_none_number_and_whitespace_are_rejected_with_a_clear_400(client_and_spy, body):
    client, spy = client_and_spy
    response = post(client, body)
    assert response.status_code == 400
    assert response.get_json()["error"] == "Please provide the job posting text."
    assert spy.seen == []


@pytest.mark.parametrize("text", ["!!!???", "12345 67890", "$$$ ### 1234-5678-9012 !!! ??? 98765 43210 ***",
                                  "0000 1111 2222 3333 4444 5555 6666 7777 8888 9999 x"])
def test_symbols_or_digits_return_insufficient_information(client_and_spy, text):
    client, spy = client_and_spy
    response = post(client, {"text": text})
    assert response.status_code == 200
    body = response.get_json()
    assert body["verdict"] == "Insufficient Information" and body["display_risk_score"] is None
    assert spy.seen == []


def test_letter_share_rule_applies_even_when_text_is_long_enough():
    assert job_routes.has_too_few_letters({"description": "12345 67890 12345 67890 12345 67890 !!! ok"})
    assert not job_routes.has_too_few_letters({"description": "Data entry clerk, 2 years experience, Rs 15,000"})
    assert job_routes.has_too_few_letters({"description": "   "})


@pytest.mark.parametrize("text", ["Pay a fee.", "Apply now for this job.", "Hiring drivers today"])
def test_a_short_pasted_sentence_returns_insufficient_information(client_and_spy, text):
    client, spy = client_and_spy
    body = post(client, {"text": text}).get_json()
    assert body["verdict"] == "Insufficient Information" and body["risk_score"] is None
    assert spy.seen == []


def test_the_form_also_rejects_symbols_and_digits(client_and_spy):
    client, spy = client_and_spy
    body = post(client, {"title": "!!!", "description": "1234567890 1234567890 1234567890 ###"}).get_json()
    assert body["verdict"] == "Insufficient Information" and spy.seen == []


def test_a_normal_pasted_posting_still_reaches_the_classifier(client_and_spy):
    client, spy = client_and_spy
    text = "Data entry clerk needed. Full-time, interview required, apply with your CV."
    body = post(client, {"text": text}).get_json()
    assert body["verdict"] == "Likely Legitimate" and spy.seen == [f"Description: {text}"]


def test_submitted_text_is_never_logged(client_and_spy, caplog):
    client, _ = client_and_spy
    secret = "### 9999 8888 7777 secret-token-xyz ###"
    with caplog.at_level(logging.DEBUG):
        post(client, {"text": secret})
        post(client, {"text": "secret-token-abc"})
    assert "secret-token" not in caplog.text
