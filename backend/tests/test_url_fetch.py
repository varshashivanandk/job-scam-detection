"""Link input: backend/services/url_fetch.py and the /api/analyze-job URL path. The network is always mocked:
DNS (socket.getaddrinfo) returns fixed addresses and url_fetch._open returns fake responses."""
import ipaddress
import json
import logging
import socket

import pytest
import requests

from backend.services import url_fetch as uf

PUBLIC = "93.184.216.34"
DNS = {"jobs.example.com": [PUBLIC], "careers.example.org": ["151.101.1.69"], "docs-range.example.com": ["203.0.113.7"], "internal.example.com": ["10.0.0.5"],
       "mixed.example.com": [PUBLIC, "192.168.1.10"], "metadata.example.com": ["169.254.169.254"],
       "v6local.example.com": ["::1"], "127.0.0.1": ["127.0.0.1"], "localhost": ["127.0.0.1"], "::1": ["::1"],
       "169.254.169.254": ["169.254.169.254"]}

JOB_HTML = """<html><head><title>Clerk | Example Careers</title></head><body>
<nav>Home | Jobs | Log in</nav>
<main><h1>Data Entry Clerk</h1>
<p>We are hiring a full-time data entry clerk to join our operations team in Pune.</p>
<h2>Responsibilities</h2><ul><li>Enter customer records into our CRM</li><li>Check data for errors</li></ul>
<h2>Requirements</h2><ul><li>Typing speed of 40 words per minute</li><li>Basic Excel skills</li></ul>
<p>Apply with your CV. Shortlisted candidates will be invited to an interview.</p></main>
<footer>(c) Example Ltd</footer></body></html>"""

JSON_LD = {"@context": "https://schema.org", "@type": "JobPosting", "title": "Warehouse Associate",
           "description": "<p>Pick, pack and ship customer orders on the night shift.</p>",
           "hiringOrganization": {"@type": "Organization", "name": "Acme Logistics"},
           "jobLocation": {"@type": "Place", "address": {"addressLocality": "Chennai", "addressCountry": "IN"}},
           "skills": ["Forklift licence", "Inventory software"], "qualifications": "High school diploma",
           "jobBenefits": "Health insurance and paid leave"}
JSON_LD_HTML = ("<html><head><script type=\"application/ld+json\">" + json.dumps({"@graph": [{"@type": "WebPage"},
                JSON_LD]}) + "</script></head><body><p>Loading...</p></body></html>")

LOGIN_HTML = """<html><body><main><h1>Sign in to continue</h1><form><input type="email">
<input type="password"><button>Log in</button></form><p>Create an account to view this job.</p></main></body></html>"""


class FakeSock:
    def __init__(self, ip):
        self.ip = ip

    def getpeername(self):
        return (self.ip, 443)


class FakeRaw:
    def __init__(self, peer):
        self.connection = type("C", (), {"sock": FakeSock(peer)})() if peer else None


class FakeResponse:
    def __init__(self, status=200, body=b"", content_type="text/html; charset=utf-8", location=None, peer=PUBLIC,
                 chunks=None):
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        if location:
            self.headers["Location"] = location
        self.encoding = "utf-8"
        self._chunks = chunks
        self._body = body if isinstance(body, bytes) else body.encode("utf-8")
        self.raw = FakeRaw(peer)
        self.consumed = 0
        self.closed = False

    def iter_content(self, size):
        source = self._chunks if self._chunks is not None else [self._body[i:i + size]
                                                                 for i in range(0, len(self._body), size)]
        for chunk in source:
            self.consumed += len(chunk)
            yield chunk

    def close(self):
        self.closed = True


@pytest.fixture()
def net(monkeypatch):
    """Fake DNS plus a scripted sequence of responses (or exceptions) for url_fetch._open."""
    calls = []
    script = {}

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host not in DNS:
            raise socket.gaierror("no such host")
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
                for ip in DNS[host]]

    def fake_open(session, url, timeout):
        calls.append(url)
        outcome = script.get(url)
        if outcome is None:
            raise AssertionError(f"unexpected fetch of {url}")
        if isinstance(outcome, Exception):
            raise outcome
        assert session.trust_env is False and 0 < timeout <= uf.TIMEOUT_SECONDS
        return outcome

    monkeypatch.setattr(uf.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(uf, "_open", fake_open)
    return script, calls


def code_of(url):
    with pytest.raises(uf.FetchError) as info:
        uf.fetch_job_page(url)
    return info.value.code


# --------------------------------------------------------------------------- pages
def test_a_valid_html_job_page_becomes_a_description(net):
    script, _ = net
    script["https://jobs.example.com/clerk"] = FakeResponse(body=JOB_HTML)
    page = uf.fetch_job_page("https://jobs.example.com/clerk")
    assert page.method == "html" and page.domain == "jobs.example.com"
    assert page.fields["title"] == "Data Entry Clerk"
    assert "hiring a full-time data entry clerk" in page.fields["description"]
    assert "Log in" not in page.fields["description"] and "(c) Example Ltd" not in page.fields["description"]
    assert page.fields["company_profile"] is None  # never guessed


def test_a_json_ld_job_posting_maps_to_the_structured_fields(net):
    script, _ = net
    script["https://careers.example.org/job/42"] = FakeResponse(body=JSON_LD_HTML)
    page = uf.fetch_job_page("https://careers.example.org/job/42")
    assert page.method == "json-ld"
    f = page.fields
    assert f["title"] == "Warehouse Associate" and f["company_profile"] == "Acme Logistics"
    assert f["description"] == "Pick, pack and ship customer orders on the night shift."
    assert "High school diploma" in f["requirements"] and "Forklift licence" in f["requirements"]
    assert f["benefits"] == "Health insurance and paid leave"
    assert page.location == "Chennai, IN"


def test_a_403_page_fails(net):
    script, _ = net
    script["https://jobs.example.com/x"] = FakeResponse(status=403, body="Forbidden")
    assert code_of("https://jobs.example.com/x") == "http_status"


def test_a_login_page_fails(net):
    script, _ = net
    script["https://jobs.example.com/private"] = FakeResponse(body=LOGIN_HTML)
    assert code_of("https://jobs.example.com/private") == "login_required"


def test_a_401_is_a_login_wall(net):
    script, _ = net
    script["https://jobs.example.com/members"] = FakeResponse(status=401)
    assert code_of("https://jobs.example.com/members") == "login_required"


def test_an_empty_page_fails_instead_of_guessing(net):
    script, _ = net
    script["https://jobs.example.com/blank"] = FakeResponse(body="<html><body><div id=app></div></body></html>")
    assert code_of("https://jobs.example.com/blank") == "empty_extraction"


def test_non_html_content_is_refused(net):
    script, _ = net
    script["https://jobs.example.com/job.pdf"] = FakeResponse(body=b"%PDF-1.7", content_type="application/pdf")
    assert code_of("https://jobs.example.com/job.pdf") == "not_html"


# --------------------------------------------------------------------------- address and redirect checks
def test_a_redirect_to_localhost_is_blocked_before_any_request(net):
    script, calls = net
    script["https://jobs.example.com/go"] = FakeResponse(status=302, location="http://127.0.0.1/admin")
    assert code_of("https://jobs.example.com/go") == "blocked_address"
    assert calls == ["https://jobs.example.com/go"]  # 127.0.0.1 was never contacted


@pytest.mark.parametrize("url", ["http://localhost/", "http://127.0.0.1:8080/", "http://internal.example.com/job",
                                 "http://metadata.example.com/latest/meta-data/", "http://v6local.example.com/",
                                 "https://mixed.example.com/job", "http://[::1]/", "http://169.254.169.254/",
                                 "http://docs-range.example.com/"])
def test_private_loopback_link_local_and_metadata_addresses_are_blocked(net, url):
    _, calls = net
    assert code_of(url) == "blocked_address" and calls == []


@pytest.mark.parametrize("url", ["ftp://jobs.example.com/job", "file:///etc/passwd", "javascript:alert(1)",
                                 "gopher://jobs.example.com/"])
def test_non_http_schemes_are_refused(net, url):
    _, calls = net
    assert code_of(url) == "scheme_not_allowed" and calls == []


def test_redirects_are_followed_and_rechecked_up_to_the_limit(net):
    script, calls = net
    script["https://jobs.example.com/a"] = FakeResponse(status=301, location="/b")
    script["https://jobs.example.com/b"] = FakeResponse(status=302, location="https://careers.example.org/c")
    script["https://careers.example.org/c"] = FakeResponse(body=JOB_HTML)
    page = uf.fetch_job_page("https://jobs.example.com/a")
    assert page.domain == "careers.example.org" and len(calls) == 3


def test_more_than_three_redirects_fail(net):
    script, calls = net
    for i in range(5):
        script[f"https://jobs.example.com/r{i}"] = FakeResponse(status=302, location=f"/r{i + 1}")
    assert code_of("https://jobs.example.com/r0") == "too_many_redirects"
    assert len(calls) == uf.MAX_REDIRECTS + 1


def test_the_connected_peer_address_is_checked_again(net):
    script, _ = net
    script["https://jobs.example.com/rebind"] = FakeResponse(body=JOB_HTML, peer="127.0.0.1")
    assert code_of("https://jobs.example.com/rebind") == "blocked_address"


def test_unknown_hosts_fail(net):
    assert code_of("https://no-such-host.example/job") == "dns_failed"


# --------------------------------------------------------------------------- size and time limits
def test_a_huge_response_stops_reading_at_the_limit(net):
    script, _ = net
    head = ("<html><body><main><h1>Night Shift Packer</h1><p>We are hiring packers. Apply with your CV; "
            "interviews are held every Monday.</p>").encode()
    filler = b"<p>" + b"lorem ipsum dolor sit amet " * 2400 + b"</p>"  # ~65 KB per chunk
    response = FakeResponse(chunks=[head] + [filler] * 200)  # about 13 MB offered
    script["https://jobs.example.com/huge"] = response
    page = uf.fetch_job_page("https://jobs.example.com/huge")
    assert response.consumed <= uf.MAX_BYTES + len(filler)  # stopped near 1.5 MB, not 13 MB
    assert page.fields["title"] == "Night Shift Packer"
    assert len(page.fields["description"]) <= uf.FIELD_LIMITS["description"]


def test_a_timeout_fails(net):
    script, _ = net
    script["https://jobs.example.com/slow"] = requests.Timeout("read timed out")
    assert code_of("https://jobs.example.com/slow") == "timeout"


def test_a_slow_body_hits_the_overall_deadline(net, monkeypatch):
    script, _ = net
    clock = iter([0.0, 0.1, 0.2, 99.0, 99.0, 99.0])
    monkeypatch.setattr(uf.time, "monotonic", lambda: next(clock))
    script["https://jobs.example.com/drip"] = FakeResponse(chunks=[b"<html>", b"<body>", b"<p>late</p>"])
    assert code_of("https://jobs.example.com/drip") == "timeout"


def test_a_connection_error_fails(net):
    script, _ = net
    script["https://jobs.example.com/down"] = requests.ConnectionError("refused")
    assert code_of("https://jobs.example.com/down") == "request_failed"


# --------------------------------------------------------------------------- helpers
@pytest.mark.parametrize("text, url", [("https://jobs.example.com/a", "https://jobs.example.com/a"),
                                       ("  http://x.example/job?id=1 \n", "http://x.example/job?id=1"),
                                       ("see https://jobs.example.com/a", None), ("ftp://x.example/", None),
                                       ("https://a.example https://b.example", None), (None, None), (5, None)])
def test_single_url_detects_only_a_lone_http_link(text, url):
    assert uf.single_url(text) == url


@pytest.mark.parametrize("ip, blocked", [("8.8.8.8", False), (PUBLIC, False), ("10.1.2.3", True),
                                         ("172.16.0.1", True), ("192.168.0.1", True), ("127.0.0.1", True),
                                         ("169.254.169.254", True), ("100.64.0.1", True), ("0.0.0.0", True),
                                         ("::1", True), ("fe80::1", True), ("fc00::1", True),
                                         ("::ffff:127.0.0.1", True), ("224.0.0.1", True)])
def test_is_blocked_ip(ip, blocked):
    assert uf.is_blocked_ip(ipaddress.ip_address(ip)) is blocked


# --------------------------------------------------------------------------- route
@pytest.fixture()
def client(monkeypatch):
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
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(job_routes.job_bp)
    return app.test_client(), Spy


@pytest.mark.parametrize("body", [{"text": "https://jobs.example.com/clerk"}, {"url": "https://jobs.example.com/clerk"},
                                  {"text": "  https://jobs.example.com/clerk\n"}])
def test_a_pasted_link_is_fetched_and_analysed_with_its_source(net, client, body):
    script, _ = net
    script["https://jobs.example.com/clerk"] = FakeResponse(body=JOB_HTML)
    c, spy = client
    response = c.post("/api/analyze-job", json=body)
    assert response.status_code == 200
    data = response.get_json()
    assert data["verdict"] == "Likely Legitimate" and data["source"]["domain"] == "jobs.example.com"
    assert spy.seen and spy.seen[0].startswith("Title: Data Entry Clerk\nDescription: We are hiring")
    assert "safe" not in json.dumps(data["source"]).lower()  # no claim about the domain


def test_a_json_ld_link_uses_the_structured_fields_and_the_company(net, client):
    script, _ = net
    script["https://careers.example.org/job/42"] = FakeResponse(body=JSON_LD_HTML)
    c, spy = client
    data = c.post("/api/analyze-job", json={"url": "https://careers.example.org/job/42"}).get_json()
    assert data["company_profile_present"] is True and data["source"]["extraction"] == "json-ld"
    assert spy.seen[0].startswith("Title: Warehouse Associate\nCompany profile: Acme Logistics\nDescription:")


@pytest.mark.parametrize("url, script_value", [
    ("https://jobs.example.com/x", FakeResponse(status=403)),
    ("https://jobs.example.com/x", FakeResponse(body=LOGIN_HTML)),
    ("https://jobs.example.com/x", requests.Timeout()),
    ("http://127.0.0.1/admin", None),
    ("ftp://jobs.example.com/x", None),
])
def test_unreadable_links_return_422_with_the_fixed_message(net, client, caplog, url, script_value):
    script, _ = net
    if script_value is not None:
        script[url] = script_value
    c, spy = client
    with caplog.at_level(logging.DEBUG):
        response = c.post("/api/analyze-job", json={"url": url})
    assert response.status_code == 422
    assert response.get_json()["error"] == "Could not read this page. Paste the job text instead."
    assert spy.seen == [] and "/x" not in caplog.text and "admin" not in caplog.text


@pytest.mark.parametrize("body", [{"url": ""}, {"url": None}, {"url": 42}])
def test_an_empty_or_non_string_url_is_a_400(net, client, body):
    c, _ = client
    assert c.post("/api/analyze-job", json=body).status_code == 400


def test_text_that_merely_contains_a_link_is_analysed_as_text(net, client):
    _, calls = net
    c, spy = client
    text = "Data entry clerk wanted, apply at https://jobs.example.com/clerk with your CV and two references."
    data = c.post("/api/analyze-job", json={"text": text}).get_json()
    assert calls == [] and spy.seen == [f"Description: {text}"] and "source" not in data


def test_a_page_of_digits_goes_through_the_same_insufficient_check(net, client):
    script, _ = net
    script["https://jobs.example.com/tiny"] = FakeResponse(body="<html><body><main><h1>#</h1>"
                                                                "<p>1234 5678 9012 3456 7890 1234 5678 !!!</p>"
                                                                "</main></body></html>")
    c, spy = client
    data = c.post("/api/analyze-job", json={"url": "https://jobs.example.com/tiny"}).get_json()
    assert data["verdict"] == "Insufficient Information" and data["source"]["domain"] == "jobs.example.com"
    assert spy.seen == []
