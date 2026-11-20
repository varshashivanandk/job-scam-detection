"""Fetch a job page from a pasted link and extract its text, safely. Never guesses content.

Safety: http and https only; every host is resolved first and every resolved address must be public (no
localhost, private, link-local, loopback, metadata or other reserved ranges); the address actually connected
to is checked again; redirects are followed by hand (at most MAX_REDIRECTS), each one re-checked; one overall
deadline of TIMEOUT_SECONDS; reading stops after MAX_BYTES; only text/html is accepted; proxy settings from
the environment are ignored so the address check sees the real server. No reputation service is called.

Extraction: a schema.org JobPosting in JSON-LD is mapped to the posting fields; otherwise the page's main
text (BeautifulSoup) becomes the description. Any failure raises FetchError with a short code.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from backend.services.dataset import clean_field

ALLOWED_SCHEMES = ("http", "https")
TIMEOUT_SECONDS = 5.0
MAX_BYTES = 1_500_000
MAX_REDIRECTS = 3
USER_AGENT = "JobScamAssistant/1.0 (checks a job posting for warning signs)"
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
MIN_EXTRACTED_CHARS = 30
FIELD_LIMITS = {"title": 300, "company_profile": 3000, "description": 12000, "requirements": 3000, "benefits": 1500}
READ_CHUNK = 64 * 1024

_URL_ONLY_RE = re.compile(r"^\s*(https?://\S+)\s*$", re.IGNORECASE)


class FetchError(Exception):
    """A short, fixed code (never page content or provider text)."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class FetchedPage:
    fields: dict
    domain: str
    method: str                      # "json-ld" or "html"
    location: str | None = None      # from JSON-LD jobLocation; not part of the model text
    notes: list = field(default_factory=list)


def single_url(text) -> str | None:
    """The URL when the whole pasted text is one http(s) link (whitespace allowed around it)."""
    if not isinstance(text, str):
        return None
    match = _URL_ONLY_RE.match(text)
    return match.group(1) if match else None


# --------------------------------------------------------------------------- address checks
def is_blocked_ip(ip) -> bool:
    """True for anything that is not a public unicast address (private, loopback, link-local, metadata, ...)."""
    ip = ipaddress.ip_address(ip)
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    return (not ip.is_global) or ip.is_multicast or ip.is_reserved or ip.is_unspecified


def resolve(host: str, port: int) -> list:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        raise FetchError("dns_failed") from None
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise FetchError("dns_failed")
    return [ipaddress.ip_address(a.split("%", 1)[0]) for a in addresses]


def check_url(url: str):
    """Validate scheme and host, resolve the host and refuse any non-public address. Returns the split URL."""
    if not isinstance(url, str) or not url.strip() or len(url) > 2048:
        raise FetchError("invalid_url")
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise FetchError("scheme_not_allowed")
    host = parts.hostname
    if not host:
        raise FetchError("invalid_url")
    if parts.username or parts.password:
        raise FetchError("invalid_url")
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError:
        raise FetchError("invalid_url") from None
    addresses = resolve(host, port)
    if any(is_blocked_ip(a) for a in addresses):
        raise FetchError("blocked_address")
    return parts, addresses


def _peer_ip(response):
    """Address the connection really reached, when the transport exposes it."""
    try:
        return ipaddress.ip_address(response.raw.connection.sock.getpeername()[0])
    except Exception:
        return None


# --------------------------------------------------------------------------- transport
def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # ignore proxy variables: the address check must see the real server
    session.max_redirects = 0
    return session


def _open(session, url: str, timeout: float):
    """One GET without following redirects; streamed so the body can be capped."""
    return session.get(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"}, stream=True,
                       allow_redirects=False, timeout=timeout)


def _read_capped(response, deadline: float) -> bytes:
    body = bytearray()
    for chunk in response.iter_content(READ_CHUNK):
        if time.monotonic() > deadline:
            raise FetchError("timeout")
        if chunk:
            body += chunk
        if len(body) >= MAX_BYTES:
            return bytes(body[:MAX_BYTES])  # stop reading; the rest of the page is never downloaded
    return bytes(body)


def fetch_html(url: str) -> tuple[str, str]:
    """(html, final_url). Follows at most MAX_REDIRECTS redirects, checking every hop."""
    deadline = time.monotonic() + TIMEOUT_SECONDS
    session = _session()
    current = url
    try:
        for hop in range(MAX_REDIRECTS + 1):
            check_url(current)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FetchError("timeout")
            try:
                response = _open(session, current, remaining)
            except requests.Timeout:
                raise FetchError("timeout") from None
            except requests.RequestException:
                raise FetchError("request_failed") from None
            try:
                # DNS may change between the check and the connection (rebinding): re-check the real peer.
                peer = _peer_ip(response)
                if peer is not None and is_blocked_ip(peer):
                    raise FetchError("blocked_address")
                if response.status_code in REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if not location:
                        raise FetchError("http_status")
                    if hop == MAX_REDIRECTS:
                        raise FetchError("too_many_redirects")
                    current = urljoin(current, location)
                    continue
                if response.status_code in (401, 407):
                    raise FetchError("login_required")
                if response.status_code != 200:
                    raise FetchError("http_status")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type != "text/html":
                    raise FetchError("not_html")
                try:
                    raw = _read_capped(response, deadline)
                except requests.RequestException:
                    raise FetchError("timeout") from None
                encoding = response.encoding or "utf-8"
                try:
                    return raw.decode(encoding, errors="replace"), current
                except LookupError:
                    return raw.decode("utf-8", errors="replace"), current
            finally:
                response.close()
        raise FetchError("too_many_redirects")
    finally:
        session.close()


# --------------------------------------------------------------------------- extraction
def _types(obj) -> list[str]:
    t = obj.get("@type") if isinstance(obj, dict) else None
    return [str(x) for x in (t if isinstance(t, list) else [t]) if x]


def _walk_json_ld(data):
    if isinstance(data, list):
        for item in data:
            yield from _walk_json_ld(item)
    elif isinstance(data, dict):
        yield data
        for key in ("@graph", "mainEntity", "itemListElement"):
            if key in data:
                yield from _walk_json_ld(data[key])


def find_job_posting(soup) -> dict | None:
    for script in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        try:
            data = json.loads(script.string or script.get_text() or "")
        except (ValueError, TypeError):
            continue
        for obj in _walk_json_ld(data):
            if "JobPosting" in _types(obj):
                return obj
    return None


def _flatten(value) -> list[str]:
    """Strings from a JSON-LD value that may be a string, a list or an object (e.g. a credential)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        return [s for v in value for s in _flatten(v)]
    if isinstance(value, dict):
        out = []
        for key in ("name", "description", "credentialCategory", "skills", "value"):
            out += _flatten(value.get(key))
        return out
    return []


def _location(job: dict) -> str | None:
    parts = []
    for place in (job.get("jobLocation") if isinstance(job.get("jobLocation"), list) else [job.get("jobLocation")]):
        address = place.get("address") if isinstance(place, dict) else None
        if isinstance(address, dict):
            bits = [address.get(k) for k in ("addressLocality", "addressRegion", "addressCountry")]
            bits = [b.get("name") if isinstance(b, dict) else b for b in bits]
            parts.append(", ".join(str(b) for b in bits if b))
        elif isinstance(address, str):
            parts.append(address)
    if str(job.get("jobLocationType", "")).upper() == "TELECOMMUTE":
        parts.append("Remote")
    text = "; ".join(p for p in parts if p)
    return text or None


def _cap(fields: dict) -> dict:
    return {k: (v[:FIELD_LIMITS[k]] if isinstance(v, str) else v) for k, v in fields.items()}


def from_job_posting(job: dict) -> dict:
    """Map a schema.org JobPosting to the five posting fields. Missing values stay None."""
    org = job.get("hiringOrganization")
    if isinstance(org, dict):
        company = clean_field(org.get("description")) or clean_field(org.get("name"))
    else:
        company = clean_field(org) if isinstance(org, str) else None
    requirements = "\n".join(clean_field(s) or "" for key in ("qualifications", "skills", "experienceRequirements",
                                                              "educationRequirements")
                             for s in _flatten(job.get(key))).strip()
    benefits = "\n".join(clean_field(s) or "" for s in _flatten(job.get("jobBenefits"))).strip()
    return _cap({
        "title": clean_field(job.get("title") or job.get("name")),
        "company_profile": company,
        "description": clean_field(job.get("description")),
        "requirements": requirements or None,
        "benefits": benefits or None,
    })


_DROP_TAGS = ("script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside", "form",
              "iframe", "button")
_LOGIN_RE = re.compile(r"\b(?:sign|log)\s?in\b|\blogin\b|\bcreate (?:an )?account\b", re.IGNORECASE)


def from_html(soup) -> dict:
    """Main visible text of the page as the description, the first h1 as the title."""
    title_tag = soup.find("h1")
    title = clean_field(title_tag.get_text(" ")) if title_tag else None
    if title_tag:
        title_tag.decompose()  # the title is its own field, not repeated at the top of the description
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or soup.body or soup
    lines = [re.sub(r"\s+", " ", line).strip() for line in root.get_text("\n").splitlines()]
    text = "\n".join(line for line in lines if line)
    return _cap({"title": title, "company_profile": None, "description": text or None,
                 "requirements": None, "benefits": None})


def extract(html: str) -> tuple[dict, str, str | None]:
    """(fields, method, location). Raises FetchError for a login wall or an empty page."""
    soup = BeautifulSoup(html, "html.parser")
    has_password = soup.find("input", attrs={"type": re.compile(r"^password$", re.I)}) is not None
    job = find_job_posting(soup)
    if job is not None:
        fields, method, location = from_job_posting(job), "json-ld", _location(job)
    else:
        fields, method, location = from_html(soup), "html", None
    body = " ".join(fields[k] for k in ("description", "requirements", "benefits") if fields.get(k))
    # A short page with a password box or sign-in wording is a login wall, not a posting.
    if method == "html" and len(body.split()) < 150 and (has_password or _LOGIN_RE.search(body)):
        raise FetchError("login_required")
    if len(body) < MIN_EXTRACTED_CHARS:
        raise FetchError("empty_extraction")
    return fields, method, location


def fetch_job_page(url: str) -> FetchedPage:
    """Fetch and extract a job page. Raises FetchError(code) on any failure; never invents content."""
    html, final_url = fetch_html(url)
    fields, method, location = extract(html)
    domain = (urlsplit(final_url).hostname or "").lower()
    return FetchedPage(fields=fields, domain=domain, method=method, location=location)
