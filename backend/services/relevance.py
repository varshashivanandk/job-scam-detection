"""Is this text a job posting at all? A cheap gate that runs BEFORE the classifier.

Two signals, no training:
  a) keywords and structure: job terms, section-like headings, minus news cues (bylines, datelines,
     "said on Monday", quotes attributed to people);
  b) semantic similarity with the existing all-MiniLM-L6-v2 embedder: the text's best match among
     short job-posting prototypes minus its best match among non-job prototypes.

Fixed rule (constants below): a text is rejected only when BOTH signals say "not a job posting".
Anything else, including text too short to judge or a missing embedder, is let through to the
classifier: rejecting a real posting is worse than letting a news item through.
Thresholds were chosen with scripts/calibrate_relevance.py on EMSCAD train and validation postings
(all must pass) and backend/tests/fixtures/non_job_samples.json. Test data was not used.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable

import numpy as np

# --------------------------------------------------------------------------- fixed rule (calibrated, see module doc)
MIN_WORDS_TO_JUDGE = 20        # shorter texts are not judged (too little signal); they go to the classifier
KEYWORD_REJECT_AT_MOST = 2.0   # keyword score at or below this counts as "no job structure"
# Semantic margin below this counts as "reads like non-job text". The lowest margin of any EMSCAD train or
# validation posting that reaches the semantic check was -0.134 (a description pasted on its own), so -0.15
# lets every one of them through (reports/relevance_calibration.json).
SEMANTIC_REJECT_BELOW = -0.15
# Stricter cut-off when the text carries a news cue. No such posting has a wire tag, "said on <day>" or an
# attributed quote; the 20 with "said in a statement"/"told reporters" all have margins of +0.17 or more.
# The dateline cue is not used here: it also matches all-caps headings in about 300 real postings.
NEWS_SEMANTIC_REJECT_BELOW = 0.10
STRICT_NEWS_CUES = ("wire", "said_on", "attributed_quote", "statement")
# Owner decision (2026-11-19): borderline text is never rejected, but text with few job words whose margin is
# below this gets a note in the report asking the user to check they pasted a job advertisement. At +0.05 the
# note reaches all six non-job samples the gate lets through and 152 of 30,370 train/validation posting
# variants (0.5%, mostly descriptions pasted alone); at 0.00 it would be 38 postings, with less room to spare.
NOTE_BELOW = 0.05
RELEVANCE_NOTE = ("This text has few typical job-posting words. Check that you pasted a job advertisement; "
                  "the result below assumes it is one.")
CHUNK_CHARS = 800              # the embedder reads about 256 tokens; longer texts are judged in windows
MAX_CHUNKS = 3

# --------------------------------------------------------------------------- signal (a): keywords and structure
_JOB_TERMS = {
    "apply": r"\bappl(?:y|ying|ication|icants?)\b",
    "responsibilities": r"\bresponsibilit(?:y|ies)\b|\bduties\b",
    "requirements": r"\brequirements?\b|\brequired\b",
    "qualifications": r"\bqualifications?\b|\bqualified\b",
    "salary": r"\bsalary\b|\bcompensation\b|\bper (?:hour|annum|year|month)\b|\bpay(?:ing)? (?:range|rate)\b|\bctc\b",
    "experience": r"\bexperience[ds]?\b",
    "position": r"\bpositions?\b|\bvacanc(?:y|ies)\b|\bopenings?\b",
    "role": r"\broles?\b",
    "hiring": r"\bhiring\b|\bwe are looking for\b|\bjoin (?:our|the) team\b|\brecruit(?:ing|ment)?\b",
    "candidate": r"\bcandidates?\b|\bapplicants?\b",
    "employment_type": r"\bfull[- ]time\b|\bpart[- ]time\b|\bcontract(?:or)?\b|\binternship\b|\btemporary\b|\bpermanent\b",
    "remote": r"\bremote(?:ly)?\b|\bwork from home\b|\bhybrid\b|\bon[- ]site\b",
    "benefits": r"\bbenefits?\b|\bperks?\b|\bpaid (?:time off|leave)\b|\binsurance\b",
    "resume": r"\bresum[eé]s?\b|\bcv\b|\bcurriculum vitae\b|\bcover letter\b",
    "interview": r"\binterviews?\b",
    "skills": r"\bskills?\b|\bproficien(?:t|cy)\b|\bknowledge of\b",
    "job": r"\bjobs?\b|\bemployment\b|\bemployer\b|\bwork(?:ing)? with\b",
    "degree": r"\bdegree\b|\bbachelor'?s?\b|\bmaster'?s?\b|\bdiploma\b|\bgraduate\b",
    "team": r"\bteam\b|\breport(?:s|ing)? to\b",
}
_JOB_RES = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in _JOB_TERMS.items()}

_HEADING_RE = re.compile(
    r"(?:^|\n|\.\s|:\s)\s*(?:key\s+)?(?:responsibilities|requirements|qualifications|benefits|duties|perks|"
    r"skills(?: required)?|job description|job summary|about (?:the )?(?:role|job|position|company|us)|"
    r"what you(?:'ll| will) do|who you are|what we offer|what we(?:'re| are) looking for|how to apply|"
    r"compensation|essential functions|minimum qualifications|preferred qualifications)\s*[:\-\n]",
    re.IGNORECASE,
)

_NEWS_CUES = {
    "byline": re.compile(r"^\s*by\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s*[.,\n]", re.MULTILINE),
    "wire": re.compile(r"\((?:Reuters|AP|AFP|PTI|IANS|ANI|Bloomberg)\)\s*[-–—]"),
    "dateline": re.compile(r"^\s*[A-Z][A-Z .]{2,}(?:,\s*[A-Z][a-z]+)?\s*(?:\(|[-–—]|:)", re.MULTILINE),
    "said_on": re.compile(r"\bsaid (?:on|late on|early on) (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
                          re.IGNORECASE),
    "statement": re.compile(r"\bsaid in a statement\b|\btold reporters\b|\bat a press conference\b|\bspokes(?:man|woman|person)\b",
                            re.IGNORECASE),
    "attributed_quote": re.compile(r"[\"”]\s*,?\s*(?:he|she|they|the [a-z]+(?: [a-z]+)?|[A-Z][a-z]+(?: [A-Z][a-z]+)?) "
                                   r"(?:said|says|told|added|stated)\b"),
}

KEYWORD_TERM_CAP = 10       # each distinct job term counts once, at most this many
HEADING_WEIGHT = 1.5
NEWS_CUE_WEIGHT = 2.0


def keyword_signal(text: str) -> dict:
    """Distinct job terms, distinct section headings and distinct news cues found in the text."""
    terms = sorted(name for name, pattern in _JOB_RES.items() if pattern.search(text))
    headings = {m.group(0).strip(" .:-\n").lower() for m in _HEADING_RE.finditer(text)}
    news = sorted(name for name, pattern in _NEWS_CUES.items() if pattern.search(text))
    score = min(len(terms), KEYWORD_TERM_CAP) + HEADING_WEIGHT * len(headings) - NEWS_CUE_WEIGHT * len(news)
    return {"score": float(score), "terms": terms, "headings": len(headings), "news_cues": news}


# --------------------------------------------------------------------------- signal (b): semantic prototypes
JOB_PROTOTYPES = [
    "We are hiring a software engineer to join our team. Responsibilities include writing code and reviewing pull requests.",
    "Job description: customer service representative, full-time, handling calls and emails from customers.",
    "Requirements: bachelor's degree, three years of experience and strong communication skills. Apply with your resume.",
    "Position: registered nurse. Benefits include health insurance, paid time off and a retirement plan.",
    "Remote data entry assistant wanted. Flexible hours, weekly pay, no experience needed. Send your CV to apply.",
    "Our company is looking for a sales manager to lead a regional team and meet quarterly targets.",
    "Qualifications: proficiency in Excel, attention to detail, ability to work independently. Salary negotiable.",
    "Internship opportunity for graduates in marketing. You will assist with campaigns and social media content.",
    "Warehouse associate needed for night shifts. Duties include picking, packing and loading orders.",
    "About the role: you will design, build and maintain cloud infrastructure. What we offer: competitive compensation.",
]
NON_JOB_PROTOTYPES = [
    "The minister said on Tuesday that the government would announce new measures, reporters were told.",
    "The match ended in a draw after a late goal in stoppage time, and the team moved up the league table.",
    "Add two cups of flour and a pinch of salt, then stir and bake for twenty minutes until golden.",
    "I bought this phone last month. The battery life is great but the camera is disappointing. Three stars.",
    "The moon rose softly over the silent hills, and the river sang a song of distant rain.",
    "def main(): for item in items: print(item) return 0 import os import sys",
    "Tomorrow will be cloudy with light rain in the afternoon and a high of 22 degrees.",
    "Hey, are you free this weekend? Let's grab coffee and catch up, call me later.",
    "Live updates: votes are being counted in the state election; see who is leading, who won the seat and by "
    "what margin.",
    "In this lecture we explain the algorithm step by step: define the set, apply the rule to each element and "
    "repeat until nothing changes.",
]

_lock = threading.Lock()
_prototype_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def default_embedder() -> Callable[[list[str]], np.ndarray]:
    """The guidance embedder (all-MiniLM-L6-v2), local files only; normalised vectors."""
    from backend.services.semantic_retrieval import _get_model

    model = _get_model(local_files_only=True)
    return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True))


def _prototypes(embed) -> tuple[np.ndarray, np.ndarray]:
    with _lock:
        key = id(embed)
        if key not in _prototype_cache:
            _prototype_cache[key] = (embed(JOB_PROTOTYPES), embed(NON_JOB_PROTOTYPES))
        return _prototype_cache[key]


def chunks(text: str) -> list[str]:
    """Up to MAX_CHUNKS windows of CHUNK_CHARS, cut at whitespace."""
    out, rest = [], text.strip()
    while rest and len(out) < MAX_CHUNKS:
        if len(rest) <= CHUNK_CHARS:
            out.append(rest)
            break
        cut = rest.rfind(" ", 0, CHUNK_CHARS)
        cut = cut if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
        out.append(rest[:cut])
        rest = rest[cut:].strip()
    return out


def semantic_signal(text: str, embed) -> float:
    """Best window's (max similarity to a job prototype - max similarity to a non-job prototype)."""
    job, non_job = _prototypes(embed)
    vectors = embed(chunks(text))
    margins = (vectors @ job.T).max(axis=1) - (vectors @ non_job.T).max(axis=1)
    return float(margins.max())


# --------------------------------------------------------------------------- the gate
GATE_FIELDS = ("title", "description", "requirements", "benefits", "company_profile")


def gate_text(fields) -> str:
    """The text the gate reads: cleaned posting fields, job content first, without the training labels."""
    from backend.services.dataset import clean_field

    return "\n".join(v for v in (clean_field(fields.get(f)) for f in GATE_FIELDS) if v)


def is_job_posting(text: str, embed=None) -> tuple[bool, float | None, list[str]]:
    """
    (looks_like_a_job_posting, semantic_margin_or_None, reasons). Never raises for a string input.

    Rejects only when the keyword score is at most KEYWORD_REJECT_AT_MOST AND the semantic margin is below
    SEMANTIC_REJECT_BELOW. Short text, or an unavailable embedder, is let through.
    """
    text = text if isinstance(text, str) else ""
    words = len(text.split())
    if words < MIN_WORDS_TO_JUDGE:
        return True, None, [f"Too short to judge ({words} words); passed to the classifier."]

    kw = keyword_signal(text)
    reasons = [f"keyword score {kw['score']:.1f} (job terms {len(kw['terms'])}, headings {kw['headings']}, "
               f"news cues {len(kw['news_cues'])})"]
    if kw["score"] > KEYWORD_REJECT_AT_MOST:
        return True, None, reasons + ["Job terms or structure found."]

    try:
        embed = embed or default_embedder()
        margin = semantic_signal(text, embed)
    except Exception:  # missing model files or any embedding failure: let it through
        return True, None, reasons + ["Semantic check unavailable; passed to the classifier."]

    reasons.append(f"semantic margin {margin:+.3f}")
    news = [cue for cue in kw["news_cues"] if cue in STRICT_NEWS_CUES]
    cutoff = NEWS_SEMANTIC_REJECT_BELOW if news else SEMANTIC_REJECT_BELOW
    if margin < cutoff:
        why = "reads like a news report" if news else "reads like non-job content"
        return False, margin, reasons + [f"Few job terms and the text {why}."]
    return True, margin, reasons + ["Borderline or job-like; passed to the classifier."]


def needs_note(looks_like_job: bool, score: float | None) -> bool:
    """True when the gate let the text through on a borderline semantic margin (few job words, score computed)."""
    return bool(looks_like_job) and score is not None and score < NOTE_BELOW
