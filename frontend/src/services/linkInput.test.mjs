import test from "node:test";
import assert from "node:assert/strict";

import { ANALYZING_LABEL, busyLabel, FETCHING_LABEL, singleUrl } from "./linkInput.js";
import {
  buildMainScreen, INSUFFICIENT, isNoScoreVerdict, NO_SCORE_VERDICTS, NOT_A_JOB, plainSentence, sourceLine,
} from "./reportView.js";

test("a borderline relevance note from the server is shown as given, and only when present", () => {
  const base = { verdict: "Likely Legitimate", display_risk_score: 16, warning_signs: [] };
  const note = "This text has few typical job-posting words. Check that you pasted a job advertisement; "
    + "the result below assumes it is one.";
  assert.equal(buildMainScreen({ ...base, relevance_note: note }).relevanceNote, note);
  assert.equal(buildMainScreen(base).relevanceNote, "");
  assert.equal(buildMainScreen({ ...base, relevance_note: 42 }).relevanceNote, "");
  assert.equal(buildMainScreen({ ...base, relevance_note: note }).score, buildMainScreen(base).score);
});

test("a lone http or https link is a link; anything else is text", () => {
  assert.equal(singleUrl("https://jobs.example.com/a"), "https://jobs.example.com/a");
  assert.equal(singleUrl("  http://x.example/job?id=1 \n"), "http://x.example/job?id=1");
  for (const text of ["see https://jobs.example.com/a", "ftp://x.example/", "https://a.example https://b.example",
    "jobs.example.com", "", null, 5]) {
    assert.equal(singleUrl(text), null, String(text));
  }
});

test("a pasted link shows Fetching page while waiting, text shows Analyzing", () => {
  assert.equal(busyLabel("https://jobs.example.com/a"), FETCHING_LABEL);
  assert.equal(FETCHING_LABEL, "Fetching page");
  assert.equal(busyLabel("Data entry clerk wanted, apply with your CV."), ANALYZING_LABEL);
});

test("Not a Job Posting sits next to Insufficient Information as a no-score verdict", () => {
  assert.deepEqual(NO_SCORE_VERDICTS, [INSUFFICIENT, NOT_A_JOB]);
  assert.equal(NOT_A_JOB, "Not a Job Posting");
  assert.ok(isNoScoreVerdict({ verdict: NOT_A_JOB }) && isNoScoreVerdict({ verdict: INSUFFICIENT }));
  assert.ok(!isNoScoreVerdict({ verdict: "Likely Legitimate" }) && !isNoScoreVerdict(null));
  assert.equal(plainSentence({ verdict: NOT_A_JOB, warning_signs: [] }), "");
});

test("the source line names only the domain and makes no safety claim", () => {
  assert.equal(sourceLine({ source: { domain: "jobs.example.com", extraction: "html" } }), "Source: jobs.example.com");
  assert.equal(sourceLine({ verdict: "Likely Legitimate" }), "");
  assert.equal(sourceLine({ source: { domain: "" } }), "");
  assert.ok(!/safe|trusted|verified/i.test(sourceLine({ source: { domain: "jobs.example.com" } })));
});
