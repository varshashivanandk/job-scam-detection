# Evidence-Grounded Detection of Fraudulent Job Postings and Recruiter Conversations (Job Scam Detector)

**An AI system that checks job postings and recruiter chats for scam signs, and explains its answer with evidence a person can verify.**

Fake job offers cost people money and personal data. This project takes a pasted job posting, a job link, or a recruiter chat (WhatsApp, Telegram, plain text) and returns a risk score, a verdict, the warning signs it found, and what to do next. Every reason points to numbered evidence: similar known postings, scam patterns, and official guidance from bodies such as the FTC, Scamwatch and India's Ministry of Home Affairs.

> The score is a risk indicator, not a probability. The tool is a cautious first check. It does not clear or condemn an offer.

## What it does

- **Scores postings and chats** on a 1 to 99 scale with a three-level verdict: Likely Legitimate, Needs Verification, High Fraud Risk.
- **Explains itself.** A language model writes the justification, but it may only cite evidence the system gives it. A checker rejects explanations that cite missing evidence or contradict the verdict, and a template takes over when the model is unavailable.
- **Never lets the language model decide.** The classifier and rules set the verdict. A zero-shot LLM judge we tested flagged about 59% of legitimate postings in a validation sample, which is why it only explains.
- **Handles real input.** Pasted text, structured fields, job links, and several chat export formats. Non-job text and too-short input get a clear message instead of a made-up score.
- **Gives next steps** and a downloadable PDF report.

## How it works

```
posting / link / chat
        |
  clean and validate
        |
        +--> DistilBERT classifier (calibrated)   -> score and verdict
        +--> scam-pattern rules                   -> warning signs, can raise the verdict
        +--> retrieval and entity graph           -> similar known postings, linked companies, official guidance
                      |
              numbered evidence items
                      |
        language model writes a cited justification (checked, with template fallback)
                      |
                    report
```

## Tech stack

| Area | Tools |
|---|---|
| Machine learning | Python, PyTorch, Hugging Face Transformers (DistilBERT), scikit-learn (TF-IDF + logistic regression baseline) |
| Retrieval and graph | Text retrieval over known postings and a guidance library, NetworkX entity graph |
| Explanations | Groq API (open-weight `gpt-oss` models), prompt-injection-aware prompting, output checker |
| Backend | Flask, REST API |
| Frontend | React 18, Vite |
| Quality | pytest and Node test suites, SHA-256-verified data splits, bootstrap confidence intervals |

## Results

Dataset: EMSCAD, 17,880 postings (866 fraudulent). Held-out test set: 2,695 postings, 144 fraudulent. The split is group-aware, so near-duplicate postings never sit on both sides of the split.

| Model | Test PR-AUC | Test F1 | False alarms |
|---|---|---|---|
| TF-IDF + logistic regression | 0.802 | 0.742 | 44 |
| DistilBERT (deployed model, one run) | 0.855 | 0.764 | 34 |
| DistilBERT (average of 3 retrained seeds) | 0.833 | 0.737 | 27 |

**What this shows, honestly:**
- DistilBERT looks better in the deployed run, but across three retrained seeds the accuracy gap over the simple baseline is within noise. Its consistent advantage is fewer false alarms.
- A fake company profile pasted into a scam posting can hide it from the classifier (in the deployed model, none of the tested postings stayed flagged). Retraining with swapped profiles kept most of them flagged but cost accuracy, so it was not deployed.
- A RoBERTa run diverged during training and is reported as a failed run, not a baseline.

## Engineering practices

- **No test-set leakage.** One shared split, seed 42, hash-checked before every experiment. The test set is locked behind a guarded script and was scored only once per planned configuration.
- **Uncertainty reported.** Group bootstrap confidence intervals, multi-seed runs, and negative results kept in the write-up.
- **Defensive LLM use.** Untrusted posting text is delimited and sanitised, output is validated against the evidence, and rate limits fall back to a template instead of failing.
- **Tested.** Hundreds of automated backend and frontend tests cover parsing, rules, verdict logic, evidence, input validation and rendering.

## Run it locally

Needs Python 3.10+, Node.js, the model files and data splits (see [Files that are not in Git](#files-that-are-not-in-git)), and optionally a Groq API key for the written explanations. Without a key the app still works and shows a template explanation.

```powershell
git clone https://github.com/varshashivanandk/job-scam-detection
cd job-scam-detection
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend; npm install; cd ..
```

Create a `.env` file (copy `.env.example`) with `GROQ_API_KEY` and `LLM_MODEL`, then use two terminals:

```powershell
python -m backend.app          # backend on http://127.0.0.1:5000
cd frontend; npm run dev       # frontend on http://localhost:5173
```

Run the tests with `python -m pytest backend/tests -q` and, in `frontend/`, `npm test`.

## Project layout

```
backend/     Flask API, classifier, rules, verdict logic, retrieval, entity graph, explanation layer, tests
frontend/    React + Vite web app
scripts/     data preparation, training baseline, validation experiments, guarded final test evaluation
data/        small synthetic test sets (the real dataset is not redistributed)
docs/        model and data hashes
reports/     data-quality and evaluation reports
```

## Files that are not in Git

Model weights, the EMSCAD dataset and its splits, and secrets are not committed because of size, licence or secrecy. `docs/model_hashes.md` lists SHA-256 hashes so a copy can be checked. The dataset must be obtained from its source, and its licence terms read first.

## Team

Varsha Shivanand K, Venessa Chrisy S, Yashwitha T S, Veena Shanmukh.