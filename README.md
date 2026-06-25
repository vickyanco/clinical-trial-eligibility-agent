# Clinical Trial Eligibility Agent

An agentic AI system that assesses whether a patient is eligible for a
clinical trial, given the trial's protocol criteria (free text) and a
structured patient record. Built with **LangGraph**, **Pydantic**, and
**FastAPI**, with a built-in **evaluation harness** that treats every
agent output as a hypothesis to be checked, not a result to be trusted.

## Why this project

Clinical research workflows depend on accurately matching patients to
trial criteria — a task that is repetitive, error-prone when done
manually at scale, and genuinely hard for AI systems to get right
reliably. This project is a small, self-contained demonstration of how
I think about building AI systems for exactly this kind of
high-stakes, regulated context:

- **Structured outputs over free text.** Every step of the agent
  produces a Pydantic-validated object, not raw LLM text. If the model
  returns something malformed, the system fails loudly instead of
  silently passing bad data downstream.
- **Confidence-aware, human-in-the-loop by design.** The agent doesn't
  just say "eligible" or "not eligible" — it returns a confidence score
  and a `flagged_for_review` flag. Low-confidence verdicts are designed
  to be reviewed by a human, never acted on blindly.
- **Evaluation as a first-class concern.** The `eval/` folder is not an
  afterthought — it's a small, deliberately adversarial test set
  (including boundary cases like "patient is exactly 18" or "eGFR is
  1 mL/min above the exclusion cutoff") that measures whether the
  system is actually correct, not just whether it runs.
- **Zero-cost, reproducible by default.** The project runs end-to-end
  with no API key and no cost, using a deterministic mock in place of
  a real LLM call. Provide a free Gemini API key and the exact same
  code path calls a real model instead — the architecture doesn't
  change, only the backing model does.

## Architecture

```
Protocol text + Patient record
            |
            v
   ┌─────────────────┐
   │ extract_criteria│   parses protocol into structured
   └────────┬────────┘   inclusion/exclusion criteria
            v
   ┌─────────────────┐
   │ assess_criteria │   checks each criterion against the
   └────────┬────────┘   patient record, with evidence + confidence
            v
   ┌─────────────────┐
   │aggregate_verdict│   combines assessments into a final,
   └────────┬────────┘   structured EligibilityVerdict
            v
   Eligible? Confidence? Flagged for review?
```

The graph is built with **LangGraph**, with each node taking and
returning a typed `AgentState`. All structured contracts (criteria,
patient records, assessments, final verdict) are defined in
`src/schemas.py` using **Pydantic**.

## Project structure

```
clinical-trial-eligibility-agent/
├── src/
│   ├── schemas.py     # Pydantic models: the structured contracts
│   ├── agent.py        # LangGraph agent: extraction, assessment, aggregation
│   └── api.py           # FastAPI service exposing the agent
├── eval/
│   ├── test_cases.json  # Hand-crafted test patients, incl. edge cases
│   └── evaluate.py        # Evaluation harness: accuracy + review-flag rate
├── data/
│   └── sample_protocol.txt
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone <this-repo>
cd clinical-trial-eligibility-agent
pip install -r requirements.txt
```

### Run the evaluation (no setup required)

```bash
cd eval
python evaluate.py
```

This runs the agent against 6 test patients (including 2 boundary
edge cases) using the built-in mock LLM, and prints an accuracy report.

### Run the API

```bash
cd src
uvicorn api:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### Use a real LLM (optional, free tier)

This project uses [Google Gemini](https://ai.google.dev/) for real LLM
calls, since it offers a genuinely free tier with no credit card
required.

```bash
export GEMINI_API_KEY="your-key-here"
```

With the key set, `agent.py` automatically routes calls to Gemini
instead of the mock — no other code changes needed.

## Example

**Request:**
```json
{
  "study_id": "STUDY-001",
  "protocol_text": "Inclusion: Age >= 18, Diagnosis of Type 2 Diabetes. Exclusion: eGFR < 30 mL/min.",
  "patient": {
    "patient_id": "P001",
    "age": 45,
    "sex": "female",
    "diagnosis": ["Type 2 Diabetes"],
    "lab_values": {"eGFR": 65.0}
  }
}
```

**Response:**
```json
{
  "patient_id": "P001",
  "study_id": "STUDY-001",
  "eligible": true,
  "overall_confidence": 0.6,
  "reasons": ["All inclusion criteria met and no exclusion criteria triggered."],
  "flagged_for_review": true
}
```

## Honest limitations

This is a portfolio-scale demonstration, not a production system. In
particular:

- The mock LLM uses simple keyword heuristics, not real reasoning —
  it exists purely to make the pipeline runnable and testable without
  cost. Real protocol text is far messier than the examples here.
- Confidence scoring is illustrative; a production system would need
  a properly calibrated, validated confidence model.
- There is no de-identification, audit logging, or access control —
  all essential for any real clinical data pipeline, intentionally
  out of scope here to keep the project focused on the agentic
  reasoning and evaluation design.

## Tech stack

Python · LangGraph · Pydantic · FastAPI · Google Gemini API