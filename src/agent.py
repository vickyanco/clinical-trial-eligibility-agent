"""
Clinical Trial Eligibility Agent — built with LangGraph.

The agent runs a multi-step workflow:
  1. extract_criteria   -> parse protocol text into structured criteria
  2. assess_criteria    -> compare patient record against each criterion
  3. aggregate_verdict  -> combine assessments into a final eligibility verdict

Design choice: every LLM call is wrapped behind `llm_call()`, which uses
Google Gemini's free tier if GEMINI_API_KEY is set in the environment,
and falls back to a deterministic mock otherwise. This means the project
runs end-to-end with zero cost and zero setup, while still being wired
for a real LLM the moment a key is provided -- the architecture doesn't
change, only the backing model does.
"""

import os
import json
from typing import TypedDict

from langgraph.graph import StateGraph, END

from schemas import (
    ProtocolCriteria,
    EligibilityCriterion,
    PatientRecord,
    CriterionAssessment,
    EligibilityVerdict,
)

CONFIDENCE_THRESHOLD = 0.7  # below this, flag for human review


# ---------------------------------------------------------------------------
# LLM call layer (Gemini if available, deterministic mock otherwise)
# ---------------------------------------------------------------------------

def llm_call(prompt: str) -> str:
    """
    Single entry point for all model calls in the agent.

    Returns raw text. Callers are responsible for parsing/validating the
    response against the expected schema -- we never trust model output
    as-is, per the project's evaluation philosophy (see eval/evaluate.py).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return _call_gemini(prompt, api_key)
    return _mock_llm(prompt)


def _call_gemini(prompt: str, api_key: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def _mock_llm(prompt: str) -> str:
    """
    Deterministic stand-in for an LLM, used when no API key is configured.

    This is intentionally simple rule-based logic, not a real language
    model -- it exists so the full pipeline (graph, schemas, evaluation)
    can be exercised and tested without any external dependency or cost.
    It is clearly separated from the Gemini path so swapping in a real
    model requires no changes to the graph logic itself.
    """
    if "EXTRACT_CRITERIA" in prompt:
        return json.dumps([
            {"criterion_id": "INC-1", "type": "inclusion", "description": "Age >= 18"},
            {"criterion_id": "INC-2", "type": "inclusion", "description": "Diagnosis of Type 2 Diabetes"},
            {"criterion_id": "EXC-1", "type": "exclusion", "description": "eGFR < 30 mL/min"},
        ])
    if "ASSESS_CRITERION" in prompt:
        # Naive mock assessment: looks for simple keyword matches in the prompt
        met = "MATCH_TRUE" in prompt
        return json.dumps({
            "met": met,
            "evidence": "Mock evidence based on keyword match in patient record",
            "confidence": 0.6,
        })
    return "{}"


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    protocol_text: str
    study_id: str
    patient: PatientRecord
    criteria: list[EligibilityCriterion]
    assessments: list[CriterionAssessment]
    verdict: EligibilityVerdict


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def extract_criteria(state: AgentState) -> AgentState:
    prompt = f"EXTRACT_CRITERIA\nProtocol text:\n{state['protocol_text']}"
    raw = llm_call(prompt)
    parsed = json.loads(raw)
    state["criteria"] = [EligibilityCriterion(**c) for c in parsed]
    return state


def assess_criteria(state: AgentState) -> AgentState:
    patient = state["patient"]
    assessments: list[CriterionAssessment] = []

    for criterion in state["criteria"]:
        match_hint = _rule_based_match_hint(criterion, patient)
        prompt = (
            f"ASSESS_CRITERION\n"
            f"Criterion: {criterion.description}\n"
            f"Patient: {patient.model_dump_json()}\n"
            f"{'MATCH_TRUE' if match_hint else 'MATCH_FALSE'}"
        )
        raw = llm_call(prompt)
        parsed = json.loads(raw)
        assessments.append(
            CriterionAssessment(
                criterion_id=criterion.criterion_id,
                met=parsed["met"],
                evidence=parsed["evidence"],
                confidence=parsed["confidence"],
            )
        )

    state["assessments"] = assessments
    return state


def aggregate_verdict(state: AgentState) -> AgentState:
    assessments = state["assessments"]
    criteria_by_id = {c.criterion_id: c for c in state["criteria"]}

    failed_inclusions = [
        a for a in assessments
        if criteria_by_id[a.criterion_id].type == "inclusion" and not a.met
    ]
    met_exclusions = [
        a for a in assessments
        if criteria_by_id[a.criterion_id].type == "exclusion" and a.met
    ]

    eligible = len(failed_inclusions) == 0 and len(met_exclusions) == 0
    overall_confidence = (
        sum(a.confidence for a in assessments) / len(assessments) if assessments else 0.0
    )

    reasons = []
    for a in failed_inclusions:
        reasons.append(f"Inclusion criterion not met: {criteria_by_id[a.criterion_id].description}")
    for a in met_exclusions:
        reasons.append(f"Exclusion criterion met: {criteria_by_id[a.criterion_id].description}")
    if eligible:
        reasons.append("All inclusion criteria met and no exclusion criteria triggered.")

    state["verdict"] = EligibilityVerdict(
        patient_id=state["patient"].patient_id,
        study_id=state["study_id"],
        eligible=eligible,
        overall_confidence=round(overall_confidence, 2),
        criterion_assessments=assessments,
        reasons=reasons,
        flagged_for_review=overall_confidence < CONFIDENCE_THRESHOLD,
    )
    return state


def _rule_based_match_hint(criterion: EligibilityCriterion, patient: PatientRecord) -> bool:
    """
    Lightweight heuristic used only to steer the mock LLM toward a
    plausible answer during offline testing. A real model call would
    reason over the criterion text directly instead of relying on this.
    """
    text = criterion.description.lower()
    if "age" in text and ">=" in text:
        try:
            threshold = int("".join(filter(str.isdigit, text.split(">=")[1])))
            return patient.age >= threshold
        except (ValueError, IndexError):
            return False
    if "diabetes" in text:
        return any("diabetes" in d.lower() for d in patient.diagnosis)
    if "egfr" in text:
        return patient.lab_values.get("eGFR", 100) < 30
    return False


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract_criteria", extract_criteria)
    graph.add_node("assess_criteria", assess_criteria)
    graph.add_node("aggregate_verdict", aggregate_verdict)

    graph.set_entry_point("extract_criteria")
    graph.add_edge("extract_criteria", "assess_criteria")
    graph.add_edge("assess_criteria", "aggregate_verdict")
    graph.add_edge("aggregate_verdict", END)

    return graph.compile()


def run_eligibility_check(protocol_text: str, study_id: str, patient: PatientRecord) -> EligibilityVerdict:
    """Convenience entry point used by both the API and the eval script."""
    app = build_graph()
    final_state = app.invoke({
        "protocol_text": protocol_text,
        "study_id": study_id,
        "patient": patient,
        "criteria": [],
        "assessments": [],
        "verdict": None,
    })
    return final_state["verdict"]