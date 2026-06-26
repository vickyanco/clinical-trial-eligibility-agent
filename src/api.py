"""
FastAPI service exposing the Clinical Trial Eligibility Agent.

Run with:
    uvicorn api:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API documentation.
"""

from fastapi import FastAPI, HTTPException

from agent import run_eligibility_check
from schemas import PatientRecord, EligibilityVerdict
from pydantic import BaseModel

app = FastAPI(
    title="Clinical Trial Eligibility Agent",
    description=(
        "Agentic AI service that assesses patient eligibility against "
        "clinical trial protocol criteria, with confidence scoring and "
        "human-in-the-loop flagging for low-confidence verdicts."
    ),
    version="0.1.0",
)


class EligibilityRequest(BaseModel):
    protocol_text: str
    study_id: str
    patient: PatientRecord


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/assess", response_model=EligibilityVerdict)
def assess_eligibility(request: EligibilityRequest) -> EligibilityVerdict:
    """
    Assess whether a patient is eligible for a clinical trial given
    the protocol's free-text criteria and a structured patient record.

    Returns a structured verdict including per-criterion evidence and
    a `flagged_for_review` field -- if true, this verdict should be
    reviewed by a qualified human before being acted upon.
    """
    try:
        return run_eligibility_check(
            protocol_text=request.protocol_text,
            study_id=request.study_id,
            patient=request.patient,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eligibility assessment failed: {exc}")