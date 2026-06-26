"""
Pydantic schemas for the Clinical Trial Eligibility Agent.

These models define the structured contracts between agent steps and
the final output returned to the caller. Using Pydantic here means the
LLM's output is validated against a strict schema rather than trusted
as free-text -- if the model returns something malformed, we know
immediately instead of silently passing bad data downstream.
"""

from typing import Literal
from pydantic import BaseModel, Field


class EligibilityCriterion(BaseModel):
    """A single inclusion or exclusion criterion extracted from a protocol."""

    criterion_id: str = Field(..., description="Short identifier, e.g. 'INC-1' or 'EXC-3'")
    type: Literal["inclusion", "exclusion"]
    description: str = Field(..., description="The criterion as stated in the protocol")


class ProtocolCriteria(BaseModel):
    """All criteria extracted from a clinical trial protocol."""

    study_id: str
    criteria: list[EligibilityCriterion]


class PatientRecord(BaseModel):
    """Minimal structured patient record used as agent input."""

    patient_id: str
    age: int
    sex: Literal["male", "female"]
    diagnosis: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    lab_values: dict[str, float] = Field(default_factory=dict)
    notes: str = Field(default="", description="Free-text clinical notes, if any")


class CriterionAssessment(BaseModel):
    """The agent's assessment of a single criterion against a patient."""

    criterion_id: str
    met: bool
    evidence: str = Field(..., description="What in the patient record supports this assessment")
    confidence: float = Field(..., ge=0.0, le=1.0)


class EligibilityVerdict(BaseModel):
    """
    Final structured output of the eligibility agent.

    This is the contract the API guarantees to callers. `flagged_for_review`
    is set whenever overall confidence is below a defined threshold --
    this is the human-in-the-loop gate: low-confidence verdicts should
    never be treated as final without a human checking them first.
    """

    patient_id: str
    study_id: str
    eligible: bool
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    criterion_assessments: list[CriterionAssessment]
    reasons: list[str] = Field(..., description="Human-readable summary of the decision")
    flagged_for_review: bool = Field(
        ..., description="True if confidence is below threshold and a human should review"
    )