"""
MCP server exposing the Clinical Trial Eligibility Agent as a tool.

This demonstrates how the agent's core capability -- assessing patient
eligibility against protocol criteria -- can be exposed via the Model
Context Protocol (MCP), making it directly callable by any MCP-compatible
client (e.g. Claude Desktop, or another agent acting as an orchestrator).

The same `run_eligibility_check` function used by the FastAPI service
(see api.py) is reused here unchanged -- MCP and REST are just two
different transport/interface layers wrapping the same underlying logic.

Run with:
    mcp dev mcp_server.py

Or register it in an MCP client (e.g. Claude Desktop's config) pointing
to this script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from agent import run_eligibility_check
from schemas import PatientRecord

mcp = FastMCP("clinical-trial-eligibility")


@mcp.tool()
def assess_patient_eligibility(
    protocol_text: str,
    study_id: str,
    patient_id: str,
    age: int,
    sex: str,
    diagnosis: list[str],
    medications: list[str],
    lab_values: dict[str, float],
    notes: str = "",
) -> dict:
    """
    Assess whether a patient is eligible for a clinical trial.

    Given a trial protocol's free-text criteria and a structured patient
    record, returns a structured verdict: eligible (bool), overall
    confidence, per-criterion evidence, and whether the case should be
    flagged for human review due to low confidence.

    This tool wraps the same LangGraph agent used by the project's
    FastAPI service -- MCP here is simply an alternative interface,
    letting any MCP-compatible client (e.g. an LLM assistant) call the
    eligibility check directly as part of a larger workflow.
    """
    patient = PatientRecord(
        patient_id=patient_id,
        age=age,
        sex=sex,
        diagnosis=diagnosis,
        medications=medications,
        lab_values=lab_values,
        notes=notes,
    )

    verdict = run_eligibility_check(
        protocol_text=protocol_text,
        study_id=study_id,
        patient=patient,
    )

    return verdict.model_dump()


@mcp.resource("protocol://sample-study-001")
def get_sample_protocol() -> str:
    """Returns the sample clinical trial protocol used in this project's tests."""
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_protocol.txt"
    return sample_path.read_text()


if __name__ == "__main__":
    mcp.run()