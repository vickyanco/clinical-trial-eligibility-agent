"""
FastAPI service exposing the Clinical Trial Eligibility Agent.

Run with:
    uvicorn api:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API documentation.
"""

import uuid

from ag_ui.core import (
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent import build_graph, run_eligibility_check
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


def _serialize_state(state: dict) -> dict:
    """JSON-safe snapshot of AgentState for StateSnapshotEvent.snapshot."""
    return {
        "protocol_text": state["protocol_text"],
        "study_id": state["study_id"],
        "patient": state["patient"].model_dump(),
        "criteria": [c.model_dump() for c in state["criteria"]],
        "assessments": [a.model_dump() for a in state["assessments"]],
        "verdict": state["verdict"].model_dump() if state["verdict"] else None,
    }


@app.post("/assess/stream")
async def assess_eligibility_stream(request: EligibilityRequest, http_request: Request):
    """
    Same eligibility check as `/assess`, but streamed as AG-UI protocol
    events (https://docs.ag-ui.com/) -- one step per graph node
    (extract_criteria, assess_criteria, aggregate_verdict) -- so a
    frontend can show live progress instead of waiting for a single
    final response.

    This endpoint accepts the same domain-specific request body as
    `/assess` rather than AG-UI's generic `RunAgentInput` (thread/message
    history), since this agent's input doesn't map naturally onto a chat
    message list; only the response side conforms to the AG-UI event
    stream. See the README's "Honest limitations" section.
    """
    encoder = EventEncoder(accept=http_request.headers.get("accept"))
    thread_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())

    async def event_generator():
        try:
            yield encoder.encode(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id))

            initial_state = {
                "protocol_text": request.protocol_text,
                "study_id": request.study_id,
                "patient": request.patient,
                "criteria": [],
                "assessments": [],
                "verdict": None,
            }
            final_verdict = None
            # stream_mode="updates" only yields once a node has finished, so
            # STEP_STARTED/STEP_FINISHED below are both emitted after the
            # fact rather than truly bracketing the node's execution -- true
            # start/finish timing would need per-node instrumentation in
            # agent.py, which is out of scope here.
            async for chunk in build_graph().astream(initial_state, stream_mode="updates"):
                for node_name, node_state in chunk.items():
                    yield encoder.encode(StepStartedEvent(type=EventType.STEP_STARTED, step_name=node_name))
                    yield encoder.encode(
                        StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=_serialize_state(node_state))
                    )
                    yield encoder.encode(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=node_name))
                    if node_state.get("verdict") is not None:
                        final_verdict = node_state["verdict"]

            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=thread_id,
                    run_id=run_id,
                    result=final_verdict.model_dump() if final_verdict else None,
                )
            )
        except Exception as exc:
            yield encoder.encode(RunErrorEvent(type=EventType.RUN_ERROR, message=str(exc)))

    return StreamingResponse(event_generator(), media_type=encoder.get_content_type())