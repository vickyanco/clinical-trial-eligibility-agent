"""
Evaluation harness for the Clinical Trial Eligibility Agent.

This is the core of the project's evaluation philosophy: agent outputs
are treated as hypotheses to be checked against known-correct answers,
not as results to be trusted on faith. Every test case has a
human-defined `expected_eligible` ground truth, including deliberately
chosen edge cases (boundary ages, borderline lab values) where a
naive implementation is most likely to fail.

Run with:
    python evaluate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import run_eligibility_check
from schemas import PatientRecord


def load_test_cases(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_evaluation(test_file: str = "test_cases.json") -> None:
    data = load_test_cases(Path(__file__).parent / test_file)
    protocol_text = data["protocol_text"]
    study_id = data["study_id"]
    cases = data["cases"]

    results = []
    correct = 0

    for case in cases:
        patient = PatientRecord(**case["patient"])
        verdict = run_eligibility_check(protocol_text, study_id, patient)

        is_correct = verdict.eligible == case["expected_eligible"]
        correct += int(is_correct)

        results.append({
            "patient_id": patient.patient_id,
            "description": case["description"],
            "expected": case["expected_eligible"],
            "predicted": verdict.eligible,
            "correct": is_correct,
            "confidence": verdict.overall_confidence,
            "flagged_for_review": verdict.flagged_for_review,
        })

    accuracy = correct / len(cases) if cases else 0.0

    print("=" * 70)
    print("CLINICAL TRIAL ELIGIBILITY AGENT — EVALUATION REPORT")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        flag = " [FLAGGED FOR REVIEW]" if r["flagged_for_review"] else ""
        print(f"[{status}] {r['patient_id']}: {r['description']}")
        print(f"       expected={r['expected']}  predicted={r['predicted']}  "
              f"confidence={r['confidence']}{flag}")
    print("-" * 70)
    print(f"Accuracy: {correct}/{len(cases)} ({accuracy:.1%})")
    flagged_count = sum(1 for r in results if r["flagged_for_review"])
    print(f"Flagged for human review: {flagged_count}/{len(cases)}")
    print("=" * 70)

    failed = [r for r in results if not r["correct"]]
    if failed:
        print("\nFailed cases (would require investigation in a real system):")
        for r in failed:
            print(f"  - {r['patient_id']}: {r['description']}")


if __name__ == "__main__":
    run_evaluation()