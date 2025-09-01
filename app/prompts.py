from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ----------------------
# KB Selection Prompts
# ----------------------

SELECT_KB_SYSTEM = (
    "You are a medical triage assistant. "
    "Choose the single most relevant complaint category record from the catalog "
    "based on the patient's main complaint and initial text. "
    "Output strict JSON only: {\"record_id\": string}."
)


def build_select_kb_messages(
    catalog: List[Dict[str, Any]],
    complaint: str,
) -> List[Dict[str, str]]:
    user = (
        "Catalog (id and keywords):\n" + json.dumps(catalog, ensure_ascii=False) + "\n\n" +
        f"Main complaint: {complaint}\n"
    )
    return [
        {"role": "system", "content": SELECT_KB_SYSTEM},
        {"role": "user", "content": user},
    ]


# ----------------------
# Interview Prompts
# ----------------------

INTERVIEW_SYSTEM = (
    "You are a medical intake assistant. "
    "Develop your own concise, empathetic questions to gather all information relevant to the case. "
    "Use the provided 'relevant_information' as guidance for what information should be collected. "
    "Ask only one question at a time, and continue asking until you have enough information. "
    "Only when you have enough information, stop asking and output a final JSON object.\n\n"
    "Final output format (strict JSON only when the interview is complete):\n"
    "{\n"
    "  \"done\": true,\n"
    "  \"collected_information\": object,\n"
    "  \"summary\": string\n"
    "}\n"
    "If you still need more information, DO NOT output JSON; instead, ask the next single question only."
    "IMPORTANT: Ask only one question at a time"
)


def build_initial_interview_messages(
    complaint: str,
    record: Dict[str, Any],
) -> List[Dict[str, str]]:
    record_id = record.get("id")
    relevant_info = record.get("relevant_information", "")
    relevant_tests = record.get("relevant_tests", "")

    user_context = (
        f"Medical category id: {record_id}\n"
        f"Main complaint: {complaint}\n\n"
        f"Relevant information to gather: {relevant_info}\n"
        f"Relevant tests (for context only, do not order tests): {relevant_tests}\n\n"
    )

    return [
        {"role": "system", "content": INTERVIEW_SYSTEM},
        {"role": "user", "content": user_context},
    ]


def append_last_user_message(
    history: List[Dict[str, str]],
    last_user_message: Optional[str],
) -> List[Dict[str, str]]:
    if last_user_message:
        return [*history, {"role": "user", "content": last_user_message}]
    return history

