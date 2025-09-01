from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from .config import settings
import logging

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    # Core inputs
    complaint: str
    last_user_message: Optional[str]
    initial_context: Optional[str]

    # Knowledge base selection
    kb_record: Dict[str, Any]  # {id, main_complaint[], relevant_information, relevant_tests}

    # Conversation and results
    # Chat history aggregated across turns
    messages: Annotated[List[Dict[str, str]], add_messages]  # {role: user|assistant|system, content: str}
    done: bool
    summary: Optional[str]
    next_question: Optional[str]
    final_collected: Optional[Dict[str, Any]]


def _load_kb() -> List[Dict[str, Any]]:
    with open(settings.kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # Try to extract first JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def retrieve_kb(state: ChatState) -> ChatState:
    kb = _load_kb()  # list of records
    if state.get("kb_record"):
        return state

    # Build a small catalog for the LLM to select from
    catalog = [
        {"id": r.get("id"), "main_complaint": r.get("main_complaint", [])}
        for r in kb if r.get("id")
    ]
    # Selection uses the initial context only (first turn)
    user_text = (state.get("initial_context") or "").strip()
    complaint_text = (state.get("complaint") or "").strip()

    system = (
        "You are a medical triage assistant. "
        "Choose the single most relevant complaint category recordfrom the catalog "
        "based on the patient's main complaint and initial text. Output strict JSON only: {\"record_id\": string}."
    )
    user = (
        "Catalog (id and keywords):\n" + json.dumps(catalog, ensure_ascii=False) + "\n\n" +
        f"Main complaint: {complaint_text}\n" +
        f"Initial message: {user_text}\n"
    )

    llm = ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key)
    choice = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    parsed = _safe_json_from_text(getattr(choice, "content", "")) or {}
    record_id = parsed.get("record_id") or parsed.get("id")

    selected: Optional[Dict[str, Any]] = None
    if record_id:
        selected = next((r for r in kb if r.get("id") == record_id), None)

    if selected:
        logger.info("KB selected record_id=%s", selected.get("id"))
    else:
        logger.error("KB selection failed; no matching record found for record_id=%s", record_id)
        raise ValueError(f"Failed to find KB record with id: {record_id}")

    new_state: ChatState = {
        **state,
        "kb_record": selected,
        "done": False,
        "summary": None,
        "next_question": None,
        "final_collected": None,
    }
    return new_state


def _build_interview_prompt(state: ChatState) -> List[Dict[str, str]]:
    existing_messages = state.get("messages", [])
    
    # Convert LangChain message objects to dicts if needed
    def message_to_dict(msg):
        if hasattr(msg, 'type'):  # LangChain message object
            return {"role": msg.type, "content": msg.content}
        return msg  # Already a dict
    
    existing_messages = [message_to_dict(msg) for msg in existing_messages]
    
    # Check if this is the first call (no messages yet)
    if not existing_messages:
        # First call: build the initial prompt
        complaint = state.get("complaint", "")
        initial_context = state.get("initial_context", "")
        record = state.get("kb_record", {})
        record_id = record.get("id")
        relevant_info = record.get("relevant_information", "")
        relevant_tests = record.get("relevant_tests", "")

        system = (
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
            "IMPORTANT: Ask only one question at a time and not more than 5 rounds of questions"
        )

        user_context = (
            f"Selected schema id: {record_id}\n"
            f"Main complaint: {complaint}\n\n"
            f"Relevant information to gather: {relevant_info}\n"
            f"Relevant tests (for context only, do not order tests): {relevant_tests}\n\n"
            f"Initial patient message: {json.dumps(initial_context or '', ensure_ascii=False)}\n"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_context},
        ]
    else:
        # Subsequent calls: use existing messages and append the last user message
        # The existing messages should already contain the system prompt and conversation history
        messages = existing_messages[:]
        last_user = state.get("last_user_message")
        if last_user:
            messages.append({"role": "user", "content": last_user})
    
    return messages


def interview(state: ChatState) -> ChatState:
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    messages = _build_interview_prompt(state)
    result = llm.invoke(messages)
    # Ensure text is always a string, even if result.content is a list or other type
    text = str(result.content if hasattr(result, "content") else result)
    parsed = _safe_json_from_text(text)

    if parsed and parsed.get("done") is True:
        summary = parsed.get("summary")
        final_collected = parsed.get("collected_information") or parsed.get("collected_fields") or None
        assistant_msg = summary or ""
        # Append assistant response to the messages we built for this call
        current_messages = messages  # The messages we just sent to LLM
        new_messages = current_messages + ([{"role": "assistant", "content": assistant_msg}] if assistant_msg else [])
        new_state: ChatState = {
            **state,
            "messages": new_messages,
            "done": True,
            "summary": summary,
            "next_question": None,
            "final_collected": final_collected,
            # Reset last user message once consumed
            "last_user_message": None,
        }
        logger.info("Interview done; final JSON returned")
        return new_state

    # Ongoing: treat model output as the next question
    assistant_msg = (text or "").strip()
    # Append assistant response to the messages we built for this call
    # (which includes system prompt on first call, conversation history on subsequent calls)
    current_messages = messages  # The messages we just sent to LLM
    new_messages = current_messages + ([{"role": "assistant", "content": assistant_msg}] if assistant_msg else [])
    new_state: ChatState = {
        **state,
        "messages": new_messages,
        "done": False,
        "summary": None,
        "next_question": assistant_msg,
        "final_collected": None,
        # Reset last user message once consumed
        "last_user_message": None,
    }
    return new_state


def make_graph():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve_kb", retrieve_kb)
    graph.add_node("interview", interview)

    # Conditional entry point based on whether KB has been retrieved
    def route_entry(state: ChatState) -> str:
        """Route to interview if KB already retrieved, otherwise to retrieve_kb"""
        if state.get("kb_record"):
            return "interview"
        return "retrieve_kb"
    
    # Use conditional entry instead of fixed entry point
    graph.add_conditional_edges(
        START,
        route_entry,
        {
            "retrieve_kb": "retrieve_kb",
            "interview": "interview"
        }
    )

    # After kb, go to interview
    graph.add_edge("retrieve_kb", "interview")

    # After one interview step, end (API handles loop across requests)
    graph.add_edge("interview", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    return app


# Singleton graph app
graph_app = make_graph()
