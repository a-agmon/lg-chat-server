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
from .prompts import (
    build_select_kb_messages,
    build_initial_interview_messages,
    append_last_user_message,
)

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    # Core inputs
    complaint: str
    last_user_message: Optional[str]

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
    if state.get("kb_record"):
        return state
    
    kb = _load_kb()  # list of records

    # Build a small catalog for the LLM to select from
    catalog = [
        {"id": r.get("id"), "main_complaint": r.get("main_complaint", [])}
        for r in kb if r.get("id")
    ]
    complaint_text = (state.get("complaint") or "").strip()
    if not complaint_text:
        logger.error("No complaint text found")
        raise ValueError("No complaint text found")

    llm = ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key)
    choice = llm.invoke(build_select_kb_messages(catalog, complaint_text))
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
    logger.info("Main complaint: %s KB selected record_id=%s (session_id=%s)", complaint_text, selected.get("id"), state.get("session_id"))

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
        # First call: build the initial prompt using the selected KB record
        complaint = state.get("complaint", "")
        record = state.get("kb_record", {})
        messages = build_initial_interview_messages(complaint, record)
    else:
        # Subsequent calls: use existing messages and append the last user message
        last_user = state.get("last_user_message")
        messages = append_last_user_message(existing_messages[:], last_user)
    
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
