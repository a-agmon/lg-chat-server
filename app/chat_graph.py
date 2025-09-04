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
    """Build the prompt to send to the LLM.

    Always include a fresh system instruction and KB user-context (transient),
    then append the dynamic conversation history (user/assistant turns only),
    and finally the latest user message if provided.

    This avoids persisting the large system/KB context on every turn and
    prevents exponential duplication with add_messages aggregator.
    """
    # Transient base: system + KB user context
    complaint = state.get("complaint", "")
    record = state.get("kb_record", {})
    base_messages = build_initial_interview_messages(complaint, record)

    # Existing dynamic history (from checkpoint)
    existing_messages = state.get("messages", [])

    # Convert LangChain message objects to dicts if needed
    def message_to_dict(msg):
        if hasattr(msg, "type") and hasattr(msg, "content"):
            # Map LangChain 'ai'/'human' to OpenAI roles 'assistant'/'user' if needed
            role = getattr(msg, "type", None)
            if role == "ai":
                role = "assistant"
            elif role == "human":
                role = "user"
            return {"role": role or "user", "content": getattr(msg, "content", "")}
        return msg  # Already a dict-like {role, content}

    normalized_history = [message_to_dict(m) for m in existing_messages]

    # Filter out any accidental persisted system or KB-context messages from prior runs
    def is_kb_context_user_message(m: Dict[str, str]) -> bool:
        try:
            c = (m or {}).get("content", "") or ""
            return c.startswith("Medical category id:")
        except Exception:
            return False

    dynamic_history: List[Dict[str, str]] = [
        m for m in normalized_history
        if m and m.get("role") in {"user", "assistant"} and not is_kb_context_user_message(m)
    ]

    # Optionally append the last user message
    last_user = state.get("last_user_message")
    if last_user:
        dynamic_history = append_last_user_message(dynamic_history, last_user)

    # Final prompt to send to the LLM
    return [*base_messages, *dynamic_history]


def interview(state: ChatState) -> ChatState:
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    messages_to_send = _build_interview_prompt(state)
    result = llm.invoke(messages_to_send)
    # Ensure text is always a string, even if result.content is a list or other type
    text = str(result.content if hasattr(result, "content") else result)
    parsed = _safe_json_from_text(text)

    if parsed and parsed.get("done") is True:
        summary = parsed.get("summary")
        final_collected = parsed.get("collected_information") or parsed.get("collected_fields") or None
        assistant_msg = summary or ""
        # Persist only the incremental turns (avoid re-adding system/KB context)
        delta_messages: List[Dict[str, str]] = []
        if state.get("last_user_message"):
            delta_messages.append({"role": "user", "content": state.get("last_user_message") or ""})
        if assistant_msg:
            delta_messages.append({"role": "assistant", "content": assistant_msg})
        new_state: ChatState = {
            **state,
            "messages": delta_messages,
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
    # Persist only incremental messages (user reply if present, then assistant question)
    delta_messages: List[Dict[str, str]] = []
    if state.get("last_user_message"):
        delta_messages.append({"role": "user", "content": state.get("last_user_message") or ""})
    if assistant_msg:
        delta_messages.append({"role": "assistant", "content": assistant_msg})
    new_state: ChatState = {
        **state,
        "messages": delta_messages,
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
