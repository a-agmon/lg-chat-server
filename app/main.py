from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import cast, List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from .config import settings
from .models import ChatResponse, ErrorResponse, MessageRequest, StartRequest
from .chat_graph import graph_app, _load_kb


logger = logging.getLogger("chat_server")
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


def get_session_checkpoint(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the checkpoint for a given session.
    
    Args:
        session_id: The session/thread ID to retrieve checkpoint for
        
    Returns:
        Checkpoint dict if found, None otherwise
    """
    try:
        cp = graph_app.checkpointer
        if cp is not None:
            cp = cast(BaseCheckpointSaver, cp)
            config: RunnableConfig = {"configurable": {"thread_id": session_id}}
            ckpt = cp.get(config)
            # Cast to dict since Checkpoint is a TypedDict that behaves like a dict
            return cast(Optional[Dict[str, Any]], ckpt)
    except Exception:
        pass
    return None


def get_session_messages(session_id: str) -> List[dict]:
    """Retrieve existing messages from the checkpointer for a given session.
    
    Args:
        session_id: The session/thread ID to retrieve messages for
        
    Returns:
        List of message dicts with 'role' and 'content' keys, empty list if not found
    """
    ckpt = get_session_checkpoint(session_id)
    if ckpt and ckpt.get("channel_values", {}).get("messages"):
        return ckpt.get("channel_values", {}).get("messages", [])
    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate required configuration once at startup
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not configured")
        raise RuntimeError("OPENAI_API_KEY is required")
    # Verify KB can be loaded
    try:
        kb = _load_kb()
        logger.info("KB loaded: %d records", len(kb) if kb else 0)
    except Exception as e:
        logger.exception("Failed to load KB: %s", e)
        raise
    
    yield
    
    # Cleanup code can be added here if needed in the future


app = FastAPI(
    title="Medical Intake Chatbot",
    version="0.1.0",
    description=(
        "LangGraph-powered medical intake chatbot. Provide a main complaint, then the service "
        "asks targeted questions until it collects required fields and returns a summary."
    ),
    contact={
        "name": "Chat Server",
        "url": "https://example.com",
    },
    license_info={
        "name": "Proprietary",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


def _invoke_graph(graph: CompiledStateGraph, state: dict, session_id: str) -> dict:
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    # One step through graph: retrieve_kb -> interview -> END
    return graph.invoke(state, config)


@app.get("/health", summary="Health check", description="Returns OK when the service is up.")
def health():
    return {"status": "ok"}


@app.post(
    "/chat/start",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Start a chat session",
    description=(
        "Initializes a new session with a main complaint and optional initial message. "
        "Returns the first question to ask or a final summary if requirements are already met."
    ),
)
def start_chat(req: StartRequest):
    session_id = req.session_id or uuid.uuid4().hex

    base_state = {
        "complaint": req.complaint,
        "initial_context": req.initial_message,
        "last_user_message": req.initial_message,
        "messages": ([{"role": "user", "content": req.initial_message}] if req.initial_message else []),
    }

    try:
        logger.info("start_chat session_id=%s complaint=%s", session_id, req.complaint)
        new_state = _invoke_graph(graph_app, base_state, session_id)
    except Exception as e:
        logger.exception("Failed to start chat: %s", e)
        raise HTTPException(status_code=500, detail="Failed to start chat")

    reply = new_state.get("summary") if new_state.get("done") else new_state.get("next_question")
    resp = ChatResponse(
        session_id=session_id,
        done=bool(new_state.get("done")),
        message=reply or "",
    )
    logger.info("start_chat response session_id=%s done=%s", session_id, resp.done)
    return resp


@app.post(
    "/chat/{session_id}/message",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Continue a chat session",
    description="Sends a user reply and receives the next question or final summary.",
)
def continue_chat(session_id: str = Path(..., description="Chat session id"), req: MessageRequest | None = None):
    if not req or not req.content:
        raise HTTPException(status_code=400, detail="Message content is required")

    # Get existing state to append messages properly
    existing_messages = get_session_messages(session_id)
    
    # Append new user message to existing messages
    state_delta = {
        "last_user_message": req.content,
        "messages": existing_messages + [{"role": "user", "content": req.content}],
    }

    try:
        logger.info("continue_chat session_id=%s", session_id)
        new_state = _invoke_graph(graph_app, state_delta, session_id)
    except Exception as e:
        logger.exception("Failed to continue chat: %s", e)
        raise HTTPException(status_code=500, detail="Failed to continue chat")

    reply = new_state.get("summary") if new_state.get("done") else new_state.get("next_question")
    resp = ChatResponse(
        session_id=session_id,
        done=bool(new_state.get("done")),
        message=reply or "",
    )
    logger.info("continue_chat response session_id=%s done=%s", session_id, resp.done)
    return resp


@app.get("/chat/{session_id}/state", summary="Get session state", description="Returns the latest stored graph state for debugging.")
def get_state(session_id: str):
    # For debugging/observability: retrieve the latest state snapshot
    try:
        ckpt = get_session_checkpoint(session_id)
        if not ckpt:
            return JSONResponse(status_code=404, content={"detail": "Session not found"})
        return ckpt.get("channel_values", {})
    except Exception as e:
        logger.exception("Failed to get state: %s", e)
        return JSONResponse(status_code=500, content={"detail": "Failed to fetch state"})
