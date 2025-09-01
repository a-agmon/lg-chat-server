from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Optional


class StartRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "complaint": "headache",
                "initial_message": "I have a bad headache since yesterday"
            }
        ]
    })
    session_id: Optional[str] = Field(
        default=None, description="Optional session id; server generates one if omitted"
    )
    complaint: str = Field(..., description="Main complaint, e.g., 'headache'")
    initial_message: Optional[str] = Field(
        default=None, description="Optional initial user message narrative"
    )


class MessageRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {"content": "It's about 7 out of 10, mostly on the right side."}
        ]
    })
    content: str = Field(..., description="User reply content")


class ChatResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "session_id": "abc123",
                "done": False,
                "message": "How severe is the pain (0-10)?"
            }
        ]
    })
    session_id: str
    done: bool
    message: str


class ErrorResponse(BaseModel):
    detail: str
