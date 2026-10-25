"""HTTP API for the ReAct agent.

通过 FastAPI 暴露流式对话接口。
"""

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from react_agent.context import Context
from react_agent.graph import builder
from react_agent.state import InputState
from react_agent.utils import get_message_text

load_dotenv()

WEB_INDEX = Path(__file__).parent / "web" / "index.html"

app = FastAPI(title="ReAct Agent API", version="0.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = builder.compile(name="ReAct Agent", checkpointer=InMemorySaver())


class ChatRequest(BaseModel):
    """A chat turn sent by the client."""

    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default_factory=lambda: str(uuid4()))


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format a Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream(request: ChatRequest) -> AsyncIterator[str]:
    yield _sse("meta", {"thread_id": request.thread_id})
    try:
        stream = cast(
            AsyncIterator[tuple[str, Any]],
            graph.astream(
                InputState(messages=[HumanMessage(content=request.message)]),
                config={"configurable": {"thread_id": request.thread_id}},
                context=Context(),
                stream_mode=["messages", "updates"],
            ),
        )
        async for mode, payload in stream:
            if mode == "messages":
                chunk, _metadata = payload
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = get_message_text(cast(BaseMessage, chunk))
                if text:
                    yield _sse("token", {"content": text})
            elif mode == "updates":
                for update in payload.values():
                    for message in (update or {}).get("messages", []):
                        if isinstance(message, AIMessage):
                            for call in message.tool_calls:
                                yield _sse(
                                    "tool_call",
                                    {
                                        "id": call["id"],
                                        "name": call["name"],
                                        "args": call["args"],
                                    },
                                )
                        elif isinstance(message, ToolMessage):
                            yield _sse(
                                "tool_result",
                                {
                                    "id": message.tool_call_id,
                                    "name": message.name,
                                    "status": message.status,
                                    "summary": _summarize(message.content),
                                },
                            )
    except Exception as exc:
        yield _sse("error", {"error": str(exc)})
    yield _sse("done", {"thread_id": request.thread_id})


def _summarize(content: Any, limit: int = 500) -> str:
    """Build a short, readable tool output summary."""
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            text = content
        else:
            results = parsed.get("results") if isinstance(parsed, dict) else None
            if isinstance(results, list):
                lines = []
                for item in results[:5]:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title") or "未命名结果"
                    url = item.get("url") or ""
                    summary = str(item.get("summary") or "").strip().splitlines()
                    lines.append(
                        "\n".join(
                            part
                            for part in (title, url, summary[0] if summary else "")
                            if part
                        )
                    )
                text = "\n\n".join(lines)
            else:
                text = content
    else:
        text = json.dumps(content, ensure_ascii=False)
    return text if len(text) <= limit else f"{text[:limit]}..."


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the built-in chat page."""
    return FileResponse(WEB_INDEX)


@app.get("/health")
async def health() -> dict[str, str]:
    """Check service health."""
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream a chat response as Server-Sent Events."""
    return StreamingResponse(
        _stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
