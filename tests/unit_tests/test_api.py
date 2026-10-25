"""Tests for the HTTP API."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from react_agent import api


class FakeGraph:
    """Stand-in for the compiled agent graph."""

    async def astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[tuple[str, Any]]:
        """Yield a fixed streaming sequence with one tool call."""
        yield "messages", (AIMessageChunk(content="Hello"), {})
        yield "messages", (AIMessageChunk(content=" world"), {})
        yield (
            "messages",
            (
                ToolMessage(
                    content="raw tool payload", tool_call_id="call-0", name="search"
                ),
                {},
            ),
        )
        yield (
            "updates",
            {
                "call_model": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call-1",
                                    "name": "search",
                                    "args": {"query": "test"},
                                }
                            ],
                        )
                    ]
                }
            },
        )
        yield (
            "updates",
            {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content="search result",
                            tool_call_id="call-1",
                            name="search",
                        )
                    ]
                }
            },
        )


def test_health() -> None:
    """Return ok from the health endpoint."""
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_chat_page() -> None:
    """Serve the built-in chat page at the root."""
    client = TestClient(api.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "ReAct Agent" in response.text


def test_chat_streams_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stream meta, token, and done events for a chat turn."""
    monkeypatch.setattr(api, "graph", FakeGraph())
    client = TestClient(api.app)

    with client.stream(
        "POST", "/chat", json={"message": "hi", "thread_id": "t-1"}
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: meta\ndata: {"thread_id": "t-1"}' in body
    assert 'event: token\ndata: {"content": "Hello"}' in body
    assert 'event: token\ndata: {"content": " world"}' in body
    assert (
        'event: tool_call\ndata: {"id": "call-1", "name": "search", "args": {"query": "test"}}'
        in body
    )
    assert '"name": "search"' in body
    assert '"status": "success"' in body
    assert '"summary": "search result"' in body
    assert "raw tool payload" not in body
    assert "event: done" in body


def test_chat_rejects_empty_message() -> None:
    """Reject empty messages with a validation error."""
    client = TestClient(api.app)

    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422
