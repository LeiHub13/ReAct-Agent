import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from react_agent import tools
from react_agent.context import Context
from react_agent.state import State

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_search_api(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    # Any accidental real HTTP request fails instead of contacting a provider.
    async_client = httpx.AsyncClient

    def unexpected_request(request):
        pytest.fail(f"Unexpected HTTP request: {request.url}")

    state = SimpleNamespace(handler=unexpected_request, requests=[])

    def dispatch(request):
        state.requests.append(request)
        return state.handler(request)

    transport = httpx.MockTransport(dispatch)
    monkeypatch.setattr(
        tools.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )
    return state


@pytest.fixture
def search_context(monkeypatch):
    context = Context(max_search_results=3)
    monkeypatch.setattr(
        tools, "get_runtime", lambda schema: SimpleNamespace(context=context)
    )
    return context


async def test_search_request_and_results(mock_search_api, search_context):
    results = {"results": [{"title": "Example", "url": "https://example.com"}]}
    mock_search_api.handler = lambda request: httpx.Response(200, json=results)

    assert await tools.search("  最新科技新闻  ") == results

    request = mock_search_api.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.deepseek.com/v1/tools/search"
    assert request.headers["authorization"] == "Bearer test-deepseek-key"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "query": "最新科技新闻",
        "count": 3,
        "need_content": True,
    }


async def test_search_accepts_environment_count(
    mock_search_api, search_context, monkeypatch
):
    monkeypatch.setenv("MAX_SEARCH_RESULTS", "5")
    search_context.max_search_results = Context().max_search_results
    mock_search_api.handler = lambda request: httpx.Response(200, json={"results": []})

    assert await tools.search("test") == {"results": []}
    assert json.loads(mock_search_api.requests[0].content)["count"] == 5


@pytest.mark.parametrize("query", ["", "   ", "字" * 129])
async def test_search_rejects_invalid_query(mock_search_api, search_context, query):
    with pytest.raises(ValueError, match="1 to 128"):
        await tools.search(query)
    assert mock_search_api.requests == []


@pytest.mark.parametrize("count", [0, 51, "invalid"])
async def test_search_rejects_invalid_count(mock_search_api, search_context, count):
    search_context.max_search_results = count
    with pytest.raises(ValueError):
        await tools.search("test")
    assert mock_search_api.requests == []


async def test_search_requires_key(mock_search_api, search_context, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        await tools.search("test")
    assert mock_search_api.requests == []


@pytest.mark.parametrize("status", [401, 402, 429, 500])
async def test_search_propagates_http_errors(mock_search_api, search_context, status):
    mock_search_api.handler = lambda request: httpx.Response(
        status, json={"error": {"message": "Request failed"}}
    )
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await tools.search("test")
    assert exc_info.value.response.status_code == status


async def test_search_propagates_timeout(mock_search_api, search_context):
    def timeout(request):
        raise httpx.ReadTimeout("Search timed out", request=request)

    mock_search_api.handler = timeout
    with pytest.raises(httpx.ReadTimeout):
        await tools.search("test")


async def test_search_runs_in_tool_node(mock_search_api):
    payload = {"results": [{"title": "Example", "summary": "Search result"}]}
    mock_search_api.handler = lambda request: httpx.Response(200, json=payload)
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("tools", ToolNode(tools.TOOLS))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search", "args": {"query": "test"}, "id": "search-1"}
                    ],
                )
            ]
        },
        context=Context(max_search_results=2),
    )

    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "search-1"
    assert json.loads(message.content) == payload
    assert json.loads(mock_search_api.requests[0].content)["count"] == 2
