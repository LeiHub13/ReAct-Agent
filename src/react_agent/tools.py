"""Provide web search through the DeepSeek Search API.

通过 DeepSeek Search API 提供网页搜索。
"""

import os
from typing import Any, Callable, List, cast

import httpx
from langgraph.runtime import get_runtime

from react_agent.context import Context


async def search(query: str) -> dict[str, Any]:
    """Search the web for current information using DeepSeek Search.

    使用 DeepSeek Search 搜索最新网页资料，返回链接、摘要和可用的正文。

    Args:
        query: Search terms containing 1 to 128 characters.
            搜索词长度须为 1 至 128 个字符。
    """
    query = query.strip()
    if not query or len(query) > 128:
        raise ValueError("Search query must contain 1 to 128 characters.")

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY to use DeepSeek Search.")

    runtime = get_runtime(Context)
    # Environment-backed context values may arrive as strings.
    # 从环境变量读取的配置值可能是字符串，发送请求前转换并检查范围。
    count = int(runtime.context.max_search_results)
    if not 1 <= count <= 50:
        raise ValueError("max_search_results must be between 1 and 50.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.deepseek.com/v1/tools/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "count": count, "need_content": True},
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())


def query() -> str:
    """Query business data."""
    return ""


TOOLS: List[Callable[..., Any]] = [search, query]
