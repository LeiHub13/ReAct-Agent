"""Define the configurable parameters for the agent.

定义智能体的可配置参数。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent.

    智能体的上下文。
    """

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent. "
            "用于智能体交互的系统提示词，用来设定智能体的上下文和行为。"
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="deepseek/deepseek-flash",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name. "
            "智能体主要交互所用的语言模型名称，格式应为 provider/model-name。"
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query. "
            "每次搜索查询最多返回的结果数量。"
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args.

        为未通过参数传入的属性读取环境变量。
        """
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
