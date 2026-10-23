"""Define the state structures for the agent.

定义智能体的状态结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    定义智能体的输入状态，为外部提供较窄的接口。

    This class is used to define the initial state and structure of incoming data.
    此类用于定义初始状态及传入数据的结构。
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.
    用于跟踪智能体主要执行状态的消息。

    Typically accumulates a pattern of:
    通常会按以下模式累积：
    1. HumanMessage - user input
       HumanMessage - 用户输入
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
       带有 .tool_calls 的 AIMessage - 智能体选择要调用的工具来收集信息
    3. ToolMessage(s) - the responses (or errors) from the executed tools
       ToolMessage（一条或多条）- 已执行工具的响应或错误
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
       不带 .tool_calls 的 AIMessage - 智能体以非结构化格式回复用户
    5. HumanMessage - user responds with the next conversational turn
       HumanMessage - 用户在下一轮对话中的回复

    Steps 2-5 may repeat as needed.
    第 2 至第 5 步可根据需要重复执行。

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    `add_messages` 注解确保新消息与现有消息合并，并按 ID 更新，从而维持“只追加”状态；
    如果提供了相同 ID 的消息，则会更新对应消息。
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    表示智能体的完整状态，在 InputState 的基础上扩展其他属性。

    This class can be used to store any information needed throughout the agent's lifecycle.
    此类可用于存储智能体整个生命周期中所需的任何信息。
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.
    指示当前步骤是否为图因超过限制而报错前的最后一步。

    This is a 'managed' variable, controlled by the state machine rather than user code.
    这是一个由状态机控制、而非由用户代码控制的“托管”变量。
    It is set to 'True' when the step count reaches recursion_limit - 1.
    当步骤计数达到 recursion_limit - 1 时，该变量会被设为 True。
    """

    # Additional attributes can be added here as needed.
    # 可根据需要在此添加其他属性。
    # Common examples include:
    # 常见示例包括：
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
