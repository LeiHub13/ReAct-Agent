"""Define a custom Reasoning and Action agent.

定义一个自定义的推理与行动（ReAct）智能体。

Works with a chat model with tool calling support.
使用支持工具调用的聊天模型。
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# Define the function that calls the model
# 定义调用模型的函数。


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    调用驱动该智能体的大语言模型。

    This function prepares the prompt, initializes the model, and processes the response.
    此函数会准备提示词、初始化模型并处理模型响应。

    Args:
        state (State): The current state of the conversation.
            当前对话状态。
        config (RunnableConfig): Configuration for the model run.
            模型运行配置。

    Returns:
        dict: A dictionary containing the model's response message.
            包含模型响应消息的字典。
    """
    # Initialize the model with tool binding. Change the model or add more tools here.
    # 初始化模型并绑定工具。可在此更换模型或添加更多工具。
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    # 格式化系统提示词。可通过修改它来调整智能体的行为。
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    # 获取模型响应。
    response = cast(  # type: ignore[redundant-cast]
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    # 处理最后一步模型仍想调用工具的情况。
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    # 将模型响应作为列表返回，以便追加到现有消息中。
    return {"messages": [response]}


# Define a new graph
# 定义一个新图。

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the two nodes we will cycle between
# 定义循环中会经过的两个节点。
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# 将 `call_model` 设为入口节点。
# This means that this node is the first one called
# 这意味着该节点会首先执行。
builder.add_edge("__start__", "call_model")


# Decide which path the workflow takes after the model responds.
# 决定模型回复后工作流的下一步走向。
def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    根据模型输出决定下一个节点。

    This function checks if the model's last message contains tool calls.
    此函数会检查模型的最后一条消息是否包含工具调用。

    Args:
        state (State): The current state of the conversation.
            当前对话状态。

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
            下一个要执行的节点名称（"__end__" 或 "tools"）。
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    # 如果没有工具调用，则结束流程。
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    # 否则执行模型请求的操作。
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
# 添加条件边，以决定 `call_model` 执行后的下一步。
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # `call_model` 执行完毕后，会据此安排下一个节点。
    # based on the output from route_model_output
    # 路由结果由 `route_model_output` 返回。
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# 添加从 `tools` 到 `call_model` 的普通边。
# This creates a cycle: after using tools, we always return to the model
# 这会形成循环：工具执行后总是返回模型节点。
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
# 将图构建器编译为可执行图。
graph = builder.compile(name="ReAct Agent")
