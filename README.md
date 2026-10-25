# ReAct Agent（LangGraph + DeepSeek）

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的 ReAct 智能体，默认使用 DeepSeek 作为对话模型，并通过 DeepSeek Search API 提供网页搜索能力。项目通过 FastAPI 暴露 SSE 流式接口，可作为独立服务接入电商客服、导购等业务系统。

## 工作原理

智能体按照 ReAct（Reasoning and Action）循环运行：

1. 接收用户问题
2. 模型推理并决定是否调用工具
3. 执行所选工具（当前为网页搜索）
4. 观察工具返回结果
5. 重复 2–4 步，直到给出最终回答

```mermaid
flowchart LR
    START --> call_model
    call_model -->|有 tool_calls| tools
    tools --> call_model
    call_model -->|无 tool_calls| END
    style START stroke-width:3px
    style call_model stroke-width:3px
    style tools stroke-width:3px
    style END stroke-width:3px
```

核心逻辑位于 [src/react_agent/graph.py](./src/react_agent/graph.py)：`call_model` 负责调用模型并绑定工具，`route_model_output` 根据最后一条消息是否包含工具调用来决定继续执行工具还是结束。达到递归上限的最后一步若仍在请求工具，会返回超步提示而不是继续循环。

## 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- 一个 DeepSeek API Key（对话与搜索共用）

## 快速开始

1. 安装依赖

```bash
uv sync --locked
```

2. 配置环境变量

```bash
cp .env.example .env
```

在 `.env` 中填入：

```dotenv
MODEL=deepseek/deepseek-flash
DEEPSEEK_API_KEY=your-deepseek-api-key
MAX_SEARCH_RESULTS=10
LANGSMITH_PROJECT=react-agent
```

`MODEL`、`MAX_SEARCH_RESULTS` 等配置会自动从 `.env` 读取（见 [context.py](./src/react_agent/context.py)）；如果改用其他厂商的模型，需要额外配置对应的 API Key，例如 `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`FIREWORKS_API_KEY`。网页搜索始终使用 `DEEPSEEK_API_KEY`。

3. 启动 API 服务

```bash
uv run uvicorn react_agent.api:app --reload --port 8000
```

浏览器访问 `http://127.0.0.1:8000/` 使用内置聊天页面，可以直接看到模型输出和工具调用过程；接口文档位于 `http://127.0.0.1:8000/docs`。

## API

### `GET /health`

健康检查，返回 `{"status": "ok"}`。

### `POST /chat`

请求体：

```json
{
  "message": "LangChain 的创始人是谁？",
  "thread_id": "optional-thread-id"
}
```

- `message`：用户输入，长度 1–4000
- `thread_id`：会话标识，可省略；省略时服务端生成并在首个事件中返回。多轮对话时复用同一个 `thread_id`

响应为 `text/event-stream`，事件类型如下：

```
event: meta
data: {"thread_id": "..."}

event: token
data: {"content": "..."}

event: error
data: {"error": "..."}

event: done
data: {"thread_id": "..."}
```

示例：

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "LangChain 的创始人是谁？"}'
```

会话由内存 checkpointer（`InMemorySaver`）保存，服务重启后丢失；生产环境应替换为 Postgres/Redis 等持久化 checkpointer。跨域来源通过 `CORS_ORIGINS` 配置（逗号分隔，默认 `*`，生产环境请收紧）。

## 配置项

运行时配置定义在 [src/react_agent/context.py](./src/react_agent/context.py)：

| 配置 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model` | `MODEL` | `deepseek/deepseek-flash` | 对话模型，格式为 `provider/model` |
| `max_search_results` | `MAX_SEARCH_RESULTS` | `10` | 每次搜索返回的结果数，范围 1–50 |
| `system_prompt` | `SYSTEM_PROMPT` | 内置提示词 | 系统提示词，运行时自动注入当前时间 |
| CORS 来源 | `CORS_ORIGINS` | `*` | API 允许的前端来源，逗号分隔 |

通过参数传入的值优先于环境变量。提示词默认定义在 [prompts.py](./src/react_agent/prompts.py)。

## 内置工具

[search](./src/react_agent/tools.py) 调用 `https://api.deepseek.com/v1/tools/search`，返回网页链接、摘要和可用的正文内容。它会校验：

- 查询词长度 1–128 个字符
- `max_search_results` 在 1–50 之间
- 必须配置 `DEEPSEEK_API_KEY`

HTTP 错误、超时等异常会以 `error` 事件返回给客户端。

## 项目结构

```
.
├── src/react_agent/
│   ├── api.py                  # FastAPI 入口与 SSE 流式接口
│   ├── web/index.html          # 内置聊天调试页面
│   ├── graph.py                # ReAct 图：节点、路由与循环
│   ├── tools.py                # DeepSeek Search 工具与业务工具占位
│   ├── context.py              # 运行时配置（模型、提示词、搜索数量）
│   ├── state.py                # 输入状态与完整状态定义
│   ├── prompts.py              # 默认系统提示词
│   └── utils.py                # 模型加载等辅助函数
└── tests/
    ├── unit_tests/             # 离线单元测试（Mock HTTP / Fake Graph）
    └── integration_tests/      # 真实模型调用集成测试
```

## 开发

### 常用命令

```bash
uv sync --extra dev                          # 安装 ruff / mypy 等开发工具
uv run pytest tests/unit_tests               # 运行单元测试
uv run pytest tests/integration_tests        # 运行集成测试（需要真实 API Key）
uv run ruff check .                          # 代码检查
uv run ruff format .                         # 格式化与 import 排序
uv run mypy --strict src/                    # 类型检查
```

也可以针对单个文件运行测试：

```bash
uv run pytest tests/unit_tests/test_tools.py
```

## 自定义

- **新增工具**：在 [tools.py](./src/react_agent/tools.py) 中编写函数并加入 `TOOLS` 列表，`call_model` 会自动绑定
- **更换模型**：修改 `MODEL` 环境变量，或调用时通过 `Context(model=...)` 传入
- **调整行为**：修改 [prompts.py](./src/react_agent/prompts.py) 中的系统提示词
- **调整流程**：在 [graph.py](./src/react_agent/graph.py) 中增加节点或修改条件边

接入业务系统时，建议把 `search` 替换为商品、订单、物流等工具，工具内部调用业务 API，写操作（退款、改地址等）增加用户确认与权限校验。
