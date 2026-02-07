# NiBot 架构设计文档

**时间**: 2026-02-06
**版本**: 0.1.0
**总代码量**: 1771行 Python (25个文件), 39个测试用例全部通过

---

## 一、设计哲学

NiBot是一个轻量级多通道AI代理框架。设计原则三条：

1. **数据结构驱动行为** -- 5个dataclass定义了系统所有数据流，没有多余的抽象
2. **消除特殊情况** -- 空Session就是新Session，Tool调用是循环内的正常路径而非分支
3. **最小依赖** -- 核心框架零外部运行时依赖（litellm/pydantic/loguru均为必要项），Channel和Web工具通过optional extras按需安装

框架源自两个项目的共同架构模式提取：
- **OpenClaw** (TypeScript, 5000+文件) -- 重量级多通道AI助手
- **Nanobot** (Python, ~3400行) -- 极简AI代理框架

以Nanobot为骨架，融入OpenClaw的关键增强。

---

## 二、分层架构

```
┌─────────────────────────────────────────────────────┐
│                    NiBot (app.py)                    │  Composition Root
│              创建、组装、启动所有组件                    │
└──────────┬──────────┬──────────┬────────────────────┘
           │          │          │
     ┌─────▼───┐ ┌────▼────┐ ┌──▼──────────┐
     │ Channel │ │  Agent  │ │  Subagent   │           执行层
     │ (N个)   │ │  Loop   │ │  Manager    │
     └────┬────┘ └────┬────┘ └──────┬──────┘
          │           │             │
     ┌────▼───────────▼─────────────▼────┐
     │          MessageBus (bus.py)       │             路由层
     │    inbound Queue + outbound Queue  │
     └────────────────┬──────────────────┘
                      │
  ┌──────┬────────┬───▼───┬──────────┬──────────┐
  │Types │Config  │Session│ Memory   │ Skills   │     基础层
  │6 DC  │Pydantic│JSONL  │ Markdown │ Frontmatter│
  └──────┴────────┴───────┴──────────┴──────────┘
```

**依赖方向严格单向**: 上层依赖下层，下层不知道上层的存在。无循环依赖。

---

## 三、核心数据流

```
用户消息                                         用户收到回复
   │                                                ▲
   ▼                                                │
┌──────────┐  Envelope  ┌──────────┐  Envelope  ┌───┴──────┐
│ Channel  │ ─────────→ │   Bus    │ ─────────→ │ Channel  │
│ (入站)   │  publish   │ inbound  │  outbound  │ (出站)   │
└──────────┘  _inbound  │  Queue   │  dispatch  └──────────┘
                        └────┬─────┘
                             │ consume_inbound
                             ▼
                     ┌───────────────┐
                     │  AgentLoop    │
                     │  ._process()  │
                     └───────┬───────┘
                             │
              ┌──────────────▼──────────────┐
              │     LLM + Tool 迭代循环      │
              │                              │
              │  1. ContextBuilder.build()   │
              │     → system + history + user│
              │                              │
              │  2. Provider.chat()          │
              │     → LLMResponse            │
              │                              │
              │  3. if has_tool_calls:       │
              │       Registry.execute()     │
              │       append results         │
              │       goto 2                 │
              │     else:                    │
              │       break (got final text) │
              │                              │
              │  4. Session.save()           │
              └──────────────────────────────┘
```

**关键设计**: Tool调用不是if-else分支，而是for循环内的正常路径。`if not has_tool_calls: break` -- 一行消除所有特殊情况。最多迭代`max_iterations`次（默认20），防止无限循环。

---

## 四、模块职责

### 4.1 数据层 (types.py, 66行)

5个dataclass，零方法（除`LLMResponse.has_tool_calls`属性）：

| 数据结构 | 职责 | 关键字段 |
|----------|------|----------|
| `Envelope` | Channel与Bus之间的消息信封 | channel, chat_id, sender_id, content, media |
| `ToolCall` | LLM发出的工具调用请求 | id, name, arguments |
| `ToolResult` | 工具执行结果 | call_id, name, content, is_error |
| `LLMResponse` | LLM完成响应 | content, tool_calls, finish_reason, usage |
| `SkillSpec` | 解析后的SKILL.md规格 | name, description, body, always, requires_* |

**设计决策**:
- `Envelope`统一入站/出站，方向由Queue决定，不需要direction字段
- LLM消息直接用`list[dict]`，不封装成Message类 -- 与OpenAI/LiteLLM兼容，零序列化开销

### 4.2 消息总线 (bus.py, 48行)

```python
class MessageBus:
    _inbound:  asyncio.Queue[Envelope]  # Channel → Agent
    _outbound: asyncio.Queue[Envelope]  # Agent → Channel
    _subscribers: dict[str, list[Callback]]  # channel_name → [send函数]
```

- `publish_inbound()` -- Channel调用，放入入站队列
- `consume_inbound()` -- AgentLoop调用，阻塞等待
- `publish_outbound()` -- AgentLoop调用，放入出站队列
- `dispatch_outbound()` -- 后台协程，按channel名分派给订阅者
- `subscribe_outbound()` -- Channel注册时调用

**为什么用Queue而非直接调用**: 解耦。Channel不知道Agent的存在，Agent不知道Channel的存在。两者通过Bus的Queue异步通信。添加新Channel不需要修改Agent代码。

### 4.3 工具系统 (registry.py, 70行)

```python
class Tool(ABC):           # 实现4个抽象: name, description, parameters, execute
class ToolRegistry:        # register, get_definitions, execute, has
```

**策略过滤** (源自OpenClaw `pi-tools.policy.ts`):
```python
def get_definitions(self, deny: list[str] | None = None) -> list[dict]:
    deny_set = set(deny or [])
    return [t.to_schema() for t in self._tools.values() if t.name not in deny_set]
```

子代理调用时传入`deny=["message", "spawn"]`，防止子代理发消息或无限嵌套生成子代理。一行代码解决权限隔离，无需复杂的策略引擎。

**错误处理**: `execute()`内部catch所有异常，返回`ToolResult(is_error=True)`。Tool永远不会让Agent崩溃。

### 4.4 会话持久化 (session.py, 96行)

```python
class Session:             # key, messages, add_message, get_history, clear
class SessionManager:      # get_or_create, save, delete (JSONL + 内存缓存)
```

**JSONL格式**:
```
{"_type":"metadata","key":"tg:123","created_at":"...","updated_at":"..."}
{"role":"user","content":"hello","timestamp":"..."}
{"role":"assistant","content":"hi","timestamp":"..."}
```

**设计决策**:
- `get_or_create`不区分新建和恢复 -- 空Session就是新Session
- 内存缓存避免重复磁盘读取
- 损坏文件返回空Session而非抛异常（优雅降级）

### 4.5 双层记忆 (memory.py, 59行)

```
memory/
├── MEMORY.md          # 长期记忆：用户偏好、项目事实
└── 2026-02-06.md      # 每日笔记：当天的临时信息
```

- `get_context()` -- 返回拼合后的记忆文本，注入system prompt
- `write_memory()` / `append_memory()` -- 长期记忆读写
- `read_daily()` / `append_daily()` -- 每日笔记读写

Agent通过内置工具（write_file）修改记忆文件。记忆系统本身是被动的 -- 只负责读写，不做智能判断。

### 4.6 渐进式技能 (skills.py, 103行)

```python
class SkillsLoader:        # load_all, get_always_skills, build_summary
```

SKILL.md格式（YAML frontmatter + Markdown body）:
```yaml
---
name: git-commit
description: Create conventional commits
metadata: '{"nanobot":{"always":false,"requires":{"bins":["git"]}}}'
---
[技能正文...]
```

**渐进加载策略** (Nanobot的核心创新):
- `always=true`的技能: 完整body注入system prompt
- 其他技能: 只注入XML摘要（名称+描述+路径），Agent按需用`read_file`加载

这避免了所有技能同时占满上下文窗口。

**需求检查**: 技能声明`requires_bins`和`requires_env`，缺失则跳过加载，无报错。

### 4.7 上下文构建 (context.py, 97行)

4层可组合的系统提示（源自OpenClaw `system-prompt.ts`的buildSystemPrompt）:

```
Layer 1: 身份文件     IDENTITY.md, AGENTS.md, SOUL.md, USER.md, TOOLS.md
Layer 2: 运行时上下文  当前时间, 当前会话标识
Layer 3: 记忆         MEMORY.md + 当日笔记
Layer 4: 技能         always技能全文 + 其他技能XML摘要
```

各层用`---`分隔，拼成一个system message。加上Session历史消息和当前用户消息，构成完整的`messages`列表传给LLM。

**多模态支持**: 如果`Envelope.media`非空，将图片base64编码为OpenAI Vision格式的content数组。

### 4.8 LLM提供者 (provider.py, 101行)

```python
class LLMProvider(ABC):    # chat(messages, tools, ...) → LLMResponse
class LiteLLMProvider:     # LiteLLM acompletion 封装
```

- 自动检测API Key类型（`sk-or-`前缀 → OpenRouter，模型名含`anthropic` → Anthropic）
- JSON解析tool_calls的arguments，malformed JSON回退为`{"raw": "..."}`而非崩溃
- LLM调用失败返回`LLMResponse(content="LLM error: ...", finish_reason="error")`

### 4.9 Agent循环 (agent.py, 106行)

```python
class AgentLoop:
    async def run(self):       # 主循环: consume → _process → publish
    async def _process(self):  # LLM+Tool迭代 → final Envelope
```

`_process`的完整流程:
1. `SessionManager.get_or_create` -- 加载或创建会话
2. `SpawnTool.set_origin` -- 设置子代理回传路由
3. `ContextBuilder.build` -- 组装system+history+user消息
4. **迭代循环**: `Provider.chat` → 有tool_calls就执行并追加结果 → 无tool_calls就break
5. `Session.save` -- 持久化用户消息和最终回复
6. 返回outbound `Envelope`

### 4.10 子代理 (subagent.py, 96行)

```python
class SubagentManager:
    async def spawn(task, label, origin_channel, origin_chat_id) → task_id
```

- `asyncio.create_task`后台运行，不阻塞主Agent
- **工具隔离**: `deny=["message", "spawn"]` -- 子代理不能发消息、不能再生子代理
- 完成后通过`bus.publish_inbound`将结果发回主Agent（channel="system"）
- `done_callback`自动清理已完成的task引用

### 4.11 Channel基类 (channel.py, 56行)

```python
class BaseChannel(ABC):
    async def start(self)              # 启动（如polling/webhook）
    async def stop(self)               # 停止
    async def send(self, envelope)     # 发送outbound消息
    def is_allowed(self, sender_id)    # 白名单检查
    async def _handle_incoming(...)    # 权限过滤 → publish_inbound
```

**白名单**: `allow_from`为空则允许所有人。支持复合ID（`user|team|group`中任一部分匹配即可）。

### 4.12 组装根 (app.py, 134行)

```python
class NiBot:
    def __init__(config_path)   # 创建所有组件
    def add_channel(channel)    # 注册Channel + 订阅outbound
    def add_tool(tool)          # 注册自定义Tool
    async def run()             # 注册内置工具 → 加载技能 → 启动所有协程
```

**内置工具自动注册**: `_register_builtin_tools()`注册9个内置工具，用户通过`add_tool()`注册的同名工具优先（`if not registry.has(tool.name)`）。

**Provider自动选择**: 根据model名称自动匹配provider配置（anthropic/deepseek/openrouter/openai）。

---

## 五、内置工具

| 工具 | 文件 | 功能 | 安全机制 |
|------|------|------|----------|
| `read_file` | file_tools.py | 读文件（分页） | `Path.is_relative_to`工作区限制 |
| `write_file` | file_tools.py | 写文件（自动建目录） | 同上 |
| `edit_file` | file_tools.py | 精确替换（唯一匹配） | 同上 |
| `list_dir` | file_tools.py | 目录列表 | 同上 |
| `exec` | exec_tool.py | Shell执行 | 危险命令正则黑名单 + 超时 |
| `web_search` | web_tools.py | Brave Search API | API Key配置检查 |
| `web_fetch` | web_tools.py | 抓取URL内容 | readability提取正文 |
| `message` | message_tool.py | 跨Channel发消息 | 子代理deny-list隔离 |
| `spawn` | spawn_tool.py | 生成子代理 | 子代理deny-list隔离 |

**路径安全**: `_resolve_path`使用`Path.is_relative_to()`而非字符串前缀检查，防止`/tmp/work_evil`绕过`/tmp/work`限制。`restrict_to_workspace`配置项控制是否启用限制。

**命令安全**: `DANGEROUS_PATTERNS`正则匹配`rm -rf`、`format`、`dd if=`、`shutdown`、`mkfs`、`fork bomb`、`chmod -R 777`等。

---

## 六、配置系统

```python
class NiBotConfig(BaseSettings):     # Pydantic Settings
    agent: AgentConfig               # name, model, max_tokens, temperature, max_iterations, workspace
    channels: ChannelsConfig         # telegram, feishu (各有enabled/token/allow_from)
    providers: ProvidersConfig       # anthropic, openai, openrouter, deepseek (各有api_key/api_base)
    tools: ToolsConfig               # restrict_to_workspace, exec_timeout, web_search_api_key
```

**配置优先级**: 环境变量 > JSON文件 > 默认值

- JSON文件: `~/.nibot/config.json`（支持camelCase自动转snake_case）
- 环境变量: `NIBOT_AGENT__MODEL=anthropic/claude-opus-4-6`（双下划线分隔嵌套）

---

## 七、扩展指南

### 添加自定义Tool

```python
from nibot import NiBot, Tool
from typing import Any

class MyTool(Tool):
    @property
    def name(self) -> str: return "my_tool"
    @property
    def description(self) -> str: return "Does something"
    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
    async def execute(self, **kwargs) -> str:
        return f"Result: {kwargs['input']}"

app = NiBot()
app.add_tool(MyTool())
```

### 添加自定义Channel

```python
from nibot import BaseChannel, Envelope

class SlackChannel(BaseChannel):
    name = "slack"
    async def start(self): ...
    async def stop(self): ...
    async def send(self, envelope: Envelope): ...

app.add_channel(SlackChannel(config, app.bus))
```

### 添加Skill

在workspace的`skills/my-skill/SKILL.md`中创建:

```yaml
---
name: my-skill
description: What this skill does
metadata: '{"nanobot":{"always":false,"requires":{"bins":["git"]}}}'
---
# Skill Instructions
[Markdown内容，Agent按需读取]
```

---

## 八、文件清单

```
nibot/                          1771行
├── types.py            (66)    5个核心dataclass
├── log.py              (8)     loguru配置
├── config.py           (104)   Pydantic Settings + JSON加载
├── bus.py              (48)    MessageBus (asyncio.Queue双队列)
├── registry.py         (70)    Tool ABC + ToolRegistry + deny-list
├── session.py          (96)    Session + SessionManager (JSONL+缓存)
├── memory.py           (59)    MemoryStore (MEMORY.md + 每日笔记)
├── skills.py           (103)   SkillsLoader (渐进加载)
├── provider.py         (101)   LLMProvider ABC + LiteLLMProvider
├── context.py          (97)    ContextBuilder (4层组合系统提示)
├── channel.py          (56)    BaseChannel ABC + 白名单
├── agent.py            (106)   AgentLoop (LLM+Tool迭代)
├── subagent.py         (96)    SubagentManager (后台任务+工具隔离)
├── app.py              (134)   NiBot Composition Root
├── __main__.py         (33)    CLI入口 + 自动接线Channel
├── __init__.py         (20)    公开API导出
├── tools/
│   ├── file_tools.py   (164)   read/write/edit/list_dir + 路径沙箱
│   ├── exec_tool.py    (82)    Shell执行 + 危险命令拦截
│   ├── web_tools.py    (104)   search (Brave API) + fetch
│   ├── message_tool.py (45)    跨Channel消息
│   └── spawn_tool.py   (47)    子代理生成
└── channels/
    ├── telegram.py     (62)    python-telegram-bot集成
    └── feishu.py       (70)    lark-oapi集成
```

---

## 九、已验证的安全修复

经Codex审查发现并修复的5个问题:

| # | 严重度 | 问题 | 修复 |
|---|--------|------|------|
| 1 | 高 | `_resolve_path`字符串前缀可被同名前缀目录绕过 | `Path.is_relative_to()` |
| 2 | 高 | `SpawnTool.origin`始终为空，子代理结果无法路由 | `AgentLoop._process`中调用`set_origin()` |
| 3 | 高 | `load_config`读到JSON后绕过环境变量覆盖 | `NiBotConfig(**file_data)`让BaseSettings合并env |
| 4 | 中 | CLI未自动接线Channel，启动后收不到消息 | `__main__.py`添加`_auto_add_channels()` |
| 5 | 低 | `restrict_to_workspace`配置是死代码 | file_tools接收`restrict`参数并传入 |

---

## 十、测试覆盖

39个测试用例，覆盖全部13个模块:

- **types**: Envelope默认值、LLMResponse.has_tool_calls
- **config**: camelCase转换、JSON加载、损坏文件降级、默认值
- **bus**: 入站发布/消费、出站分派/停止、未订阅Channel忽略
- **registry**: 注册/定义/执行、deny-list过滤、未知工具/异常处理
- **session**: 添加消息/历史截取/清除、保存/加载/删除、损坏文件降级
- **memory**: 读写/追加、每日笔记、空上下文
- **context**: 4层系统提示组装、多模态用户内容
- **file_tools**: 读写编辑列目录、路径沙箱、**同名前缀攻击防御**
- **exec_tool**: 正常命令、多种危险模式拦截、超时
- **channel**: 空白名单/精确匹配/复合ID、入站发布/拒绝
- **provider**: API Key配置、tool_calls解析（含malformed JSON）
- **agent**: 无工具直接回复、工具循环、max_iterations边界
