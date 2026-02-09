# Inspiration Projects

NiBot v1.2+ 方向参考。记录值得借鉴的开源项目和技术文章。

---

## OpenClaw

- **仓库**: github.com/openclaw/openclaw
- **定位**: 本地优先个人 AI 助理平台，177k stars
- **核心架构**: WebSocket Hub + 12+ 消息渠道 + 会话级 Docker 沙箱

**NiBot 可借鉴**:

| 方向 | 说明 |
|------|------|
| 工具沙箱化 | 高危工具（exec/shell）在独立容器中执行，公开渠道安全隔离 |
| Agent 间双向通信 | `sessions_send` 机制，子 Agent 可主动通知父 Agent |
| 群组 @提及门控 | 群聊中只在 @bot 时激活，避免噪音 |

---

## Nanobot

- **仓库**: github.com/HKUDS/nanobot
- **定位**: 极简 AI 助手，3448 行代码
- **核心架构**: Provider Registry + 消息总线 + 可插拔技能

**NiBot 可借鉴**:

| 方向 | 说明 |
|------|------|
| 代码极简原则 | 审查 NiBot 23 个工具中哪些可合并 |
| Provider Registry | 统一的注册/发现机制，热插拔 Provider |

---

## Pi-Mono

- **仓库**: github.com/badlogic/pi-mono
- **作者**: Mario Zechner
- **定位**: AI Agent 基础设施全栈 Monorepo
- **核心架构**: 7 层分离，只有 Read/Write/Edit/Bash 4 个核心工具

**NiBot 可借鉴**:

| 方向 | 说明 |
|------|------|
| Extension 系统 | Agent 自己写代码扩展能力，无需预定义工具 |
| 会话分支树 | 子 Agent 实验在隔离分支中进行，不污染主会话 |
| 最小核心工具哲学 | 4 个核心工具 + AI 按需生成其余功能 |

---

## Armin Ronacher 的 Pi 文章

- **来源**: lucumr.pocoo.org/2026/1/31/pi/
- **作者**: Flask/Jinja2 创造者

**核心观点**: "工具膨胀是反模式，让 AI 按需生成代码实现功能。"

**NiBot 可借鉴**:

| 方向 | 说明 |
|------|------|
| 技能系统进化 | 从静态 SKILL.md 到 Agent 自生成可执行脚本 |
| 工具精简 | 重新审视 23 个工具：哪些是核心（Read/Write/Exec），哪些应变为 AI 生成的技能 |

---

## v1.2+ 综合优先级

| 优先级 | 方向 | 来源 | 说明 |
|--------|------|------|------|
| 高 | 子 Agent 分模型（Codex） | 用户需求 | coder/tester/reviewer 用 Codex，主脑用 Opus |
| 高 | Agent 自生成技能 | Pi/Ronacher | 从 SKILL.md 到可执行脚本 |
| 中 | 工具沙箱化 | OpenClaw/Nanobot | 高危工具独立容器执行 |
| 中 | 管理面板增强 | 用户需求 | 支持通过 UI 修改配置 |
| 低 | Agent 间双向通信 | OpenClaw | SubagentManager 加 notify_parent |
| 低 | 会话分支树 | Pi-Mono | 子 Agent 实验不污染主会话 |
