# P0 当前实现记录

更新时间：2026-07-02

## 1. 当前产品形态

P0 当前定位为“心理对话协助 + 风险判断 + 话题覆盖 + 自然结束报告”的最小可用产品。

当前主入口：

- `frontend/`：Vue 3 + Pinia + Element Plus 前端；
- `api/index.py`：Vercel Python 函数入口；
- `src/product_app/`：FastAPI 产品后端；
- `vercel.json`：前端静态构建与 `/api/*` 后端转发；
- `requirements.txt`：Vercel 后端依赖。

旧的 `src/product_app/static/` 仍保留，但当前 Vercel demo 和本地开发以 `frontend/` 为准。

## 2. 用户流程

已完成：

- 注册；
- 登录；
- token 鉴权；
- 隐私同意必要流程；
- 对话；
- 风险判断；
- 话题状态展示；
- 模型后端状态展示；
- 对话结束判断与结束报告。

隐私政策正文仍是占位，当前只保证流程存在。

## 3. 数据持久化

SQLite 默认位置：

- 本地：`data/product_app.sqlite3`；
- Vercel：`/tmp/product_app.sqlite3`。

核心表：

- `users`：用户名、密码哈希、同意版本和同意时间；
- `conversations`：会话时间与 `topic_state`；
- `messages`：用户和助手消息；
- `sessions`：旧 session 表保留，当前 token 主要使用签名 token。

`topic_state` 是 JSON 字符串，当前字段包括：

- `stage`：`warmup` 或 `planned`；
- `warmup_turns`：预热轮数；
- `planned_topics`：本轮咨询计划覆盖的话题；
- `covered_topics`：已经主动覆盖的话题；
- `observed_topics`：从用户内容中观察到的话题；
- `current_topic`：当前正在推进的话题；
- `session_status`：`active` 或 `ended`；
- `stop_reason`：结束原因。

## 4. 话题持久化策略

实现文件：

- `src/product_app/topics.py`

规则：

- 预热阶段至少 2 轮；
- 如果 2 轮后已经观察到话题，则进入计划阶段；
- 信息不足时最多预热到 3 轮；
- 计划阶段根据观察话题与核心话题生成 `planned_topics`；
- 下一步话题从 `planned_topics - covered_topics` 中选择；
- 用户回复当前话题后，上一轮 `current_topic` 会进入 `covered_topics`；
- 高风险时优先切到“安全与线下支持”。

注意：用户随口提到某个关键词只进入 `observed_topics`，不直接等于已经覆盖。

## 5. 对话停止逻辑

实现文件：

- `src/product_app/stop.py`

当前有两条停止路径：

1. 用户主动结束：
   - 例如“先到这里”“结束对话”“不用继续”“输出报告”；
   - 系统接受结束，不再追问新话题；
   - 输出本轮咨询报告。

2. 系统自行结束：
   - 必须处于 `planned` 阶段；
   - 必须 `planned_topics` 全部进入 `covered_topics`；
   - 满足后触发自然收束和完整结束报告。

高风险优先级高于普通结束。例如“结束生命”不会被当作“结束对话”，会继续进入安全支持流程。

结束报告要求：

- 确认本轮收束；
- 总结已覆盖主题；
- 整理主要困扰和可能触发因素；
- 给出一到两个低负担行动；
- 给出风险与线下求助边界；
- 说明之后可以继续补充。

## 6. 模型调用与 fallback

实现文件：

- `src/product_app/deepseek_client.py`

DeepSeek 使用 OpenAI-compatible API：

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_TIMEOUT_SECONDS`

如果没有 `DEEPSEEK_API_KEY`，或 SDK 初始化失败，则走本地 fallback。

前端状态栏会显示：

- `deepseek`：走 DeepSeek API；
- `fallback`：走本地 fallback；
- `deepseek / json-failed`：尝试 DeepSeek，但 JSON 解析或校验失败后 fallback。

重要修复：

- fallback 不再直接拼接内部 `prompt_instruction`；
- 用户侧不会看到“不要像问卷”这类内部提示词；
- 高风险回复不交给模型自由生成。

## 7. 风险判断

实现文件：

- `src/product_app/risk.py`

当前仍是 P0 关键词规则：

- high：自杀、自残、轻生、结束生命、不想活、活着没意思；
- medium：绝望、消极、无望、崩溃、抑郁、严重焦虑。

规则会做基础文本归一化，能覆盖简单空格和标点规避。

该机制只能作为 P0 第一层风险判断，不等于正式危机识别系统。

## 8. 前端风格

当前前端使用黑白为主的轻量咨询工作台风格：

- 标题使用非常规 serif 字体栈；
- 主色调为黑、暖白、浅灰；
- 圆角仅保留在外壳、主面板、消息、输入区等关键位置；
- 状态面板展示用户、风险、分数、话题、阶段、会话、模型、已覆盖、下一步、计划；
- 隐私同意只作为必要流程，不在页面中过度强调。

## 9. 验证命令

后端编译：

```bash
python -m compileall api src/product_app scripts/p0_adversarial_audit.py
```

前端构建：

```bash
npm run build
```

风险与 DeepSeek JSON 检查：

```bash
PYTHONPATH=src python scripts/p0_adversarial_audit.py
```

本地开发：

```bash
PYTHONPATH=src python -m uvicorn product_app.main:app --host 127.0.0.1 --port 8000
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

## 10. 当前已知限制

- Vercel `/tmp` SQLite 不适合生产持久化；
- 隐私政策正文未完成；
- 风险判断仍是关键词规则，不能替代专业危机识别；
- 停止逻辑已经具备基本条件，但报告质量依赖模型能力；
- RAG 尚未接入产品对话；
- 本地没有 `DEEPSEEK_API_KEY` 时会走 fallback，前端会显示 `fallback`。
