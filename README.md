# 抑郁症辅助对话系统

一个用于心理支持场景的 AI 对话系统。项目提供 Vue 3 前端、FastAPI 后端、用户认证、知情同意、结构化会话流程、风险提示、管理员监控和轻量级 RAG 知识检索。

线上演示：[https://depression-system-eight.vercel.app](https://depression-system-eight.vercel.app)

> 本项目仅用于心理健康辅助与技术演示，不能替代专业诊断、治疗或危机干预。如出现自伤、自杀或其他紧急风险，应立即联系当地急救服务、危机热线或线下专业人员。

## 主要功能

- **结构化多轮对话**：通过热身、主题推进和会话收束组织心理支持流程。
- **风险识别**：识别对话中的风险信号，并给出安全边界提示。
- **Mini RAG**：基于本地心理健康知识库检索相关内容，支持可选向量检索和重排序。
- **用户系统**：支持注册、登录、签名 Token 和知情同意版本控制。
- **会话持久化**：使用 SQLite 保存用户、会话、消息、主题状态和风险结果。
- **监控面板**：展示会话状态、风险等级、热身进度和主题覆盖情况。
- **研究管线**：保留合成对话、评分蒸馏、SFT 数据转换、训练配置和本地评测脚本。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 前端 | Vue 3, TypeScript, Vite, Pinia, Ant Design Vue |
| 后端 | Python, FastAPI, Pydantic, SQLite |
| LLM | DeepSeek API，本地 fallback 回复 |
| RAG | SQLite FTS5, jieba, sentence-transformers / FAISS 可选 |
| 部署 | Vercel Static + Python Serverless Function |

## 快速开始

安装后端依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

启动后端：

```bash
PYTHONPATH=src python -m uvicorn product_app.main:app --host 127.0.0.1 --port 8000
```

启动前端：

```bash
npm --prefix frontend install
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

没有 `DEEPSEEK_API_KEY` 时，系统会使用 fallback 回复，便于本地开发和演示。

## 常用配置

| 变量 | 说明 |
| --- | --- |
| `APP_SECRET` | Token 签名密钥，生产环境必须修改 |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | 管理员账号 |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `DEEPSEEK_MODEL` | DeepSeek 模型名 |
| `PRODUCT_DATABASE_URL` / `PRODUCT_DB_PATH` | 数据库地址 |
| `MINI_RAG_ENABLED` | 是否启用 Mini RAG |
| `MINI_RAG_ENABLE_EMBEDDING` / `MINI_RAG_ENABLE_RERANK` | 是否启用语义检索和重排序 |

## RAG 知识库

知识源位于 `data/knowledge/`，默认索引为 `data/knowledge_index.db`。

```bash
python scripts/build_knowledge_index.py --rebuild
```

启用可选向量检索：

```bash
pip install -r src/product_app/requirements-rag.txt
python scripts/build_knowledge_index.py --rebuild --enable-embedding --vector-store auto
```

知识库规范见 `data/knowledge/README.md`。

## 部署

项目已包含 Vercel 配置：

- `frontend/` 构建为静态资源；
- `api/index.py` 暴露 FastAPI Serverless Function；
- `/api/*` 请求转发到后端，其余请求回退到前端路由。

部署前至少修改 `APP_SECRET` 和 `ADMIN_PASSWORD`。更多说明见 `docs/DEPLOYMENT.md`。

## 目录

| 路径 | 说明 |
| --- | --- |
| `frontend/` | Vue 3 产品前端 |
| `src/product_app/` | FastAPI 产品后端 |
| `api/` | Vercel 后端入口 |
| `data/knowledge/` | Mini RAG 知识源 |
| `data/generate/` | 合成对话与 SFT 数据转换 |
| `training/` | 评分蒸馏与 LLaMA-Factory 配置 |
| `eval/` | 本地 benchmark 与评测说明 |
| `docs/` | 产品计划、实现记录和部署文档 |

## 训练与评估

研究侧流程包括合成对话生成、评分蒸馏、SFT 数据准备、LLaMA-Factory 训练配置和 benchmark v2 评测。可从 `scripts/run_unified_workflow.sh`、`training/README.md` 和 `eval/README.md` 开始。

## 参考

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [MindEval](https://github.com/SWORDHealth/mind-eval)
