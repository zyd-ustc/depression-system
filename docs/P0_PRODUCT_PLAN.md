# P0 产品化方案：心理对话协助与风险判断

## 1. 产品边界

当前 P0 定义为“心理对话协助与风险判断”，不是自动诊断、自动治疗或急救系统。

用户侧只展示：

- 对话支持；
- 风险等级：low / medium / high；
- 是否建议寻求专业帮助；
- 必要时提示联系线下紧急支持。

不向普通用户展示：

- 0-100 抑郁严重度分；
- PHQ-9 / HAM-D / MADRS / QIDS 的模型推断分；
- “确诊”“治疗方案”“用药剂量”等医疗结论。

## 2. 当前风险机制的对抗性审查

P0 沿用现有关键词机制，来源与 `eval/benchmark_v2/dataset_builder.py` 的 `HIGH_RISK_KEYWORDS`、`MEDIUM_RISK_KEYWORDS` 保持一致：

- high：自杀、自残、轻生、结束生命、不想活、活着没意思；
- medium：绝望、消极、无望、崩溃、抑郁、严重焦虑。

本次产品壳在 `src/product_app/risk.py` 中增加了最小归一化：去掉空格、常见标点、零宽字符，用于命中“我 想 自 杀”一类简单规避。

结论：

- 可用于 P0 的“明显风险触发”；
- 不足以独立承担正式危机识别；
- 对隐喻、代称、反讽、拼音、谐音、图片输入、长上下文风险累积不稳定；
- 上线试点前必须把它作为“第一层规则”，再叠加模型分类器和人工复核。

可运行审查：

```bash
PYTHONPATH=src python scripts/p0_adversarial_audit.py
```

其中 `missed_euphemism`、`missed_codeword` 预期会暴露关键词机制的漏检风险。

## 3. 登录与隐私同意

P0 已新增极简产品应用：

- `src/product_app/main.py`：FastAPI 后端；
- `src/product_app/static/`：黑白极简前端；
- SQLite 数据库默认写入 `data/product_app.sqlite3`；
- 注册、登录、token session；
- 隐私同意版本记录；
- 未同意用户禁止调用 `/api/chat`；
- 政策正文先留白，只保留流程和版本字段。

数据库表：

- `users`：用户名、密码哈希、同意版本、同意时间；
- `sessions`：登录 token；
- `conversations`：会话；
- `messages`：用户与助手消息、风险等级。

P0 暂不做：

- 真实身份认证；
- 手机号/邮箱验证；
- 多租户；
- 管理后台；
- 数据删除工作流；
- 权限分级。

## 4. DeepSeek Prompt 与 JSON 合法性

模型训练暂不进入 P0。产品对话通过 DeepSeek OpenAI-compatible API 调用。

默认配置：

- `DEEPSEEK_BASE_URL=https://api.deepseek.com`
- `DEEPSEEK_MODEL=deepseek-v4-flash`

环境变量可覆盖。

Prompt 要求模型只输出合法 JSON：

```json
{
  "assistant_reply": ""
}
```

后端会用 Pydantic 校验 JSON。若 DeepSeek 未配置、返回非法 JSON、或调用失败，则进入 fallback 回复。

高风险消息不交给模型自由生成，直接使用本地高风险回复。

## 5. RAG 数据处理审查

当前 RAG 代码可作为 P0 后续知识增强基础，但不建议直接默认开启。

已发现并修补：

- `Data_process` 缺少 `rerank` 方法；
- `rag/src/pipeline.py` 调用 rerank 参数顺序不对。

仍需注意：

- `FAISS.load_local` 依赖 pickle 反序列化，生产环境只能加载可信向量库；
- RAG 文档来源需要版本号、标题、发布日期和人工审核状态；
- 用户对话不能直接写入知识库；
- 检索内容不能覆盖风险规则；
- 回复中应区分“知识库资料”与“系统建议”。

P0 建议先让 RAG 处于后台可测状态，不在用户端默认开启。

## 6. Score 分配策略

原来的 score 是为模型训练和 benchmark 服务的：

- `depression_score` 0-100；
- `depression_level`；
- PHQ-9 / HAM-D / MADRS / QIDS 代理分；
- teacher/student 标签来源。

产品侧建议去专业化、降复杂度：

- 用户界面只展示 low / medium / high 风险；
- 不展示临床量表分数；
- 不把模型输出写成“诊断结果”；
- 内部可保留数值用于趋势、质检和评测；
- 真正面向用户的语言应是“建议寻求专业帮助”“建议记录睡眠和情绪变化”，而不是“你是中度抑郁”。

## 7. P0 验收标准

- 用户能注册、登录；
- 用户必须经过隐私同意流程才能对话；
- 普通输入能得到支持性回复；
- 明显自伤/自杀关键词会触发 high；
- DeepSeek 返回 JSON 能被校验；
- DeepSeek 不可用时系统仍可 fallback；
- RAG rerank 不再因缺方法直接报错；
- 有部署文档和一键启动脚本。
