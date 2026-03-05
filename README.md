# 抑郁症对话系统（完整流程版）

本项目覆盖从数据合成、蒸馏评分、SFT 数据构建、模型训练、本地推理到评测的完整链路。

你可以按下面步骤从零跑通，或者直接执行一键脚本。

## 1. 项目结构

```text
depression_system/
├── data/
│   └── generate/
│       ├── main.py                     # 生成医生-患者多轮对话
│       ├── data_convert.py             # 对话转 SFT 样本（带蒸馏元数据）
│       ├── cards.jsonl                 # cards 模式患者画像
│       ├── DataSynCards/dialogues.jsonl
│       └── processed/
│           ├── distilled_scores.jsonl  # 蒸馏评分结果
│           └── processed_dialogues.json
├── training/
│   ├── score_distill/                  # DeepSeek + 学生模型蒸馏
│   ├── sft_config.yaml                 # LLaMA-Factory SFT 配置
│   └── dataset_info.json
├── src/inference_pipeline/             # Gradio 推理界面
├── eval/benchmark_v2/                  # 本地评测框架
└── scripts/run_unified_workflow.sh     # 一键串联流程
```

## 2. 环境准备

推荐 Python 3.10+。

### 2.1 安装基础依赖

```bash
pip install -r src/inference_pipeline/requirements_ui.txt
pip install -r training/score_distill/requirements.txt
```

### 2.2 配置 API Key（可选但推荐）

蒸馏阶段如果配置了 key，会调用 DeepSeek 作为 teacher；否则走本地启发式 teacher（可跑通但质量较弱）。

```bash
export DEEPSEEK_API_KEY="你的key"
# 可选：自定义兼容接口地址
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
```

## 3. 步骤一：生成对话数据

在仓库根目录执行：

```bash
cd data/generate
python main.py
cd ../..
```

预期输出：

- `data/generate/DataSynCards/dialogues.jsonl`

## 4. 步骤二：蒸馏抑郁评分（DeepSeek + 学生模型）

```bash
python -m training.score_distill.run_distill \
  --input-dialogues data/generate/DataSynCards/dialogues.jsonl \
  --output-labels data/generate/processed/distilled_scores.jsonl \
  --sample-ratio 0.1 \
  --epochs 30 \
  --batch-size 32
```

预期输出：

- `data/generate/processed/distilled_scores.jsonl`
- `training/score_distill/artifacts/student_model.pt`
- `training/score_distill/artifacts/distill_summary.json`

`distilled_scores.jsonl` 每条会包含：

- `depression_score`
- `depression_level`
- `emotion_scores`
- `scales`（`PHQ-9` / `HAM-D` / `MADRS` / `QIDS`）
- `label_source` / `teacher_score`

## 5. 步骤三：构建 SFT 训练数据（自动注入评分元数据）

```bash
python data/generate/data_convert.py
```

预期输出：

- `data/generate/processed/processed_dialogues.json`

这个文件里的每个样本包含 `metadata`，会透传蒸馏评分、情绪和量表字段，避免训练和评测标签口径割裂。

## 6. 步骤四：在 LLaMA-Factory 训练

本仓库只提供配置，不内置 LLaMA-Factory 框架本体。

### 6.1 准备数据

1. 将 `data/generate/processed/processed_dialogues.json` 复制到 LLaMA-Factory 的 `data/depression.json`。  
2. 将 `training/dataset_info.json` 合并进 LLaMA-Factory 的 `data/dataset_info.json`。  
3. 使用 `training/sft_config.yaml` 作为训练配置（按你的模型路径修改 `model_name_or_path`）。

### 6.2 启动训练

```bash
cd /path/to/LLaMA-Factory
llamafactory-cli train /path/to/depression_system/training/sft_config.yaml
```

## 7. 步骤五：本地推理（Gradio UI）

```bash
export MODEL_ID="你的微调模型路径或HF模型ID"
bash src/inference_pipeline/run_gradio.sh
```

打开：

- `http://localhost:7860`

## 8. 步骤六：评测（本地 benchmark v2）

### 8.1 构建评测样本

优先使用蒸馏标签：

```bash
python -m eval.benchmark_v2.dataset_builder \
  --source auto \
  --distilled-path data/generate/processed/distilled_scores.jsonl \
  --output-samples eval/benchmark_v2/data/samples.v1.json \
  --output-counterfactuals eval/benchmark_v2/data/counterfactuals.v1.json
```

### 8.2 运行评测

Mock 模式（先验证流程）：

```bash
python -m eval.benchmark_v2.run_benchmark_v2 \
  --samples eval/benchmark_v2/data/samples.v1.json \
  --counterfactuals eval/benchmark_v2/data/counterfactuals.v1.json \
  --mode mock \
  --repeats 3 \
  --output-dir eval/benchmark_v2/outputs
```

闭源模型模式（配置文件）：

```bash
python -m eval.benchmark_v2.run_benchmark_v2 \
  --config eval/benchmark_v2/configs/closed_source.v1.json \
  --samples eval/benchmark_v2/data/samples.v1.json \
  --counterfactuals eval/benchmark_v2/data/counterfactuals.v1.json \
  --repeats 3 \
  --output-dir eval/benchmark_v2/outputs
```

预期输出：

- `eval/benchmark_v2/outputs/results_raw.jsonl`
- `eval/benchmark_v2/outputs/summary.json`

## 9. 一键执行完整流程

如果你已经准备好依赖和 key，可以直接执行：

```bash
bash scripts/run_unified_workflow.sh
```

该脚本会顺序执行：

1. 蒸馏评分
2. SFT 转换
3. benchmark 样本构建
4. benchmark 运行

## 10. 常见问题

### 10.1 没有网络或无法访问 HuggingFace

蒸馏模块会自动回退到哈希特征，不会中断流程；但评分质量会低于真实 RoBERTa 特征。

### 10.2 没有配置 `DEEPSEEK_API_KEY`

会使用启发式 teacher，流程可跑通，但建议在正式数据生产时配置 key。

### 10.3 `data_convert.py` 里评分字段为空

先确认你已成功执行步骤二，且 `data/generate/processed/distilled_scores.jsonl` 存在。

### 10.4 评测结果不稳定

固定 `--repeats`、随机种子和样本文件，避免每次重采样导致排名漂移。

## 11. 参考

- LLaMA-Factory: `https://github.com/hiyouga/LLaMA-Factory`
- MindEval: `https://github.com/SWORDHealth/mind-eval`
