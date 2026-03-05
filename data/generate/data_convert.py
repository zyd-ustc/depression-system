"""Convert dialogue JSON/JSONL to SFT format (instruction, history, output, system)."""
from pathlib import Path
import json
from tqdm import tqdm

FILE_NAME = "DataSynCards/dialogues.jsonl"
OUT_FILE_NAME = "processed_dialogues.json"
DISTILL_FILE_NAME = "processed/distilled_scores.jsonl"
BASE_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """你是一位专业的心理咨询师，专门从事抑郁症的诊断和治疗。你能够提供专业、富有同理心的心理支持和指导。

你的任务是根据用户的问题和提供的相关医学文献、治疗指南等上下文信息，提供专业的抑郁症诊疗建议。

请遵循以下原则：
1. 仔细分析用户的问题和症状描述
2. 提供专业、准确、富有同理心的回复
3. 如果上下文信息不足以回答问题，请诚实说明，并建议用户咨询专业医生
4. 始终以患者的安全和福祉为首要考虑

请用中文回复，语气要温和、专业、富有同理心。"""


def load_data(in_path: Path):
    suffix = in_path.suffix.lower()
    data = []
    if suffix == ".jsonl":
        with open(in_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    elif suffix == ".json":
        with open(in_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .json or .jsonl")
    return data


def load_distilled_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows[str(payload.get("key"))] = payload
    return rows


def main():
    in_path = (BASE_DIR / FILE_NAME).resolve()
    if not in_path.exists():
        in_path = (BASE_DIR.parent / FILE_NAME).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    data = load_data(in_path)
    distill_path = (BASE_DIR / DISTILL_FILE_NAME).resolve()
    distilled = load_distilled_labels(distill_path)
    converted = []

    for source_idx, item in enumerate(tqdm(data, desc="Converting")):
        conversation = item.get("conversation") or []
        if not isinstance(conversation, list) or len(conversation) < 2:
            continue

        doctors = []
        patients = []
        for turn in conversation:
            if not isinstance(turn, dict):
                doctors.append(None)
                patients.append(None)
                continue
            doctors.append((turn.get("doctor") or "").strip() or None)
            patients.append((turn.get("patient") or "").strip() or None)

        for i in range(1, len(conversation)):
            doctor_reply = doctors[i]
            user_utter = patients[i - 1]
            if not doctor_reply or not user_utter:
                continue
            history = []
            for j in range(0, i - 1):
                u = patients[j]
                a = doctors[j + 1] if (j + 1) < len(doctors) else None
                if u and a:
                    history.append((u, a))
            key = f"{source_idx}:{i}"
            distill_info = distilled.get(key, {})
            converted.append({
                "instruction": user_utter,
                "input": "",
                "output": doctor_reply,
                "system": SYSTEM_PROMPT,
                "history": history,
                "metadata": {
                    "source_idx": source_idx,
                    "turn_index": i,
                    "distill_key": key,
                    "depression_score": distill_info.get("depression_score"),
                    "depression_level": distill_info.get("depression_level"),
                    "emotion_scores": distill_info.get("emotion_scores"),
                    "scales": distill_info.get("scales"),
                    "teacher_score": distill_info.get("teacher_score"),
                    "score_label_source": distill_info.get("label_source"),
                },
            })

    out_dir = (BASE_DIR / "processed").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_FILE_NAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=4)
    print(f"Done: {in_path} -> {out_path}, samples: {len(converted)}")


if __name__ == "__main__":
    main()
