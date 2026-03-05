from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TurnSample:
    source_idx: int
    turn_index: int
    instruction: str
    doctor_reply: str
    dialogue_window: str

    @property
    def key(self) -> str:
        return f"{self.source_idx}:{self.turn_index}"


def load_dialogues(path: str | Path) -> list[dict]:
    data_path = Path(path)
    rows: list[dict] = []
    with data_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_turn_samples(dialogues: list[dict], max_turns: int | None = None) -> list[TurnSample]:
    samples: list[TurnSample] = []
    for source_idx, item in enumerate(dialogues):
        conversation = item.get("conversation") or []
        if not isinstance(conversation, list):
            continue
        doctors: list[str | None] = []
        patients: list[str | None] = []
        for turn in conversation:
            if not isinstance(turn, dict):
                doctors.append(None)
                patients.append(None)
                continue
            doctors.append((turn.get("doctor") or "").strip() or None)
            patients.append((turn.get("patient") or "").strip() or None)

        for turn_index in range(1, len(conversation)):
            instruction = patients[turn_index - 1]
            doctor_reply = doctors[turn_index]
            if not instruction or not doctor_reply:
                continue
            if max_turns is not None and turn_index > max_turns:
                continue
            history_fragments: list[str] = []
            for j in range(0, turn_index):
                if doctors[j]:
                    history_fragments.append(f"医生：{doctors[j]}")
                if patients[j]:
                    history_fragments.append(f"患者：{patients[j]}")
            dialogue_window = "\n".join(history_fragments[-8:])
            samples.append(
                TurnSample(
                    source_idx=source_idx,
                    turn_index=turn_index,
                    instruction=instruction,
                    doctor_reply=doctor_reply,
                    dialogue_window=dialogue_window,
                )
            )
    return samples


def write_distilled_labels(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_distilled_labels(path: str | Path) -> dict[str, dict]:
    labels_path = Path(path)
    if not labels_path.exists():
        return {}
    rows: dict[str, dict] = {}
    with labels_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows[str(payload["key"])] = payload
    return rows

