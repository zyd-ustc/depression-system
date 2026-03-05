from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_tools_api import choice, fmt, read_json


@dataclass
class Bias:
    id: int
    name: str
    types: List[str]
    desc: str = ""
    example: str = ""


@dataclass
class DialogueState:
    bias: Bias
    event: str
    people: str
    thought: str
    emotion: str
    intensity: int
    behavior: str
    evidence_for: str
    evidence_against: str
    alternative: str
    action: str
    # 用于输出标签
    tags: Dict[str, Any] = field(default_factory=dict)


def load_biases(prompts_dir: str | Path) -> List[Bias]:
    p = Path(prompts_dir) / "cognitive_bias" / "biases.json"
    raw = read_json(p)
    biases: List[Bias] = []
    for item in raw:
        biases.append(
            Bias(
                id=int(item.get("id")),
                name=str(item.get("name")),
                types=list(item.get("types") or []),
                desc=str(item.get("desc") or ""),
                example=str(item.get("example") or ""),
            )
        )
    return biases


def _pick_person_and_experience(keyword_json: Dict[str, Any], rng: random.Random) -> Tuple[str, str, str]:
    time = choice(rng, keyword_json.get("time", ["最近"]))
    people_dict = keyword_json.get("people", {}) or {}
    exp_dict = keyword_json.get("experience", {}) or {}

    # 选择一个 group 作为“人/经历”的对应域
    groups = [g for g in people_dict.keys() if g in exp_dict]
    if not groups:
        # fallback：任意一个 people group
        groups = list(people_dict.keys()) or ["自己"]

    group = choice(rng, groups)
    people = choice(rng, people_dict.get(group, ["我"]))
    exp = choice(rng, exp_dict.get(group, ["发生了一些不顺心的事"]))
    return time, people, exp


def build_state(
    patient_template: Dict[str, Any],
    prompts_dir: str | Path,
    rng: random.Random,
) -> DialogueState:
    # 1) 选偏差
    biases = load_biases(prompts_dir)
    bias = choice(rng, biases)

    # 2) 选触发事件：基于 age/gender 关键词库
    age = int(patient_template.get("年龄", 30))
    gender = patient_template.get("性别", "男")
    if gender == "男":
        if age <= 20:
            kw_file = "male_20.json"
        elif age <= 50:
            kw_file = "male_adult.json"
        else:
            kw_file = "male_old.json"
    else:
        if age <= 20:
            kw_file = "female_20.json"
        elif age <= 50:
            kw_file = "female_adult.json"
        else:
            kw_file = "female_old.json"

    kw_path = Path(prompts_dir) / "patient" / kw_file
    keyword_json = read_json(kw_path)
    time, people, exp = _pick_person_and_experience(keyword_json, rng)
    # 简单把关键词拼成事件描述
    event = f"{time}的时候，和{people}有关，发生了{exp}"

    # 3) 生成“自动想法/证据/替代想法/行动”占位内容（模板化）
    pt = read_json(Path(prompts_dir) / "cognitive_bias" / "patient_templates.json")
    bias_thoughts = (pt.get("bias_thoughts") or {}).get(bias.name) or []
    thought = choice(rng, bias_thoughts) if bias_thoughts else "我当时就觉得一切都完了。"

    emotion = choice(rng, ["难过", "焦虑", "绝望", "内疚", "烦躁"])
    intensity = rng.randint(6, 10)
    behavior = choice(
        rng,
        [
            "我就开始回避人，能不见就不见。",
            "我把事情拖着不做，越拖越怕。",
            "我那阵子会忍不住哭，整个人很没劲。",
            "我会突然发脾气，然后更自责。",
        ],
    )
    evidence_for = choice(rng, ["他们没回复我消息", "我最近确实表现很差", "之前也发生过类似的事"])
    evidence_against = choice(rng, ["也许对方只是忙", "也有人对我挺好的", "我并没有掌握全部信息"])
    alternative = choice(rng, ["事情可能没有我想的那么绝对", "我可以先把事实弄清楚再下结论", "我现在的情绪在放大问题"])
    action = choice(rng, ["先问清楚一次", "把任务拆成最小一步先做5分钟", "今天先出门走10分钟并记录感受"])

    return DialogueState(
        bias=bias,
        event=event,
        people=str(people),
        thought=thought,
        emotion=emotion,
        intensity=intensity,
        behavior=behavior,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        alternative=alternative,
        action=action,
        tags={
            "patient_id": patient_template.get("患者"),
            "diagnosis": patient_template.get("诊断结果"),
        },
    )


def doctor_questions(prompts_dir: str | Path) -> List[Dict[str, Any]]:
    dt = read_json(Path(prompts_dir) / "cognitive_bias" / "doctor_templates.json")
    return list(dt.get("stages") or [])


def doctor_ending(prompts_dir: str | Path, rng: random.Random) -> str:
    dt = read_json(Path(prompts_dir) / "cognitive_bias" / "doctor_templates.json")
    return choice(rng, dt.get("endings") or ["诊断结束。"])


def patient_reply(stage_name: str, state: DialogueState, prompts_dir: str | Path, rng: random.Random) -> str:
    pt = read_json(Path(prompts_dir) / "cognitive_bias" / "patient_templates.json")
    default = pt.get("default") or {}
    bias_thoughts = (pt.get("bias_thoughts") or {}).get(state.bias.name) or []

    if stage_name in ("开场", "澄清事件"):
        return fmt(choice(rng, default.get("event", ["{event}"])), event=state.event)
    if stage_name == "自动想法":
        if bias_thoughts and rng.random() < 0.7:
            return choice(rng, bias_thoughts)
        return state.thought
    if stage_name == "情绪强度":
        return fmt(choice(rng, default.get("emotion", ["{intensity}/10"])), intensity=state.intensity)
    if stage_name == "行为反应":
        return state.behavior
    if stage_name == "识别偏差":
        # 病人不需要“承认”，只表达感受/疑惑
        return choice(rng, ["听起来像是…我也说不清，但我脑子里就会自动这么想。", "可能是吧，我一紧张就会往坏处想。"])
    if stage_name == "证据支持":
        return fmt(choice(rng, default.get("evidence_for", ["{evidence_for}"])), evidence_for=state.evidence_for)
    if stage_name == "证据反对":
        return fmt(choice(rng, default.get("evidence_against", ["{evidence_against}"])), evidence_against=state.evidence_against)
    if stage_name == "替代想法":
        return fmt(choice(rng, default.get("alternative", ["{alternative}"])), alternative=state.alternative)
    if stage_name == "行为实验":
        return choice(rng, [f"我可以试试：{state.action}。", f"我愿意先做一个小的：{state.action}。"])
    if stage_name == "总结":
        return choice(rng, ["嗯，这样总结我能理解。", "听你这么说，我感觉没那么绝望了。"])
    return choice(rng, ["嗯。", "我明白。"])


def render_doctor(stage_name: str, stage_questions: List[str], state: DialogueState, rng: random.Random) -> str:
    q = choice(rng, stage_questions)
    return fmt(
        q,
        bias_name=state.bias.name,
        bias_types="、".join(state.bias.types),
        event=state.event,
        thought=state.thought,
        alternative=state.alternative,
        diagnosis=state.tags.get("diagnosis", ""),
    )


