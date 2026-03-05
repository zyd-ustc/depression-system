from doctor import Doctor
from patient import Patient
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

MODE = "cards"

CARDS_PATH = str(BASE_DIR / "cards.jsonl")
OUTPUT_DATASYN_PATH = str(BASE_DIR / "DataSynCards")
DOCTOR_MODEL_NAME = "deepseek-chat"
PATIENT_MODEL_NAME = "deepseek-chat"
MAX_TURNS = 12

DOCTOR_PROMPT_PATH = str(BASE_DIR / "prompts" / "doctor" / "doctor_persona.json")
PATIENT_INFO_PATH = str(BASE_DIR / "raw_data" / "pa20.json")
DIAGTREE_PATH = str(BASE_DIR / "prompts" / "diagtree")
NUM = 5
OUTPUT_PASTEXP_PATH = str(BASE_DIR / "prompts" / "patient" / "background_story")


def _read_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def run_cards():
    out_dir = Path(OUTPUT_DATASYN_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dialogues.jsonl"

    cards = _read_jsonl(CARDS_PATH)
    total_cost = 0.0

    with open(out_path, "w", encoding="utf-8") as wf:
        for idx, card in enumerate(tqdm(cards, desc="Generating")):
            profile = dict(card)
            profile.setdefault("gender_cn", None)

            doc = Doctor(profile, doctor_prompt_path=None, diagtree_path=None, model_path=DOCTOR_MODEL_NAME, use_api=True)
            pat = Patient(profile, model_path=PATIENT_MODEL_NAME, use_api=True, story_path=None)

            dialogue_history: list[str] = []
            conversation: list[dict] = []

            doctor_text, is_end, doctor_cost = doc.doctor_response_gen_cards(dialogue_history)
            dialogue_history.append("医生：" + doctor_text)
            conversation.append({"doctor": doctor_text})

            for _ in range(MAX_TURNS):
                patient_text, patient_cost = pat.patient_response_gen_cards(doctor_text, dialogue_history)
                dialogue_history.append("患者：" + patient_text)
                conversation[-1]["patient"] = patient_text

                doctor_text, is_end, doctor_cost = doc.doctor_response_gen_cards(dialogue_history)
                if is_end:
                    conversation.append({"doctor": doctor_text})
                    dialogue_history.append("医生：" + doctor_text)
                    break
                conversation.append({"doctor": doctor_text})
                dialogue_history.append("医生：" + doctor_text)

            total_cost = float(doctor_cost) + float(patient_cost)
            record = {
                "conversation": conversation,
                "meta": {
                    "source": "cards.jsonl",
                    "idx": idx,
                    "seed_scenario": (card.get("metadata") or {}).get("seed_scenario", ""),
                    "model_doctor": DOCTOR_MODEL_NAME,
                    "model_patient": PATIENT_MODEL_NAME,
                    "cognitive_bias": card.get("cognitive_bias", ""),
                    "cost_estimate": total_cost,
                },
            }
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done. Output:", str(out_path))


def run_pa20():
    import os
    total_cost = 0
    with open(PATIENT_INFO_PATH, "r", encoding="utf-8") as f:
        patient_info = json.load(f)

    out_dir = Path(BASE_DIR / "DataSyn")
    out_dir.mkdir(parents=True, exist_ok=True)

    for patient_template in tqdm(patient_info, desc="pa20"):
        total_output_list = []
        for i in range(NUM):
            dialogue_history = []
            output_list = []
            output_dict = {}
            story_path = os.path.join(OUTPUT_PASTEXP_PATH, "patient_{}".format(patient_template["患者"]), "story_{}.txt".format(i + 1))

            doc = Doctor(patient_template, DOCTOR_PROMPT_PATH, DIAGTREE_PATH, DOCTOR_MODEL_NAME, True)
            pat = Patient(patient_template, DOCTOR_MODEL_NAME, True, story_path)

            doctor_response = doc.doctor_response_gen(None, None)
            output_dict["doctor"] = doctor_response
            dialogue_history.append("医生：" + doctor_response)
            current_topic = "患者的精神状况"
            while True:
                patient_response, patient_cost = pat.patient_response_gen(current_topic, dialogue_history)
                output_dict["patient"] = patient_response
                dialogue_history.append("患者：" + patient_response)
                output_list.append(output_dict)
                output_dict = {}
                doctor_response, current_topic, doctor_cost = doc.doctor_response_gen(patient_response, dialogue_history)
                if "诊断结束，你的诊断结果" in doctor_response:
                    output_dict = {"doctor": doctor_response}
                    output_list.append(output_dict)
                    break
                else:
                    dialogue_history.append("医生：" + doctor_response)
                    output_dict["doctor"] = doctor_response
            total_output_list.append({"conversation": output_list})
            total_cost += doctor_cost + patient_cost

        with open(out_dir / "patient_{}.json".format(patient_template["患者"]), "w", encoding="utf-8") as f:
            json.dump(total_output_list, f, indent=2, ensure_ascii=False)

    print("Total cost:", total_cost)


if __name__ == "__main__":
    if MODE == "cards":
        run_cards()
    else:
        run_pa20()