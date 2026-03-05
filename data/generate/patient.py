import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import llm_tools_api
from typing import Tuple

class Patient(llm_tools_api.PatientCost):
    def __init__(self, patient_template, model_path, use_api, story_path) -> None:
        super().__init__(model_path.split('/')[-1])
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.patient_model = None
        self.patient_tokenizer = None
        self.experience = None
        self.patient_template = patient_template
        self.messages = []
        self.use_api = use_api
        self.client = None
        self.story_path = story_path
        self.dialbegin = True
        self.cards_mode = isinstance(self.patient_template, dict) and ("basic_info" in self.patient_template)
        self.max_history_turns = 10

        if self.cards_mode:
            self._cards_init_persona()
        else:
            self.system_prompt = "你是一名{}岁的{}性{}患者，正在和一位精神科医生交流，使用口语化的表达，输出一整段没有空行的内容。如果医生的问题可以用是/否来回答，你的回复要简短精确。".format(
                self.patient_template['年龄'],
                self.patient_template['性别'],
                self.patient_template['诊断结果']
            )


    def patientbot_init(self):
        if self.use_api:
            self.client = llm_tools_api.patient_client_init(self.model_name)
        else:
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.patient_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.messages.append({"role": "system", "content": self.system_prompt})

    def _cards_init_persona(self):
        bi = self.patient_template.get("basic_info", {}) or {}
        age = bi.get("age", 28)
        gender_cn = self.patient_template.get("gender_cn") or random.choice(["男", "女"])
        self.patient_template["gender_cn"] = gender_cn
        occ = bi.get("occupation", "")
        mbti = bi.get("mbti", "")
        bias = self.patient_template.get("cognitive_bias", "")
        inner = self.patient_template.get("inner_monologue", "")

        self.system_prompt = (
            f"你是一个{age}岁的{gender_cn}性，职业/状态是“{occ}”，MBTI是{mbti}。"
            "你正在精神科/心理科门诊与医生对话，表达方式要口语化、真实、具体，输出一整段不要空行。"
            "你有抑郁相关困扰：可能出现情绪低落、兴趣减退、精力不足、睡眠/食欲/注意力变化、自责内疚、功能受损等（按医生提问如实回答）。"
            "安全：如果医生询问自伤/自杀相关，只描述想法/冲动/担心/保护因素，不描述任何方式/步骤/工具细节。"
            f"你的常见负性思维模式更接近：{bias}（不要直接说出这个词，而是在叙述里自然体现）。"
            f"这是你更私密的内心独白/背景（仅供你一致性参考，不要整段照抄）：{inner}"
        )
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def patient_response_gen_cards(self, doctor_text: str, dialogue_history) -> Tuple[str, float]:
        if not self.use_api:
            raise NotImplementedError("Cards mode requires use_api=True")
        if self.client is None:
            self.client = llm_tools_api.patient_client_init(self.model_name)

        history = dialogue_history[-self.max_history_turns:] if dialogue_history else []
        user_prompt = (
            "你与医生的对话历史如下："
            + str(history)
            + "\n医生刚刚说："
            + str(doctor_text)
            + "\n请生成你作为患者的回复，要求："
            + "\n1) 第一人称口语化、具体，不要空泛，不要反问医生，不要总以“医生，”开头。"
            + "\n2) 尽量体现事件-想法-情绪-行为（如果医生问到），并与抑郁困扰一致。"
            + "\n3) 不要重复历史里已经说过的同一句话；遇到病例没写到的细节可以合理虚构但要自洽。"
            + "\n输出一整段文字，不要空行。"
        )
        self.messages.append({"role": "user", "content": user_prompt})
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            top_p=0.85,
            frequency_penalty=0.7,
        )
        super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
        patient_text = chat_response.choices[0].message.content
        self.messages.pop()
        return patient_text, super().get_cost()


    def patient_response_gen(self, current_topic, dialogue_history):
        # self.messages.append({"role": "user", "content": doctor_response})
        if self.use_api:
            if self.cards_mode:
                last_doc = ""
                if dialogue_history and isinstance(dialogue_history[-1], str) and dialogue_history[-1].startswith("医生："):
                    last_doc = dialogue_history[-1].replace("医生：", "", 1).strip()
                patient_text, _ = self.patient_response_gen_cards(last_doc, dialogue_history)
                return patient_text, super().get_cost()
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            patient_template = {key:val for key, val in self.patient_template.items() if key != '处理意见'} 
            if self.experience is None:
                self.experience, x = llm_tools_api.api_patient_experience_trigger(self.model_name, dialogue_history, self.story_path)
                super().money_cost(x[0], x[1])
            if self.experience is None:
                patient_prompt = "你是一名{}患者，正在和一位精神卫生中心临床心理科医生进行交流。如果医生的问题可以用是/否来回答，你的回复要简短精确。\n你的病例为“{}”，\n你和医生的对话历史为{}， \
                    \n现在请根据下面要求生成:\n1.使用第一人称口语化的回答，如果不是必要情况，不要生成疑问句，不要总是以”医生，“开头。\n2.回答围绕{}展开，如果医生的问题可以用是/否来回答，你的回复要简短精确。在对话历史中提到过的内容不要重复再提起。\n3.回复内容必须根据病例内容，对话历史。如果出现不在病例内容中的问题，发挥想象力虚构回答。" \
                    .format(self.patient_template['诊断结果'], patient_template, dialogue_history[-8:], current_topic)
                self.messages.append({"role": "user", "content": patient_prompt})
                chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,
                        top_p=0.85,
                        frequency_penalty=0.8
                    )
                super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                patient_response = chat_response.choices[0].message.content
                self.messages.pop()
            else:
                patient_prompt = "你是一名{}患者，正在和一位精神卫生中心临床心理科医生进行交流。 \
                    \n\n现在请根据下面要求生成对医生的回答:\n1.回复内容必须根据：\n  （1）病例：“{}“\n  （2）过去的创伤经历：“{}”\n  （3）对话历史：“{}”。\n2.你当前的回复需要围绕话题“{}”展开，要精炼精确。在对话历史中提到过的内容不要重复再提起。\n3.涉及到过去的创伤经历时，需要详细清晰的阐述。但是如果已经说过，不允许重复生成。 \n4.使用第一人称口语化的回答，不要生成疑问句，不要总是以”医生，“开头。如果遇到不在病例和过去创伤经历中的问题，发挥想象力虚构回答，不要输出类似”一些事情“，要具体细节。" \
                    .format(self.patient_template['诊断结果'], patient_template, self.experience, dialogue_history[-8:], current_topic)
                self.messages.append({"role": "user", "content": patient_prompt})
                chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,
                        top_p=0.85,
                        frequency_penalty=0.7
                    )
                super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                patient_response = chat_response.choices[0].message.content
                self.messages.pop()
                # self.messages.append({"role": "assistant", "content": patient_response})
        else:
            #TODO
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            text = self.patient_tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            patient_model_inputs = self.patient_tokenizer([text], return_tensors="pt").to(self.patient_model.device)
            generated_ids = self.patient_model.generate(
                patient_model_inputs.input_ids,
                max_new_tokens=2048
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(patient_model_inputs.input_ids, generated_ids)
            ]
            patient_response = self.patient_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.messages.append({"role": "assistant", "content": patient_response})
        
        return patient_response, super().get_cost()
    
class Roleplay_Patient:
    def __init__(self, patient_template, model_path, use_api) -> None:
        self.patient_template = patient_template
        self.use_api = use_api
        self.model_name = model_path.split('/')[-1]
        self.system_prompt = "你是一名{}岁的{}性{}患者，正在和一位精神科医生交流，使用口语化的表达，输出一整段没有空行的内容。如果医生的问题可以用是/否来回答，你的回复要简短精确。".format(self.patient_template['年龄'], self.patient_template['性别'], self.patient_template['诊断结果'])
        self.messages = []
        self.dialbegin = True

    def patientbot_init(self):
        if self.use_api:
            self.client = llm_tools_api.patient_client_init(self.model_name)

    def patient_response_gen(self, dialogue_history):
        if self.use_api:
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            patient_template = {key:val for key, val in self.patient_template.items() if key != '处理意见'} 
            patient_prompt = "你是一名{}患者，正在和一位精神卫生中心临床心理科医生进行交流。如果医生的问题可以用是/否来回答，你的回复要简短精确。\n你的病例为“{}”，\n你和医生的对话历史为{}， \
                    \n现在请根据下面要求生成:\n1.使用第一人称口语化的回答，如果不是必要情况，不要生成疑问句，不要总是以”医生，“开头。\n2.如果医生的问题可以用是/否来回答，你的回复要简短精确。在对话历史中提到过的内容不要重复再提起。\n3.回复内容必须根据病例内容，对话历史。如果出现不在病例内容中的问题，发挥想象力虚构回答。" \
                    .format(self.patient_template['诊断结果'], patient_template, dialogue_history[-8:])
            self.messages.append({"role": "user", "content": patient_prompt})
            chat_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    top_p=0.85,
                    frequency_penalty=0.8
                )
            patient_response = chat_response.choices[0].message.content
            self.messages.pop()
        return patient_response