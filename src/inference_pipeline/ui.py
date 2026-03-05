import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

import gradio as gr

from core import logger_utils
from core.config import settings
from inference_pipeline.llm_twin import DepressionTreatmentSystem

logger = logger_utils.get_logger(__name__)


def _new_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"


def _new_session_id() -> str:
    return f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"


def load_model(model_path: str, mock_mode: bool, enable_rag: bool = False, enable_rerank: bool = False) -> tuple[Any, str]:
    try:
        system = DepressionTreatmentSystem(
            model_path=model_path,
            mock=mock_mode,
            use_local_model=True,
            enable_rag=enable_rag,
            enable_rerank=enable_rerank,
        )
        
        rag_status = system.get_rag_status()
        rag_info = ""
        if enable_rag:
            if rag_status.get("vectorstore_available", False):
                rag_info = f"\n🔍 RAG enabled (rerank: {'on' if enable_rerank else 'off'})"
            else:
                rag_info = "\n⚠️ RAG init failed, using base mode"
        return system, f"✅ Model loaded: {model_path}{rag_info}"
    except Exception as e:
        logger.exception("Model load failed")
        return None, f"❌ Load failed: {e}"


def reset_session() -> tuple[str, str, list[tuple[str, str]], list[dict], str]:
    return _new_user_id(), _new_session_id(), [], [], "Session reset."


def chat(
    user_message: str,
    chat_history: list[tuple[str, str]],
    system: Any,
    conversation_state: list[dict],
    user_id: str,
    session_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    do_sample: bool,
    enable_rag: bool = False,
) -> tuple[
    str,
    list[tuple[str, str]],
    list[dict],
]:
    if system is None:
        return "", chat_history, conversation_state

    user_message = (user_message or "").strip()
    if not user_message:
        return "", chat_history, conversation_state

    try:
        response = system.generate(
            query=user_message,
            user_id=user_id,
            session_id=session_id,
            enable_rag=enable_rag,
            conversation_history=conversation_state,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=bool(do_sample),
        )
        answer = response.get("answer", "").strip() or "(empty)"

        conversation_state = [
            *conversation_state,
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]

        chat_history = [*chat_history, (user_message, answer)]
        return "", chat_history, conversation_state
    except Exception as e:
        logger.exception("Generate failed")
        chat_history = [*chat_history, (user_message, f"❌ Error: {e}")]
        return "", chat_history, conversation_state


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="抑郁症诊疗系统（Gradio）") as demo:
        gr.Markdown("## 抑郁症诊疗系统（本地微调模型推理）")

        system_state = gr.State(None)  # DepressionTreatmentSystem | None
        conversation_state = gr.State([])  # list[dict]

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="诊疗对话", height=520)
                user_input = gr.Textbox(
                    label="请输入您的问题",
                    placeholder="例如：我最近总是感觉很累、睡眠不好，对什么都提不起兴趣…",
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")

            with gr.Column(scale=5):
                gr.Markdown("### 系统配置")
                model_path = gr.Textbox(
                    label="模型路径（本地目录或 HuggingFace 模型ID）",
                    value=settings.MODEL_ID,
                )
                mock_mode = gr.Checkbox(label="Mock 模式（不加载真实模型）", value=False)
                
                gr.Markdown("### RAG 功能")
                enable_rag = gr.Checkbox(label="启用 RAG 检索增强", value=False)
                enable_rerank = gr.Checkbox(label="启用重排序", value=False)

                gr.Markdown("### 生成参数")
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                max_new_tokens = gr.Slider(16, 2048, value=512, step=16, label="max_new_tokens")
                do_sample = gr.Checkbox(label="do_sample", value=True)

                gr.Markdown("### 会话信息")
                user_id = gr.Textbox(label="user_id", value=_new_user_id())
                session_id = gr.Textbox(label="session_id", value=_new_session_id())

                with gr.Row():
                    load_btn = gr.Button("加载/重载模型", variant="primary")
                    reset_btn = gr.Button("重置会话")
                status = gr.Markdown()

        load_btn.click(
            load_model,
            inputs=[model_path, mock_mode, enable_rag, enable_rerank],
            outputs=[system_state, status],
        )

        reset_btn.click(
            reset_session,
            inputs=[],
            outputs=[user_id, session_id, chatbot, conversation_state, status],
        )

        clear_btn.click(
            lambda: ([], []),
            inputs=[],
            outputs=[chatbot, conversation_state],
        )

        send_btn.click(
            chat,
            inputs=[
                user_input,
                chatbot,
                system_state,
                conversation_state,
                user_id,
                session_id,
                temperature,
                top_p,
                max_new_tokens,
                do_sample,
                enable_rag,
            ],
            outputs=[user_input, chatbot, conversation_state],
        )
        user_input.submit(
            chat,
            inputs=[
                user_input,
                chatbot,
                system_state,
                conversation_state,
                user_id,
                session_id,
                temperature,
                top_p,
                max_new_tokens,
                do_sample,
                enable_rag,
            ],
            outputs=[user_input, chatbot, conversation_state],
        )

        gr.Examples(
            examples=[
                "医生，我最近总是感觉很累，对什么都提不起兴趣，这是抑郁症吗？",
                "我晚上很难入睡，经常早醒，情绪也很低落，怎么办？",
                "抑郁症的治疗方式有哪些？需要吃药吗？",
                "我有自伤念头但又害怕告诉别人，我该怎么做？",
            ],
            inputs=[user_input],
        )

    return demo


if __name__ == "__main__":
    build_demo().queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
