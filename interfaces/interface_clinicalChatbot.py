import uuid
import gradio as gr
from gradio_modal import Modal
from gradio_i18n import gettext as i18n
import pandas as pd
from modules.module_ollama import ModelWrapper
import json
from datetime import datetime
import os
from dotenv import dotenv_values
from data.clinical_patients.clinical_prompts import clinical_prompts
from html_constants import HTML_FEEDBACK_TITLE


# --- Interface ---
def interface(
    token_id,
    age,
    gender,
    nationality,
    region,
    school,
    consent_checkbox,
    patient_id,
    participant_area,
) -> gr.Blocks:

    secrets = dotenv_values("./.env")
    os.environ["OLLAMA_API_KEY"] = secrets["OLLAMA_API_KEY"]
    llm = ModelWrapper(
        token=secrets["OLLAMA_API_KEY"],
        model="vllm/gemma4-26b",
    )

    def predict(
        message,
        history,
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        patient_id,
        participant_area,
    ):
        temperature = 1.0
        if patient_id not in clinical_prompts:
            return "Please select a patient before starting the consultation."

        prompt = clinical_prompts[patient_id]
        history_text = []
        for human, ai in history:
            history_text.append(f"User: {human}")
            history_text.append(f"Assistant: {ai}")
        history_text.append(f"User: {message}")
        user_prompt = "\n".join(history_text)
        model_response = llm.invoke(prompt, user_prompt)
        response_content = model_response["content"]

        with open("./logs/logs_clinical_chatbot.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "token_id": token_id,
                "age": age,
                "gender": gender,
                "nationality": nationality,
                "region": region,
                "school": school,
                "patient_id": patient_id,
                "participant_area": participant_area,
                "message": message,
                "response": response_content,
                "history": history,
                "current_prompt": prompt,
                "temperature": temperature
            }, ensure_ascii=False) + "\n")
        return response_content
    
    def open_turn_feedback_modal(
        x: gr.LikeData,
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        patient_id,
        participant_area,
    ):
        return {
            "selected_message": x.value,
            "is_like": x.liked,
            "turn_index": x.index,
            "token_id": token_id,
            "age": age,
            "gender": gender,
            "nationality": nationality,
            "region": region,
            "school": school,
            "participant_area": participant_area,
            "patient_id": patient_id,
        }, Modal(visible=True)

    def send_turn_feedback_modal(turn_info_for_feedback, text_feedback):
        with open("./logs/logs_clinical_chatbot_feedback.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "selected_message": turn_info_for_feedback["selected_message"],
                "is_like": turn_info_for_feedback["is_like"],
                "turn_index": turn_info_for_feedback["turn_index"],
                "text_feedback": text_feedback,
                "token_id": turn_info_for_feedback["token_id"],
                "age": turn_info_for_feedback["age"],
                "gender": turn_info_for_feedback["gender"],
                "nationality": turn_info_for_feedback["nationality"],
                "region": turn_info_for_feedback["region"],
                "school": turn_info_for_feedback["school"],
                "participant_area": turn_info_for_feedback["participant_area"],
                "patient_id": turn_info_for_feedback["patient_id"],
                "prompt": clinical_prompts[turn_info_for_feedback["patient_id"]],
            }, ensure_ascii=False) + "\n")
        gr.Info("Feedback enviado con éxito")
        return (Modal(visible=False), "")

    with gr.Blocks(css=".contain { display: flex !important; flex-direction: column !important; }"
    "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
    "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
    "#col { height: 100vh !important; }") as interface:
        turn_info_for_feedback = gr.State({
            "selected_message": None,
            "is_like": None,
            "turn_index": None,
            "token_id": None,
            "age": None,
            "gender": None,
            "nationality": None,
            "region": None,
            "school": None,
            "patient_id": None,
            "participant_area": None
        })
        with gr.Column(visible=True, elem_id='col') as chat_col:
            gr.HTML("<h1 style='text-align: center;'>" + i18n("ClinicalChatbotTitle") + "</h1>")
            gr.Markdown(i18n("ClinicalChatbotDescription"))
            chatbot = gr.Chatbot(
                show_copy_button=True,
                # likeable=True,
            )
            chat_interface = gr.ChatInterface(
                    predict,
                    additional_inputs=[
                        token_id,
                        age,
                        gender,
                        nationality,
                        region,
                        school,
                        patient_id,
                        participant_area
                    ],
                    chatbot=chatbot,
                    submit_btn=i18n("ClinicalChatbotSubmitButton"),
                    stop_btn=None,
                )
            
        ### MODAL
        with Modal(visible=False) as turn_feedback_modal:
            _ = gr.HTML(HTML_FEEDBACK_TITLE)

            with gr.Row():
                text_feedback = gr.Textbox(
                    label=i18n("ClinicalChatbotFeedbackLabel"),
                    lines=2,
                    placeholder=i18n("ClinicalChatbotFeedbackPlaceholder"),
                )

            modal_submit_button = gr.Button(i18n("ClinicalChatbotSubmitButton"))
            
        chatbot.like(
            open_turn_feedback_modal,
            [token_id, age, gender, nationality, region, school, patient_id, participant_area],
            [turn_info_for_feedback, turn_feedback_modal],
        )
        modal_submit_button.click(send_turn_feedback_modal, [turn_info_for_feedback, text_feedback], [turn_feedback_modal, text_feedback])
    return interface
