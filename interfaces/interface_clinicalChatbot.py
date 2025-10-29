import uuid
import gradio as gr
from gradio_modal import Modal
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
from datetime import datetime
import os
from dotenv import dotenv_values
from data.clinical_patients.clinical_prompts import clinical_prompts
from html_constants import HTML_FEEDBACK_TITLE


# --- Interface ---
def interface() -> gr.Blocks:

    secrets = dotenv_values("./.env")
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

    def predict(message, history, token_id, age, gender, patient_id, participant_area):
        temperature = 1.0
        llm = ChatOpenAI(model="gpt-4.1", temperature=temperature)
        
        prompt = clinical_prompts[patient_id]
        history_langchain_format = [SystemMessage(content=prompt)]
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = llm.invoke(history_langchain_format)

        with open("./logs/logs_clinical_chatbot.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "token_id": token_id,
                "age": age,
                "gender": gender,
                "patient_id": patient_id,
                "participant_area": participant_area,
                "message": message,
                "response": gpt_response.content,
                "history": history,
                "current_prompt": prompt,
                "temperature": temperature
            }, ensure_ascii=False) + "\n")
        return gpt_response.content
    
    def open_turn_feedback_modal(x: gr.LikeData, token_id, age, gender, patient_id, participant_area):    
        return {
            "selected_message": x.value,
            "is_like": x.liked,
            "turn_index": x.index,
            "token_id": token_id,
            "age": age,
            "gender": gender,
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
            "patient_id": None,
            "participant_area": None
        })
        with gr.Row():
            with gr.Column():
                token_id = gr.Textbox(
                    label="Escriba su identificador",
                    lines=1,
                )
            with gr.Column():
                age = gr.Number(
                    value=0,
                    label="Seleccione su edad",
                    visible=True
                )
            with gr.Column():
                gender = gr.Radio(
                    ["M", "F", "X"],
                    label="Seleccione su género",
                    value="X",
                    visible=True
                )
            with gr.Row():
                consent_checkbox = gr.Checkbox(
                    label='He leído y acepto los consentimientos informados ➡️',
                    value=False,
                )
                _ = gr.HTML(
                    value="<a href='https://docs.google.com/document/d/17Feum83dTqjcicgJxuWdZ3qLuL3emmVY2idGym_usLU/edit?usp=sharing'>Link 1 🔗</a>",
                )
                _ = gr.HTML(
                    value="<a href='https://docs.google.com/document/d/1kAblUcbZK_EepoQxOviAuysRD0AM9vxJ/edit?usp=sharing&ouid=116057151073336320924&rtpof=true&sd=true'>Link 2 🔗</a>",
                )
        with gr.Row():
            with gr.Column():
                participant_area = gr.Dropdown(
                    ["Clínica Médica", "Medicina Familiar", "PFO", "Salud Mental"],
                    interactive=True,
                    label="Seleccione su área de participación",
                    visible=True
                )
            with gr.Column():
                patient_id = gr.Dropdown(
                    ["Paciente A", "Paciente B", "Paciente C"],
                    interactive=True,
                    label="Seleccione su paciente",
                    visible=True
                )

        with gr.Column(visible=False, elem_id='col') as chat_col:
            gr.HTML("<h1 style='text-align: center;'>Chatbot Clínico vía EDIA</h1>")
            gr.HTML(
                "<p>En esta pestaña, llamada <b>Chatbot Clínico</b>, podrás interactuar con diversos pacientes simulados por inteligencia artificial. "
                "Tu rol es ser el médico de estos pacientes, practicando tanto tus habilidades conversacionales como clínicas. "
                "Ten en cuenta que los pacientes pueden haber tenido estudios previos, los cuales puedes solicitar explícitamente durante la consulta.</p>"
            )
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
                        patient_id,
                        participant_area
                    ],
                    chatbot=chatbot,
                    submit_btn="Enviar",
                    stop_btn=None,
                )
            
        ### MODAL
        with Modal(visible=False) as turn_feedback_modal:
            _ = gr.HTML(HTML_FEEDBACK_TITLE)

            with gr.Row():
                text_feedback = gr.Textbox(
                    label="¿Tienes algún comentario sobre este mensaje?",
                    lines=2,
                    placeholder="Escribe tu comentario aquí...",
                )

            modal_submit_button = gr.Button("Enviar")
            
        with gr.Column(visible=True) as personal_data_missing:
            gr.Markdown("""
                ### Ingrese sus datos personales y confirme su consentimiento para poder realizar la consulta!

            """
            )

            
        def toggle_chat(token_id, age, gender, consent_checkbox):
            if not (token_id is None
                or age is None
                or gender is None
                or consent_checkbox is None
                or age <= 0
                or age >= 100
                or len(token_id) == 0
                or not consent_checkbox):
                return gr.Column(visible=True), gr.Column(visible=False)
            else:
                return gr.Column(visible=False), gr.Column(visible=True)
        
        token_id.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])
        age.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])
        gender.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])
        consent_checkbox.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])

        chatbot.like(open_turn_feedback_modal, [token_id, age, gender, patient_id, participant_area], [turn_info_for_feedback, turn_feedback_modal])
        modal_submit_button.click(send_turn_feedback_modal, [turn_info_for_feedback, text_feedback], [turn_feedback_modal, text_feedback])
    return interface
