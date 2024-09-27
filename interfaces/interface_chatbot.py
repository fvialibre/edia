import gradio as gr
from gradio_modal import Modal
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
from datetime import datetime
import os
from dotenv import dotenv_values
from auth import school_list
from prompts import prompts
from html_constants import HTML_FEEDBACK_TITLE


# --- Interface ---
def interface() -> gr.Blocks:

    secrets = dotenv_values("./.env")
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def predict(message, history, token_id, school, age, gender, group_id):
        history_langchain_format = [SystemMessage(content=prompts[group_id])]
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = llm(history_langchain_format)

        with open("./logs/logs_chatActivity2.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "token_id": token_id,
                "school": school,
                "age": age,
                "gender": gender,
                "message": message,
                "response": gpt_response.content,
                "history": history,
                "current_prompt": prompts[group_id],
                "group_id": group_id,
                "prompts": prompts,
            }, ensure_ascii=False) + "\n")
        return gpt_response.content
    
    def open_turn_feedback_modal(x: gr.LikeData, token_id, school, age, gender, group_id):    
        return {
            "selected_message": x.value,
            "is_like": x.liked,
            "turn_index": x.index,
            "token_id": token_id,
            "school": school,
            "age": age,
            "gender": gender,
            "group_id": group_id,
        }, Modal(visible=True)

    def send_turn_feedback_modal(turn_info_for_feedback, q1, q2):
        with open("./logs/logs_chatActivity2_feedback.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "selected_message": turn_info_for_feedback["selected_message"],
                "is_like": turn_info_for_feedback["is_like"],
                "turn_index": turn_info_for_feedback["turn_index"],
                "q1": q1,
                "q2": q2,
                "token_id": turn_info_for_feedback["token_id"],
                "school": turn_info_for_feedback["school"],
                "age": turn_info_for_feedback["age"],
                "gender": turn_info_for_feedback["gender"],
                "group_id": turn_info_for_feedback["group_id"],
                "prompts": prompts,
            }, ensure_ascii=False) + "\n")
        gr.Info("Feedback enviado con éxito")
        return Modal(visible=False)

    with gr.Blocks(css=".contain { display: flex !important; flex-direction: column !important; }"
    "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
    "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
    "#col { height: 100vh !important; }") as interface:
        turn_info_for_feedback = gr.State({
            "selected_message": None,
            "is_like": None,
            "turn_index": None,
            "token_id": None,
            "school": None,
            "age": None,
            "gender": None,
            "group_id": None,
        })
        with gr.Row():
            with gr.Column():
                token_id = gr.Textbox(
                    label="Escriba su correo electrónico",
                    lines=1,
                )
            with gr.Column():
                school = gr.Dropdown(
                    choices=school_list,
                    label="Seleccione su escuela",
                    multiselect=False,
                    allow_custom_value=False,
                )
            with gr.Column():
                age = gr.Dropdown(
                    choices=[str(i) for i in range(1, 100)],
                    label="Seleccione su edad",
                    multiselect=False,
                    allow_custom_value=False,
                )
            with gr.Column():
                gender = gr.Radio(
                    ["M", "F", "X"],
                    label="Seleccione su género",
                )
            with gr.Column():
                group_id = gr.Dropdown(
                    choices=list(prompts.keys()),
                    label="Seleccione su grupo",
                    multiselect=False,
                    allow_custom_value=False,
                )
            with gr.Column():
                with gr.Row():
                    consent_checkbox = gr.Checkbox(
                        label='He leído y acepto el consentimiento informado ➡️',
                        value=False
                    )
                    _ = gr.HTML(
                        value="<a href='https://docs.google.com/document/d/1yR_spvGWiq9ivuz4iOI8kzwI0Ope6chS/edit'>Link 🔗</a>",
                    )
            
        with gr.Column(visible=False, elem_id='col') as chat_col:
            gr.HTML("<h1 style='text-align: center;'>ChatGPT vía EDIA</h1>")
            gr.HTML("<p>En esta oportunidad vas a interactuar con el modelo de lenguaje ChatGPT.\nImportante: Para completar la actividad debes cargar los datos en el formulario contando cómo interactuaste con este modelo. Si cerrás la pestaña, no se guarda la conversación, así que recordá cópiarlo antes. Ahí mismo tenes un video que explica paso a paso cómo ingresar la información.</p>")
                    
            chatbot = gr.Chatbot(
                show_copy_button=True,
                likeable=True,
            )
            chat_interface = gr.ChatInterface(
                    predict,
                    additional_inputs=[
                        token_id,
                        school,
                        age,
                        gender,
                        group_id,
                    ],
                    chatbot=chatbot,
                    retry_btn=None,
                    undo_btn=None,
                    clear_btn=None,
                    submit_btn="Enviar",
                    stop_btn=None,
                )
            
        ### MODAL
        with Modal(visible=False) as turn_feedback_modal:
            _ = gr.HTML(HTML_FEEDBACK_TITLE)

            with gr.Row():
                with gr.Column():
                    q1 = gr.Slider(
                        1,
                        7,
                        value=4,
                        step=1,
                        label="Veracidad",
                        info="1 es mentira, 7 es verdad",
                        interactive=True,
                    )
                with gr.Column():
                    q2 = gr.Slider(
                        1,
                        7,
                        value=4,
                        step=1,
                        label="Sesgo",
                        info="1 es sesgado, 7 es justo",
                        interactive=True,
                    )
            modal_submit_button = gr.Button("Enviar")
            
        with gr.Column(visible=True) as personal_data_missing:
            gr.Markdown("""
                ### Ingrese sus datos personales y confirme su consentimiento para poder realizar la consulta!

            """
            )

            
        def toggle_chat(token_id, school, age, gender, group_id,consent_checkbox):
            if not (token_id is None
                or school is None
                or age is None
                or gender is None
                or group_id is None
                or consent_checkbox is None
                or len(token_id) == 0
                or len(school) == 0
                or len(age) == 0
                or len(group_id) == 0
                or not consent_checkbox):
                return gr.Column(visible=True), gr.Column(visible=False)
            else:
                return gr.Column(visible=False), gr.Column(visible=True)
        
        token_id.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender,
                group_id,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])
        school.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender,
                group_id,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])  
        age.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender,
                group_id,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])
        gender.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender,
                group_id,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])
        group_id.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender,
                group_id,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])
        consent_checkbox.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender,
                group_id,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])

        chatbot.like(open_turn_feedback_modal, [token_id, school, age, gender, group_id], [turn_info_for_feedback, turn_feedback_modal])
        modal_submit_button.click(send_turn_feedback_modal, [turn_info_for_feedback, q1, q2], turn_feedback_modal)
    return interface
