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

    def predict(message, history, token_id, school, age, gender, prompt, temperature, top_p, max_tokens):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        history_langchain_format = [SystemMessage(content=prompt)]
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = llm.invoke(history_langchain_format)

        with open("./logs/logs_chatbot.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "token_id": token_id,
                "school": school,
                "age": age,
                "gender": gender,
                "message": message,
                "response": gpt_response.content,
                "history": history,
                "current_prompt": prompt,
                "temperature": temperature
            }, ensure_ascii=False) + "\n")
        return gpt_response.content
    
    def open_turn_feedback_modal(x: gr.LikeData, token_id, school, age, gender, prompt):    
        return {
            "selected_message": x.value,
            "is_like": x.liked,
            "turn_index": x.index,
            "token_id": token_id,
            "school": school,
            "age": age,
            "gender": gender,
            "prompt": prompt,
        }, Modal(visible=True)

    def send_turn_feedback_modal(turn_info_for_feedback, q1_checkbox, q1, q2_checkbox, q2, q3_checkbox, q3, q4_checkbox, q4):
        with open("./logs/logs_chatbot_feedback.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "selected_message": turn_info_for_feedback["selected_message"],
                "is_like": turn_info_for_feedback["is_like"],
                "turn_index": turn_info_for_feedback["turn_index"],
                "q1_checkbox": q1_checkbox,
                "q1": q1,
                "q2_checkbox": q2_checkbox,
                "q2": q2,
                "q3_checkbox": q3_checkbox,
                "q3": q3,
                "q4_checkbox": q4_checkbox,
                "q4": q4,
                "token_id": turn_info_for_feedback["token_id"],
                "school": turn_info_for_feedback["school"],
                "age": turn_info_for_feedback["age"],
                "gender": turn_info_for_feedback["gender"],
                "prompt": turn_info_for_feedback["prompt"],
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
            "prompt": None,
        })
        with gr.Row():
            with gr.Column(scale=70):
                token_id = gr.Textbox(
                    label="Escriba su identificador",
                    lines=1,
                )
            with gr.Column(scale=70, visible=False):
                with gr.Row():
                    with gr.Column():
                        school = gr.Number(
                            value=10000,
                            label="Seleccione el identificador de su escuela (Ver ➡️)",
                            visible=False
                        )
                        school_name = gr.HTML(
                            value=f"<p>No seleccionaste ningún colegio</p>",
                            visible=False
                        )
                    with gr.Column():
                        _ = gr.HTML(
                            value="<a href='https://docs.google.com/spreadsheets/d/1SQaQqXh46_J_VrcHo3YJUfPSfKIjbKi73EEtaImzk9c/edit'>Lista de escuelas 🔗</a>",
                            visible=False
                        )
            with gr.Column(scale=30):
                with gr.Row():
                    consent_checkbox = gr.Checkbox(
                        label='He leído y acepto el consentimiento informado ➡️',
                        value=False,
                    )
                    _ = gr.HTML(
                        value="<a href='https://docs.google.com/document/d/17Feum83dTqjcicgJxuWdZ3qLuL3emmVY2idGym_usLU/edit?usp=sharing'>Link 🔗</a>",
                    )
        with gr.Row():
            with gr.Column():
                age = gr.Number(
                    value=99,
                    label="Seleccione su edad",
                    visible=False
                )
            with gr.Column():
                gender = gr.Radio(
                    ["M", "F", "X"],
                    label="Seleccione su género",
                    value="X",
                    visible=False
                )
        with gr.Row():
            prompt = gr.Textbox(
                label="Escriba el system prompt (dejar vacío para usar ChatGPT normal)",
                lines=3,
            )
            with gr.Row():
                temperature = gr.Slider(
                    visible=False,
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.01,
                    label="Temperatura",
                    info="Controla la creatividad del modelo (0.0 = determinista, 2.0 = creativo)"
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label="Top-p",
                    info="Probabilidad acumulada para muestreo nuclear",
                    visible=False
                )
                max_tokens = gr.Slider(
                    minimum=1,
                    maximum=4096,
                    value=1024,
                    step=1,
                    label="Máx. tokens",
                    info="Límite de longitud de la respuesta",
                    visible=False
                )
            
        with gr.Column(visible=False, elem_id='col') as chat_col:
            gr.HTML("<h1 style='text-align: center;'>ChatGPT vía EDIA</h1>")
            gr.HTML("<p>En esta oportunidad vas a interactuar con el modelo de lenguaje ChatGPT.\nImportante: Si cerrás la pestaña, no se guarda la conversación, así que recordá copiarlo antes.</p>")
                    
            chatbot = gr.Chatbot(
                show_copy_button=True,
                # likeable=True,
            )
            chat_interface = gr.ChatInterface(
                    predict,
                    additional_inputs=[
                        token_id,
                        school,
                        age,
                        gender,
                        prompt,
                        temperature,
                        top_p,
                        max_tokens
                    ],
                    chatbot=chatbot,
                    # retry_btn=None,
                    # undo_btn=None,
                    # clear_btn=None,
                    submit_btn="Enviar",
                    stop_btn=None,
                )
            
        ### MODAL
        with Modal(visible=False) as turn_feedback_modal:
            _ = gr.HTML(HTML_FEEDBACK_TITLE)

            with gr.Row():
                with gr.Column():
                    q1_checkbox = gr.Checkbox(label="Affect")
                    q1 = gr.Slider(
                        1,
                        7,
                        value=4,
                        step=1,
                        label="Affect",
                        info="1 es tóxico, 7 es empático",
                        interactive=True,
                        visible=False,
                    )
                with gr.Column():
                    q2_checkbox = gr.Checkbox(label="Veracidad")
                    q2 = gr.Slider(
                        1,
                        7,
                        value=4,
                        step=1,
                        label="Veracidad",
                        info="1 es alucinación, 7 es factual",
                        interactive=True,
                        visible=False,
                    )
                with gr.Column():
                    q3_checkbox = gr.Checkbox(label="Sesgo")
                    q3 = gr.Slider(
                        1,
                        7,
                        value=4,
                        step=1,
                        label="Sesgo",
                        info="1 es sesgado, 7 es justo",
                        interactive=True,
                        visible=False,
                    )
                with gr.Column():
                    q4_checkbox = gr.Checkbox(label="Utilidad")
                    q4 = gr.Slider(
                        1,
                        7,
                        value=4,
                        step=1,
                        label="Utilidad",
                        info="1 es inútil, 7 es valioso",
                        interactive=True,
                        visible=False,
                    )

            def toggle_slider_visibility(checkbox):
                return gr.update(visible=checkbox)

            q1_checkbox.change(toggle_slider_visibility, inputs=[q1_checkbox], outputs=[q1])
            q2_checkbox.change(toggle_slider_visibility, inputs=[q2_checkbox], outputs=[q2])
            q3_checkbox.change(toggle_slider_visibility, inputs=[q3_checkbox], outputs=[q3])
            q4_checkbox.change(toggle_slider_visibility, inputs=[q4_checkbox], outputs=[q4])

            modal_submit_button = gr.Button("Enviar")
            
        with gr.Column(visible=True) as personal_data_missing:
            gr.Markdown("""
                ### Ingrese sus datos personales y confirme su consentimiento para poder realizar la consulta!

            """
            )

            
        def toggle_chat(token_id, school, age, gender, prompt, consent_checkbox):
            if not (token_id is None
                or school is None
                or age is None
                or gender is None
                or prompt is None
                or consent_checkbox is None
                or age < 0
                or age > 100
                or school == 0
                or school not in school_list
                or len(token_id) == 0
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
                prompt,
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
                prompt,
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
                prompt,
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
                prompt,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])
        prompt.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender,
                prompt,
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
                prompt,
                consent_checkbox
            ],
            outputs=[chat_col, personal_data_missing])
        
        def update_school_name(school):
            if school is None or school == 0:
                return (
                    gr.HTML(
                        value=f"<p>No seleccionaste ningún colegio</p>",
                    )
                )
            elif school not in school_list:
                return (
                    gr.HTML(
                        value=f"<p>El colegio seleccionado no existe</p>",
                    )
                )
            else:
                return (
                    gr.HTML(
                        value=f"<p>Seleccionaste: {school_list[school]}</p>",
                    )
                )
        school.change(
            fn=update_school_name,
            inputs=[
                school
            ],
            outputs=[school_name])

        chatbot.like(open_turn_feedback_modal, [token_id, school, age, gender, prompt], [turn_info_for_feedback, turn_feedback_modal])
        modal_submit_button.click(send_turn_feedback_modal, [turn_info_for_feedback, q1_checkbox, q1, q2_checkbox, q2, q3_checkbox, q3, q4_checkbox, q4], turn_feedback_modal)
    return interface
