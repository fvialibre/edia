import gradio as gr
from gradio_modal import Modal
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
from datetime import datetime
import os
from dotenv import dotenv_values
from auth import SCHOOL_LIST
from prompts import prompts
from html_constants import HTML_FEEDBACK_TITLE


# --- Interface ---
def interface(
    token_id,
    age,
    gender,
    nationality,
    region,
    school,
    consent_checkbox
) -> gr.Blocks:

    secrets = dotenv_values("./.env")
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

    def predict(message, history, token_id, age, gender, nationality, region, school, prompt, temperature, top_p, max_tokens):
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
                "age": age,
                "gender": gender,
                "nationality": nationality,
                "region": region,
                "school": school,
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
            "age": age,
            "gender": gender,
            "school": school,
            "prompt": prompt,
            "nationality": nationality,
            "region": region,
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
                "age": turn_info_for_feedback["age"],
                "gender": turn_info_for_feedback["gender"],
                "nationality": turn_info_for_feedback["nationality"],
                "region": turn_info_for_feedback["region"],
                "school": turn_info_for_feedback["school"],
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
            "age": None,
            "gender": None,
            "nationality": None,
            "region": None,
            "school": None,
            "prompt": None,
        })
        _ = gr.Markdown(
            "# " + "LLM via EDIA" + "\n\n" +
            "### " + "En esta oportunidad vas a interactuar con el modelo de lenguaje ChatGPT.\nImportante: Si cerrás la pestaña, no se guarda la conversación, así que recordá copiarlo antes."
        )

        with gr.Row():
            prompt = gr.Textbox(
                label="Escriba el system prompt (dejar vacío para usar ChatGPT normal)",
                lines=3,
            )
            with gr.Row():
                temperature = gr.Slider(
                    visible=True,
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

        chatbot = gr.Chatbot(
            show_copy_button=True,
        )
        _ = gr.ChatInterface(
                predict,
                additional_inputs=[
                    token_id,
                    age,
                    gender,
                    nationality,
                    region,
                    school,
                    prompt,
                    temperature,
                    top_p,
                    max_tokens
                ],
                chatbot=chatbot,
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

        chatbot.like(open_turn_feedback_modal, [token_id, school, age, gender, prompt], [turn_info_for_feedback, turn_feedback_modal])
        modal_submit_button.click(send_turn_feedback_modal, [turn_info_for_feedback, q1_checkbox, q1, q2_checkbox, q2, q3_checkbox, q3, q4_checkbox, q4], turn_feedback_modal)
    return interface
