import gradio as gr
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
from datetime import datetime
import os
from dotenv import dotenv_values
from auth import school_list


# --- Interface ---
def interface() -> gr.Blocks:

    secrets = dotenv_values("./.env")
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def predict(message, history, user_email, school, age, gender):
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = llm(history_langchain_format)

        with open("./logs/logs_chatActivity1.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "user_email": user_email,
                "school": school,
                "age": age,
                "gender": gender,
                "message": message,
                "response": gpt_response.content,
                "history": history,
            }, ensure_ascii=False) + "\n")
        return gpt_response.content

    def toggle_chat(token_id, school, age, gender):
        if not (token_id is None
            or school is None
            or age is None
            or gender is None
            or len(token_id) == 0
            or len(school) == 0
            or len(age) == 0):
            return gr.update(visible=True)

    with gr.Blocks() as interface:
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
                    # info="Seleccione su escuela",
                    multiselect=False,
                    allow_custom_value=False,
                )
            with gr.Column():
                age = gr.Dropdown(
                    choices=[str(i) for i in range(1, 100)],
                    label="Seleccione su edad",
                    # info="Seleccione su edad",
                    multiselect=False,
                    allow_custom_value=False,
                )
            with gr.Column():
                gender = gr.Radio(
                    ["M", "F", "X"],
                    label="Seleccione su género",
                )
        with gr.Column(visible=False) as chat_col:
            chat = gr.ChatInterface(
                    predict,
                    title="ChatGPT vía EDIA",
                    description="En esta oportunidad vas a interactuar con el modelo de lenguaje ChatGPT.\nImportante: Para completar la actividad debes cargar los datos en el formulario contando cómo interactuaste con este modelo. Si cerrás la pestaña, no se guarda la conversación, así que recordá cópiarlo antes. Ahí mismo tenes un video que explica paso a paso cómo ingresar la información.",
                    additional_inputs=[
                        token_id,
                        school,
                        age,
                        gender
                    ],
                    retry_btn=None,
                    undo_btn=None,
                    clear_btn=None,
                    submit_btn="Enviar",
                    stop_btn=None,
                )
        token_id.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender
            ],
            outputs=chat_col)
        school.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender
            ],
            outputs=chat_col)  
        age.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender
            ],
            outputs=chat_col)
        gender.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                school,
                age,
                gender
            ],
            outputs=chat_col)

    return interface
