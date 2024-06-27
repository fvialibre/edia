import gradio as gr
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
from datetime import datetime
import os
from dotenv import dotenv_values


# --- Interface ---
def interface(
    user_email: str="",
) -> gr.Blocks:

    secrets = dotenv_values("./.env")
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def predict(message, history, user_email):
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
                "message": message,
                "response": gpt_response.content,
                "history": history,
            }, ensure_ascii=False) + "\n")
        return gpt_response.content

    with gr.Blocks() as interface:
        _ = gr.ChatInterface(
                predict,
                title="ChatGPT vía EDIA",
                description="En esta oportunidad vas a interactuar con el modelo de lenguaje ChatGPT.\nImportante: Para completar la actividad debes cargar los datos en el formulario contando cómo interactuaste con este modelo. Si cerrás la pestaña, no se guarda la conversación, así que recordá cópiarlo antes. Ahí mismo tenes un video que explica paso a paso cómo ingresar la información.",
                additional_inputs=[
                    gr.Textbox(
                        value=user_email,
                        visible=False,
                    ),                                        
                ],
                retry_btn=None,
                undo_btn=None,
                clear_btn=None,
                submit_btn="Enviar",
                stop_btn=None,
            )

    return interface
