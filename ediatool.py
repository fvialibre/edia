# --- Imports libs ---
import os
import gradio as gr
import pandas as pd
import configparser
import json
from datetime import datetime


# --- Imports modules ---
from modules.model_embbeding import Embedding
from modules.module_vocabulary import Vocabulary
from modules.module_languageModel import LanguageModel
from modules.module_generativeLanguageModel import GenerativeLanguageModel
from modules.utils import parse_cmd_line_args


# --- Imports interfaces ---
from interfaces.interface_WordExplorer import interface as interface_wordExplorer
from interfaces.interface_BiasWordExplorer import interface as interface_biasWordExplorer
from interfaces.interface_data import interface as interface_data
from interfaces.interface_biasPhrase import interface as interface_biasPhrase
from interfaces.interface_chatActivity1 import interface as interface_chatActivity1
# from interfaces.interface_crowsPairs import interface as interface_crowsPairs


# --- Imports FastAPI ---
from fastapi import FastAPI
import uvicorn
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

# --- Imports Constants ---
from html_constants import NAVBAR_HTML, FOOTER_HTML, css


# --- Tool config ---
cmd_line_args = parse_cmd_line_args()
cfg = configparser.ConfigParser()
cfg.read('tool.cfg')

LANGUAGE            = cfg['INTERFACE']['language']
EMBEDDINGS_PATH     = cfg['WORD_EXPLORER']['embeddings_path']
NN_METHOD           = cfg['WORD_EXPLORER']['nn_method']
MAX_NEIGHBORS       = int(cfg['WORD_EXPLORER']['max_neighbors'])
CONTEXTS_DATASET    = cfg['DATA']['contexts_dataset']
VOCABULARY_SUBSET   = cfg['DATA']['vocabulary_subset']
AVAILABLE_WORDCLOUD = cfg['DATA'].getboolean('available_wordcloud')
LANGUAGE_MODEL      = cfg['LMODEL']['language_model']
AVAILABLE_LOGS      = cfg['LOGS'].getboolean('available_logs')

# Server
QUEUE_MAX_SIZE       = int(cfg['SERVER']['queue_max_size'])
REQUESTS_CONCURRENCY = int(cfg['SERVER']['requests_concurrency'])


# --- Init classes ---
embedding = Embedding(
    path=EMBEDDINGS_PATH,
    limit=100000,
    randomizedPCA=False,
    max_neighbors=MAX_NEIGHBORS,
    nn_method=NN_METHOD
)

vocabulary = Vocabulary(
    subset_name=VOCABULARY_SUBSET
)

beto_lm = LanguageModel(
    model_name=LANGUAGE_MODEL
)

generative_lm = GenerativeLanguageModel(
    model_name="facebook/xglm-564M"
)

labels_path = f"language/{LANGUAGE}.json"
if not os.path.isfile(labels_path):
    raise FileNotFoundError(labels_path)
labels = pd.read_json(labels_path)["app"]


# --- Main App ---

async def not_found(request, exc):
    return RedirectResponse(url="/")


exceptions = {
    404: not_found,
}

app = FastAPI(exception_handlers=exceptions)

@app.get('/')
async def root(request: Request):
    global app
    user_email = request.headers['ngrok-auth-user-email']
    user_name = request.headers['ngrok-auth-user-name']

    with open("./logs/logs_logins.jsonl", "a+", encoding='utf-8') as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "user_email": user_email,
            "user_name": user_name,
            "headers": str(request.headers),
        }, ensure_ascii=False) + "\n")

    INTERFACE_LIST = [
        interface_chatActivity1(
            user_email=user_email,
        ),
        interface_biasPhrase(
            language_model=beto_lm,
            generative_language_model=generative_lm,
            available_logs=AVAILABLE_LOGS,
            lang=LANGUAGE,
            user_email=user_email),
        interface_biasWordExplorer(
            embedding=embedding,
            available_logs=AVAILABLE_LOGS,
            lang=LANGUAGE,
            user_email=user_email),
        interface_wordExplorer(
            embedding=embedding,
            available_logs=AVAILABLE_LOGS,
            max_neighbors=MAX_NEIGHBORS,
            lang=LANGUAGE,
            user_email=user_email),
        interface_data(
            vocabulary=vocabulary,
            contexts=CONTEXTS_DATASET,
            available_logs=AVAILABLE_LOGS,
            available_wordcloud=AVAILABLE_WORDCLOUD,
            lang=LANGUAGE,
            user_email=user_email),
        # interface_crowsPairs(
        #     language_model=beto_lm,
        #     available_logs=AVAILABLE_LOGS,
        #     lang=LANGUAGE,
        #     user_email=user_email),
    ]

    TAB_NAMES = [
        "Actividad asincrónica 1",
        labels["phraseExplorer"],
        labels["biasWordExplorer"],
        labels["wordExplorer"],
        labels["dataExplorer"],
        # labels["crowsPairsExplorer"]
    ]

    if LANGUAGE != 'es':
        # Skip data tab when using other than spanish language
        INTERFACE_LIST = INTERFACE_LIST[:2] + INTERFACE_LIST[3:]
        TAB_NAMES = TAB_NAMES[:2] + TAB_NAMES[3:]

    edia_theme = gr.themes.Base.from_hub('guidoivetta/edia-theme')

    with gr.Blocks(theme=edia_theme, css=css, title="E.D.I.A.") as iface:
        _ = gr.HTML(NAVBAR_HTML)
        _ = gr.TabbedInterface(
            interface_list= INTERFACE_LIST,
            tab_names=TAB_NAMES,
        )
        _ = gr.HTML(FOOTER_HTML)

    # iface.queue(
    #     max_size=QUEUE_MAX_SIZE,
    #     concurrency_count=REQUESTS_CONCURRENCY
    # ) 
    user_path = f"/{user_email}"
    app = gr.mount_gradio_app(
        app=app,
        blocks=iface,
        path=user_path,
        # root_path=user_path,
    )
    return RedirectResponse(url=user_path)

if __name__ == '__main__':
    uvicorn.run(app, port=cmd_line_args['port'])