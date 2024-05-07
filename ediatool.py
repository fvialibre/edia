# --- Imports libs ---
import os
import gradio as gr
import pandas as pd
import configparser


# --- Imports modules ---
from modules.model_embbeding import Embedding
from modules.module_vocabulary import Vocabulary
from modules.module_languageModel import LanguageModel
from modules.utils import parse_cmd_line_args


# --- Imports interfaces ---
from interfaces.interface_WordExplorer import interface as interface_wordExplorer
from interfaces.interface_BiasWordExplorer import interface as interface_biasWordExplorer
from interfaces.interface_data import interface as interface_data
from interfaces.interface_biasPhrase import interface as interface_biasPhrase
# from interfaces.interface_crowsPairs import interface as interface_crowsPairs


# --- Imports FastAPI ---
from fastapi import FastAPI
import uvicorn
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request


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

labels_path = f"language/{LANGUAGE}.json"
if not os.path.isfile(labels_path):
    raise FileNotFoundError(labels_path)
labels = pd.read_json(labels_path)["app"]


# --- Main App ---

app = FastAPI()

@app.get('/')
async def root(request: Request):
    global app
    user_email = request.headers['ngrok-auth-user-email']

    INTERFACE_LIST = [
        interface_biasPhrase(
            language_model=beto_lm,
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

    HEADER_HTML = """
        <div style="background-color: black; color: white; padding: 10px; margin-bottom: 30px; text-align: center; width: 100%; top: 0; left: 0; display: flex; align-items: center; justify-content: flex-start;">
            <a href="https://www.vialibre.org.ar/" style="color: white; padding: 5px 10px; font-weight: bold; text-decoration: none; background-color: #333; border: none; border-radius: 5px; cursor: pointer; margin-right: auto;">VOLVER</a>
            <div style="flex: 1; text-align: center; color: white; font-weight: bold">EDIA: Estereotipos y Discriminación en Inteligencia Artificial</div>
        </div>
    """

    with gr.Blocks(theme='snehilsanyal/scikit-learn') as iface:
        _ = gr.TabbedInterface(
            interface_list= INTERFACE_LIST,
            tab_names=TAB_NAMES,
            title=HEADER_HTML,
        )

    # iface.queue(
    #     max_size=QUEUE_MAX_SIZE,
    #     concurrency_count=REQUESTS_CONCURRENCY
    # )   
    app = gr.mount_gradio_app(app, iface, "/home")
    return RedirectResponse(url='/home')

if __name__ == '__main__':
    uvicorn.run(app)