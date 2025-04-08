# --- Imports libs ---
import configparser
import json
import os
from datetime import datetime

import gradio as gr
import pandas as pd

# --- Imports Constants ---
from html_constants import FOOTER_HTML, NAVBAR_HTML, css
from interfaces.interface_arena import interface as interface_arena
from interfaces.interface_biasPhrase import interface as interface_biasPhrase
from interfaces.interface_BiasWordExplorer import \
    interface as interface_biasWordExplorer
from interfaces.interface_chatbot import interface as interface_chatbot
from interfaces.interface_data import interface as interface_data
from interfaces.interface_logsData import interface as interface_logsData
from interfaces.interface_validator import interface as interface_validator
# --- Imports interfaces ---
from interfaces.interface_WordExplorer import \
    interface as interface_wordExplorer
# --- Imports modules ---
from modules.model_embbeding import Embedding
from modules.module_languageModel import LanguageModel
from modules.module_vocabulary import Vocabulary
from modules.utils import parse_cmd_line_args

# from interfaces.interface_crowsPairs import interface as interface_crowsPairs



# --- Tool config ---
cmd_line_args = parse_cmd_line_args()
cfg = configparser.ConfigParser()
cfg.read('tool.cfg')

LANGUAGE                = cfg['INTERFACE']['language']
EMBEDDINGS_PATH         = cfg['WORD_EXPLORER']['embeddings_path']
NN_METHOD               = cfg['WORD_EXPLORER']['nn_method']
MAX_NEIGHBORS           = int(cfg['WORD_EXPLORER']['max_neighbors'])
CONTEXTS_DATASET        = cfg['DATA']['contexts_dataset']
VOCABULARY_SUBSET       = cfg['DATA']['vocabulary_subset']
AVAILABLE_WORDCLOUD     = cfg['DATA'].getboolean('available_wordcloud')
SPANISH_LANGUAGE_MODEL  = cfg['LMODEL']['spanish_language_model']
ENGLISH_LANGUAGE_MODEL  = cfg['LMODEL']['english_language_model']
AVAILABLE_LOGS          = cfg['LOGS'].getboolean('available_logs')

# Server
QUEUE_MAX_SIZE       = int(cfg['SERVER']['queue_max_size'])
REQUESTS_CONCURRENCY = int(cfg['SERVER']['requests_concurrency'])


# --- Init classes ---
# embedding = Embedding(
#     path=EMBEDDINGS_PATH,
#     limit=100000,
#     randomizedPCA=False,
#     max_neighbors=MAX_NEIGHBORS,
#     nn_method=NN_METHOD
# )

# vocabulary = Vocabulary(
#     subset_name=VOCABULARY_SUBSET
# )

# spanish_lm = LanguageModel(
#     model_name=SPANISH_LANGUAGE_MODEL
# )

# english_lm = LanguageModel(
#     model_name=ENGLISH_LANGUAGE_MODEL
# )

labels_path = f"language/{LANGUAGE}.json"
if not os.path.isfile(labels_path):
    raise FileNotFoundError(labels_path)
labels = pd.read_json(labels_path)["app"]


# # --- Main App ---

INTERFACE_LIST = [
    interface_validator(lang=LANGUAGE),
    # interface_biasPhrase(
    #     spanish_language_model=spanish_lm,
    #     english_language_model=english_lm,
    #     available_logs=AVAILABLE_LOGS,
    #     lang=LANGUAGE,),
    # interface_data(
    #     vocabulary=vocabulary,
    #     contexts=CONTEXTS_DATASET,
    #     available_logs=AVAILABLE_LOGS,
    #     available_wordcloud=AVAILABLE_WORDCLOUD,
    #     lang=LANGUAGE,),
    # interface_biasWordExplorer(
    #     embedding=embedding,
    #     available_logs=AVAILABLE_LOGS,
    #     lang=LANGUAGE,),
    # interface_chatbot(),
    # # interface_arena(),
    # interface_wordExplorer(
    #     embedding=embedding,
    #     available_logs=AVAILABLE_LOGS,
    #     max_neighbors=MAX_NEIGHBORS,
    #     lang=LANGUAGE,),
    # # interface_crowsPairs(
    # #     language_model=beto_lm,
    # #     available_logs=AVAILABLE_LOGS,
    # #     lang=LANGUAGE,
    # #     user_email=user_email),
    # interface_logsData(
    #     available_logs=AVAILABLE_LOGS,
    #     lang=LANGUAGE,),
]

TAB_NAMES = [
    labels["stereotypeValidator"],
    # labels["phraseExplorer"],
    # labels["dataExplorer"],
    # labels["biasWordExplorer"],
    # "LLM vía EDIA",
    # # "Arena",
    # labels["wordExplorer"],
    # # labels["crowsPairsExplorer"],
    # "Visualizar datos",
]

if LANGUAGE != 'es':
    # Skip data tab when using other than spanish language
    INTERFACE_LIST = INTERFACE_LIST[:2] + INTERFACE_LIST[3:]
    TAB_NAMES = TAB_NAMES[:2] + TAB_NAMES[3:]

edia_theme = gr.themes.Base.from_hub('guidoivetta/edia-theme')

with gr.Blocks(theme=edia_theme, css=css, title="EDIA") as iface:
    # _ = gr.HTML(NAVBAR_HTML)
    _ = gr.TabbedInterface(
        interface_list=INTERFACE_LIST,
        tab_names=TAB_NAMES,
    )
    _ = gr.HTML(FOOTER_HTML)

iface.queue(
    max_size=QUEUE_MAX_SIZE,
)

iface.launch(
    server_port=cmd_line_args['port'],
    max_threads = REQUESTS_CONCURRENCY
)
