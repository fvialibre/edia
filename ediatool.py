# --- Imports libs ---
import os
import gradio as gr
import pandas as pd
import configparser


# --- Imports modules ---
from modules.model_embbeding import Embedding
from modules.module_vocabulary import Vocabulary
from modules.module_languageModel import LanguageModel


# --- Imports interfaces ---
from interfaces.interface_WordExplorer import interface as interface_wordExplorer
from interfaces.interface_BiasWordExplorer import interface as interface_biasWordExplorer
from interfaces.interface_data import interface as interface_data
from interfaces.interface_biasPhrase import interface as interface_biasPhrase
from interfaces.interface_crowsPairs import interface as interface_crowsPairs


# --- Tool config ---
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
INTERFACE_LIST = [
    interface_biasWordExplorer(
        embedding=embedding,
        available_logs=AVAILABLE_LOGS,
        lang=LANGUAGE),
    interface_wordExplorer(
        embedding=embedding,
        available_logs=AVAILABLE_LOGS,
        max_neighbors=MAX_NEIGHBORS,
        lang=LANGUAGE),
    interface_data(
        vocabulary=vocabulary,
        contexts=CONTEXTS_DATASET,
        available_logs=AVAILABLE_LOGS,
        available_wordcloud=AVAILABLE_WORDCLOUD,
        lang=LANGUAGE),
    interface_biasPhrase(
        language_model=beto_lm,
        available_logs=AVAILABLE_LOGS,
        lang=LANGUAGE),
    interface_crowsPairs(
        language_model=beto_lm,
        available_logs=AVAILABLE_LOGS,
        lang=LANGUAGE),
]

TAB_NAMES = [
    labels["biasWordExplorer"],
    labels["wordExplorer"],
    labels["dataExplorer"],
    labels["phraseExplorer"],
    labels["crowsPairsExplorer"]
]

if LANGUAGE != 'es':
    # Skip data tab when using other than spanish language
    INTERFACE_LIST = INTERFACE_LIST[:2] + INTERFACE_LIST[3:]
    TAB_NAMES = TAB_NAMES[:2] + TAB_NAMES[3:]

iface = gr.TabbedInterface(
    interface_list= INTERFACE_LIST,
    tab_names=TAB_NAMES,
    title='EDIA Tool'
)

iface.queue(
    max_size=QUEUE_MAX_SIZE, 
    concurrency_count=REQUESTS_CONCURRENCY
)

iface.launch()