# --- Imports libs ---
import gradio as gr
import pandas as pd


# --- Imports modules ---
from modules.model_embbeding import Embedding
from modules.module_vocabulary import Vocabulary
from modules.module_languageModel import LanguageModel

# --- Imports interfaces ---
from interfaces.interface_WordExplorer import interface as interface_explorar_palabras
from interfaces.interface_BiasWordExplorer import interface as interface_sesgo_en_palabras
from interfaces.interface_datos import interface as interface_datos
from interfaces.interface_sesgoEnFrases import interface as interface_sesgoEnFrases
from interfaces.interface_crowsPairs import interface as interface_crowsPairs

# --- Tool config ---
AVAILABLE_LOGS      = False
LANGUAGE            = "spanish"                     # [spanish | english]
EMBEDDINGS_PATH     = "data/fasttext-sbwc.100k.vec"
MAX_NEIGHBORS       = 20
VOCABULARY_SUBSET   = "mini"                        # [full | semi | half | mini]
CONTEXTS_DATASET    = "nanom/splittedspanish3bwc"
LANGUAGE_MODEL      = "dccuchile/bert-base-spanish-wwm-uncased"

# --- Init classes ---
embedding = Embedding(
    path=EMBEDDINGS_PATH,
    binary=EMBEDDINGS_PATH.endswith('.bin'),
    limit=None,
    randomizedPCA=False,
    max_neighbors=MAX_NEIGHBORS
)
vocabulary = Vocabulary(
    subset_name=VOCABULARY_SUBSET
)
beto_lm = LanguageModel(
    model_name=LANGUAGE_MODEL
)
labels = pd.read_json(f"language/{LANGUAGE}.json")["app"]

# --- Main App ---
INTERFACE_LIST = [
    interface_sesgo_en_palabras(
        embedding=embedding,
        available_logs=AVAILABLE_LOGS,
        lang=LANGUAGE),
    interface_explorar_palabras(
        embedding=embedding,
        available_logs=AVAILABLE_LOGS,
        max_neighbors=MAX_NEIGHBORS,
        lang=LANGUAGE),
    interface_datos(
        vocabulary=vocabulary,
        contexts=CONTEXTS_DATASET,
        available_logs=AVAILABLE_LOGS,
        lang=LANGUAGE),
    interface_sesgoEnFrases(
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

iface = gr.TabbedInterface(
    interface_list=INTERFACE_LIST, 
    tab_names=TAB_NAMES
)

iface.queue(concurrency_count=8)
iface.launch(debug=False)
