# --- Imports libs ---
import gradio as gr


# --- Imports modules ---
from modules.model_embbeding import Embedding
from modules.module_vocabulary import Vocabulary
from modules.module_languageModel import LanguageModel

# --- Imports interfaces ---
from interfaces.interface_ExplorarPalabras import interface as interface_explorar_palabras
from interfaces.interface_ExplorarSesgoEnPalabras import interface as interface_sesgo_en_palabras
from interfaces.interface_datos import interface as interface_datos
from interfaces.interface_sesgoEnFrases import interface as interface_sesgoEnFrases
from interfaces.interface_crowsPairs import interface as interface_crowsPairs

# --- Tool config ---
AVAILABLE_LOGS      = False
EMBEDDING_SUBSET   = "mini"                                        # [full | mini]
VOCABULARY_SUBSET   = "mini"                        # [full | semi | half | mini]
CONTEXTS_DATASET    = "nanom/splittedspanish3bwc"
LANGUAGE_MODEL      = "dccuchile/bert-base-spanish-wwm-uncased"

# --- Init classes ---
embedding = Embedding(
    subset_name=EMBEDDING_SUBSET
)
vocabulary = Vocabulary(
    subset_name=VOCABULARY_SUBSET
)
beto_lm = LanguageModel(
    model_name=LANGUAGE_MODEL
)


# --- Main App ---
INTERFACE_LIST = [
    interface_explorar_palabras(
        embedding=embedding,
        available_logs=AVAILABLE_LOGS),
    interface_sesgo_en_palabras(
        embedding=embedding,
        available_logs=AVAILABLE_LOGS),
    interface_datos(
        vocabulary=vocabulary,
        contexts=CONTEXTS_DATASET,
        available_logs=AVAILABLE_LOGS),
    interface_sesgoEnFrases(
        language_model=beto_lm,
        available_logs=AVAILABLE_LOGS),
    interface_crowsPairs(
        language_model=beto_lm,
        available_logs=AVAILABLE_LOGS),
]


TAB_NAMES = [
    "Explorar palabras",
    "Sesgo en palabras",
    "Explorar datos",
    "Sesgo en frases",
    "Crows-Pairs"
]

iface = gr.TabbedInterface(
    interface_list=INTERFACE_LIST, 
    tab_names=TAB_NAMES
)

iface.queue(concurrency_count=8)
iface.launch(debug=False)