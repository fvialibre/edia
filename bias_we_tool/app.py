# --- Imports libs ---
import gradio as gr


# --- Imports modules ---
from modules.model_embbeding import Embedding
from modules.module_vocabulary import Vocabulary

# --- Imports interfaces ---
from interfaces.interface_ExplorarPalabras import interface as interface_explorar_palabras
from interfaces.interface_ExplorarSesgoEnPalabras import interface as interface_sesgo_en_palabras
from interfaces.interface_datos import interface as interface_datos

# --- Tool config ---
AVAILABLE_LOGS      = False
EMBEDDING_SUBSET   = "mini"                                        # [full | mini]
VOCABULARY_SUBSET   = "mini"                        # [full | semi | half | mini]
CONTEXTS_DATASET    = "nanom/splittedspanish3bwc"

# --- Init classes ---
embedding = Embedding(
    subset_name=EMBEDDING_SUBSET
)
vocabulary = Vocabulary(
    subset_name=VOCABULARY_SUBSET
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
    available_logs=AVAILABLE_LOGS
    )
]


TAB_NAMES = [
    "Explorar palabras",
    "Sesgo en palabras",
    "Explorar datos"
]

iface = gr.TabbedInterface(
    interface_list=INTERFACE_LIST, 
    tab_names=TAB_NAMES
)

iface.queue(concurrency_count=8)
iface.launch(debug=False)
