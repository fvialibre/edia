# --- Imports libs ---
import gradio as gr


# --- Imports modules ---
from modules.model_embbeding import Embedding


# --- Imports interfaces ---
from interfaces.interface_ExplorarPalabras import interface as interface_explorar_palabras
from interfaces.interface_ExplorarSesgoEnPalabras import interface as interface_sesgo_en_palabras

# --- Tool config ---
AVAILABLE_LOGS      = False
EMBEDDING_SUBSET   = "mini"                                        # [full | mini]


# --- Init classes ---
embedding = Embedding(
    subset_name=EMBEDDING_SUBSET
)


# --- Main App ---
INTERFACE_LIST = [
    interface_explorar_palabras(
        embedding=embedding,
        available_logs=AVAILABLE_LOGS),
    interface_sesgo_en_palabras(
        embedding=embedding,
        available_logs=AVAILABLE_LOGS
    )
]


TAB_NAMES = [
    "Explorar palabras",
    "Sesgo en palabras"
]

iface = gr.TabbedInterface(
    interface_list=INTERFACE_LIST, 
    tab_names=TAB_NAMES
)

iface.queue(concurrency_count=8)
iface.launch(debug=False)
