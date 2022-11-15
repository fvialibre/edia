import gradio as gr 

from examples.examples import examples_explorar_relaciones_entre_palabras
from modules.module_logsManager import HuggingFaceDatasetSaver
from tool_info import TOOL_INFO
import matplotlib.pyplot as plt
from modules.module_connection import WordExplorerConnector

plt.rcParams.update({'font.size': 14})

LABEL_WORD_LIST_1 = 'Lista de palabras 1'
LABEL_WORD_LIST_2 = 'Lista de palabras 2'
LABEL_WORD_LIST_3 = 'Lista de palabras 3'
LABEL_WORD_LIST_4 = 'Lista de palabras 4'
LABEL_WORD_LIST_DIAGNOSE = 'Lista de palabras a diagnosticar'

def interface(embedding, available_logs):
    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs
    )
    # --- Init vars ---
    connector = WordExplorerConnector(embedding=embedding)

    # --- Interface ---
    interface = gr.Blocks()
    with interface:
        gr.Markdown("Escribi algunas palabras para visualizar sus palabras relacionadas")
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5):
                        diagnose_list = gr.Textbox(lines=2, label=LABEL_WORD_LIST_DIAGNOSE)
                    with gr.Column(scale=1,min_width=10):
                        color_wordlist = gr.ColorPicker(label="",value='#000000',)
                with gr.Row():
                    with gr.Column(scale=5): 
                        wordlist_1 = gr.Textbox(lines=2, label=LABEL_WORD_LIST_1)
                    with gr.Column(scale=1,min_width=10): 
                        color_wordlist_1 = gr.ColorPicker(label="",value='#1f78b4')
                with gr.Row():
                    with gr.Column(scale=5): 
                        wordlist_2 = gr.Textbox(lines=2, label=LABEL_WORD_LIST_2)
                    with gr.Column(scale=1,min_width=10):     
                        color_wordlist_2 = gr.ColorPicker(label="",value='#33a02c')
                with gr.Row():
                    with gr.Column(scale=5):    
                        wordlist_3 = gr.Textbox(lines=2, label=LABEL_WORD_LIST_3)
                    with gr.Column(scale=1,min_width=10): 
                        color_wordlist_3 = gr.ColorPicker(label="",value='#e31a1c')
                with gr.Row():
                    with gr.Column(scale=5):    
                        wordlist_4 = gr.Textbox(lines=2, label=LABEL_WORD_LIST_4)
                    with gr.Column(scale=1,min_width=10): 
                        color_wordlist_4 = gr.ColorPicker(label="",value='#6a3d9a')
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Row():
                        gr.Markdown('Graficar palabras relacionadas')
                        n_neighbors = gr.Slider(minimum=0,maximum=100,step=1,label='Cantidad')
                    with gr.Row():
                        alpha = gr.Slider(minimum=0.1,maximum=0.9, value=0.3, step=0.1,label='Transparencia')
                        fontsize=gr.Number(value=18, label='Tamaño de fuente')
                    with gr.Row():
                        btn_plot = gr.Button('¡Graficar en el espacio!')
                with gr.Row(): 
                    err_msg = gr.Markdown(label="", visible=True)
                with gr.Row():
                    word_proyections = gr.Image(shape=(10, 10))

        with gr.Row():
            gr.Examples(
                fn=connector.plot_proyection_2d,
                inputs=[diagnose_list,wordlist_1,wordlist_2,wordlist_3,wordlist_4],
                outputs=[word_proyections,err_msg],
                examples=examples_explorar_relaciones_entre_palabras
            )

        with gr.Row():
            gr.Markdown(TOOL_INFO)

        btn_plot.click(
            fn=connector.plot_proyection_2d,
            inputs=[
                diagnose_list,
                wordlist_1,
                wordlist_2,
                wordlist_3,
                wordlist_4,
                color_wordlist,
                color_wordlist_1,
                color_wordlist_2,
                color_wordlist_3,
                color_wordlist_4,
                alpha,
                fontsize,
                n_neighbors
            ],
            outputs=[word_proyections,err_msg]
        )

        # --- Logs ---
        save_field = [diagnose_list,wordlist_1,wordlist_2,wordlist_3,wordlist_4]
        log_callback.setup(components=save_field, flagging_dir="explorar_palabras")
        
        btn_plot.click(
            fn=lambda *args: log_callback.flag(
                    flag_data=args,
                    flag_option="explorar_palabras",
                    username="vialibre",
            ),
            inputs=save_field,
            outputs=None, 
            preprocess=False
        )
        return interface