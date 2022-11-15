from tkinter import image_names
import gradio as gr

from modules.module_BiasExplorer import  WEBiasExplorer2d, WEBiasExplorer4d
from examples.examples import examples1_explorar_sesgo_en_palabras, examples2_explorar_sesgo_en_palabras
from modules.module_logsManager import HuggingFaceDatasetSaver
from modules.module_connection import BiasWordExplorerConnector
from tool_info import TOOL_INFO

LABEL_WORD_LIST_1 = 'Lista de palabras 1'
LABEL_WORD_LIST_2 = 'Lista de palabras 2'
LABEL_WORD_LIST_3 = 'Lista de palabras 3'
LABEL_WORD_LIST_4 = 'Lista de palabras 4'

LABEL_WORD_LIST_DIAGNOSE = 'Lista de palabras a diagnosticar'

# def make_gallery(plot,saved_images):
#     saved_images = saved_images.append(plot)
#     image_list = [i for i in saved_images]
#     gallery= gr.Gallery(value=image_list)
#     return saved_images, image_list

# --- Interface ---
def interface(embedding,available_logs):
    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs
    )
    # --- Init vars ---
    we_bias = WEBiasExplorer2d(embedding)
    we_bias_2d = WEBiasExplorer4d(embedding)
    connector = BiasWordExplorerConnector(embedding=embedding)
    # saved_images = gr.State([])

    interface = gr.Blocks()
    with interface:
        gr.Markdown("1. Escribi palabras para diagnosticar separadas por comas")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    diagnose_list = gr.Textbox(lines=2, label=LABEL_WORD_LIST_DIAGNOSE)
                with gr.Row():
                    gr.Markdown("2. Para graficar 2 espacios, completa las siguientes listas:")
                with gr.Row():
                    wordlist_1 = gr.Textbox(lines=2, label=LABEL_WORD_LIST_1)
                    wordlist_2 = gr.Textbox(lines=2, label=LABEL_WORD_LIST_2)
                with gr.Row():
                    gr.Markdown("2. Para graficar 4 espacios, además completa las siguientes listas:")
                with gr.Row():
                    wordlist_3 = gr.Textbox(lines=2, label=LABEL_WORD_LIST_3)
                    wordlist_4 = gr.Textbox(lines=2, label=LABEL_WORD_LIST_4)
            with gr.Column():
                with gr.Row():
                    bias2d = gr.Button('¡Graficar 2 estereotipos!')
                with gr.Row():
                    bias4d = gr.Button('¡Graficar 4 estereotipos!')    
                # with gr.Row():
                #     save_image = gr.Button('Guardar exploracion')    
                with gr.Row():
                    err_msg = gr.Markdown(label='',visible=True)
                with gr.Row():
                    bias_plot = gr.Image(shape=(15, 15))
            # with gr.Accordion(label='Exploraciones guardadas'):
            #     gallery = gr.Gallery(value=[])     
        with gr.Row():
            examples = gr.Examples(
                fn=connector.calculate_bias_2d,
                inputs=[wordlist_1, wordlist_2, diagnose_list],
                outputs=[bias_plot, err_msg],
                examples=examples1_explorar_sesgo_en_palabras
            )
        with gr.Row():
            examples = gr.Examples(
                fn=connector.calculate_bias_4d,
                inputs=[wordlist_1, wordlist_2,
                        wordlist_3, wordlist_4, diagnose_list],
                outputs=[bias_plot, err_msg],
                examples=examples2_explorar_sesgo_en_palabras
            )

        with gr.Row():
            gr.Markdown(TOOL_INFO)

        bias2d.click(
            fn=connector.calculate_bias_2d, 
            inputs=[wordlist_1,wordlist_2,diagnose_list],
            outputs=[bias_plot,err_msg]
        )
            
        bias4d.click(
            fn=connector.calculate_bias_4d,
            inputs=[wordlist_1,wordlist_2,wordlist_3,wordlist_4,diagnose_list],
            outputs=[bias_plot,err_msg]
        )
        # save_image.click(
        #     fn=make_gallery,
        #     inputs=[bias_plot,saved_images],
        #     outputs=[saved_images,gallery]
        # )
        # --- Logs ---
        save_field = [wordlist_1,wordlist_2,wordlist_3,wordlist_4,diagnose_list]
        log_callback.setup(components=save_field, flagging_dir="sesgo_en_palabras")

        bias2d.click(
            fn=lambda *args: log_callback.flag(
                    flag_data=args, 
                    flag_option="plot_2d",
                    username="vialibre"
            ),
            inputs=save_field,
            outputs=None, 
            preprocess=False
        )
        
        bias4d.click(
            fn=lambda *args: log_callback.flag(
                    flag_data=args,
                    flag_option="plot_4d",
                    username="vialibre"
            ),
            inputs=save_field,
            outputs=None, 
            preprocess=False
        )
    return interface