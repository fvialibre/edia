import gradio as gr 
import pandas as pd
import matplotlib.pyplot as plt
from tool_info import TOOL_INFO
from modules.module_connection import WordExplorerConnector

plt.rcParams.update({'font.size': 14})

def interface(
    embedding, # Class Embedding instance
    available_logs: bool, 
    max_neighbors: int,
    lang: str="es",
) -> gr.Blocks:

    # -- Load examples ---
    if lang == 'es':
        from examples.examples_es import examples_explorar_relaciones_entre_palabras
    elif lang == 'en':
        from examples.examples_en import examples_explorar_relaciones_entre_palabras


    # --- Init vars ---
    connector = WordExplorerConnector(
        embedding=embedding,
        lang=lang,
        logs_file_name=f"logs_edia_we_wordexplorer_{lang}" if available_logs else None
    )

    # --- Load language ---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["WordExplorer_interface"]

    # --- Interface ---
    interface = gr.Blocks()

    with interface:
        token_id = gr.Textbox(
            placeholder=labels['token_id'],
            lines=1,
            show_label=False
        )
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    value=labels["title"]
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        diagnose_list = gr.Textbox(
                            lines=2, 
                            label=labels["wordListToDiagnose"]
                        )
                    with gr.Column(scale=1,min_width=10):
                        color_wordlist = gr.ColorPicker(
                            label="",
                            value='#000000'
                        )

                with gr.Row():
                    with gr.Column(scale=5): 
                        wordlist_1 = gr.Textbox(
                            lines=2, 
                            label=labels["wordList1"]
                        )
                    with gr.Column(scale=1,min_width=10): 
                        color_wordlist_1 = gr.ColorPicker(
                            label="",
                            value='#1f78b4'
                        )
                with gr.Row():
                    with gr.Column(scale=5): 
                        wordlist_2 = gr.Textbox(
                            lines=2, 
                            label=labels["wordList2"]
                        )
                    with gr.Column(scale=1,min_width=10):
                        color_wordlist_2 = gr.ColorPicker(
                            label="",
                            value='#33a02c'
                        )
                with gr.Row():
                    with gr.Column(scale=5):    
                        wordlist_3 = gr.Textbox(
                            lines=2, 
                            label=labels["wordList3"]
                        )
                    with gr.Column(scale=1,min_width=10): 
                        color_wordlist_3 = gr.ColorPicker(
                            label="",
                            value='#e31a1c'
                        )
                with gr.Row():
                    with gr.Column(scale=5):    
                        wordlist_4 = gr.Textbox(
                            lines=2, 
                            label=labels["wordList4"]
                        )
                    with gr.Column(scale=1,min_width=10): 
                        color_wordlist_4 = gr.ColorPicker(
                            label="",
                            value='#6a3d9a'
                        )
            with gr.Column(scale=4):
                gr.Markdown(
                    value=labels["plotNeighbours"]["title"]
                )
                with gr.Row():
                    n_neighbors = gr.Slider(
                        minimum=0,
                        maximum=max_neighbors,
                        step=1,
                        label=labels["plotNeighbours"]["quantity"]
                    )
                    alpha = gr.Slider(
                        minimum=0.1,
                        maximum=0.9, 
                        value=0.3, 
                        step=0.1,
                        label=labels["options"]["transparency"]
                    )
                    fontsize=gr.Number(
                        value=25, 
                        label=labels["options"]["font-size"]
                    )
                with gr.Group():
                    with gr.Row():
                        btn_plot = gr.Button(
                            value=labels["plot_button"]
                        )
                    with gr.Row():
                        highlight_query = gr.Checkbox(
                            label=labels['highlight_query'],
                            value=False
                        )
                with gr.Row(): 
                    err_msg = gr.Markdown(
                        label="", 
                        visible=True
                    )
                with gr.Row():
                    word_proyections = gr.Plot(
                        label="", 
                        show_label=False
                    )

        with gr.Row():
            gr.Examples(
                fn=connector.plot_proyection_2d,
                inputs=[diagnose_list,wordlist_1,wordlist_2,wordlist_3,wordlist_4],
                outputs=[word_proyections,err_msg],
                examples=examples_explorar_relaciones_entre_palabras,
                label=labels["examples"]
            )

        with gr.Row():
            gr.Markdown(
                value=TOOL_INFO
            )

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
                n_neighbors,
                token_id, highlight_query
            ],
            outputs=[word_proyections, err_msg],
            api_name="word_explorer"
        )
        
        return interface