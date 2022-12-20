import gradio as gr
import pandas as pd
from tool_info import TOOL_INFO
from modules.module_connection import Word2ContextExplorerConnector


def interface(
    vocabulary, # Vocabulary class instance
    contexts: str,
    available_logs: bool,
    available_wordcloud: bool,
    lang: str="es"
) -> gr.Blocks:

    # --- Init Class ---
    connector = Word2ContextExplorerConnector(
        vocabulary=vocabulary, 
        context=contexts,
        logs_file_name=f"logs_edia_datos_{lang}" if available_logs else None
    )

    # --- Load language ---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["DataExplorer_interface"]

    # --- Interface ---
    iface = gr.Blocks(
        css=".container { max-width: 90%; margin: auto;}"
    )

    with iface:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown(
                        value=labels["step1"]
                    )
                    with gr.Row(): 
                        input_word = gr.Textbox(
                            label=labels["inputWord"]["title"], 
                            show_label=False, 
                            placeholder=labels["inputWord"]["placeholder"]
                        )
                    with gr.Row(): 
                        btn_get_w_info = gr.Button(
                            value=labels["wordInfoButton"]
                        )

                with gr.Group():
                    gr.Markdown(
                        value=labels["step2"]
                    )
                    n_context = gr.Slider(
                        label="", 
                        step=1, minimum=1, maximum=30, value=5, 
                        visible=True, 
                        interactive=True
                    )
                with gr.Group():
                    gr.Markdown(
                        value=labels["step3"]
                    )
                    subsets_choice = gr.CheckboxGroup(
                        label="Conjuntos",
                        show_label=False,
                        interactive=True, 
                        visible=True
                    )
                    with gr.Row(): 
                        btn_get_contexts = gr.Button(
                            value=labels["wordContextButton"], 
                            visible=True
                        )
                
                with gr.Row(): 
                    out_msj = gr.Markdown(
                        label="", 
                        visible=True
                    )

            with gr.Column():
                with gr.Group():
                    gr.Markdown(
                        value=labels["wordDistributionTitle"]
                    )
                    dist_plot = gr.Plot(
                        label="", 
                        show_label=False
                    )
                    wc_plot = gr.Plot(
                        label="", 
                        show_label=False, 
                        visible=available_wordcloud
                    )

                with gr.Group():
                    gr.Markdown(
                        value=labels["frequencyPerSetTitle"]
                    )
                    subsets_freq = gr.HTML(
                        label=""
                    )
    
        with gr.Row():
            with gr.Group():
                with gr.Row(): 
                    gr.Markdown(
                        value=labels["contextList"]
                    )
                with gr.Row(): 
                    out_context = gr.Dataframe(
                        label="", 
                        interactive=False, 
                        value=pd.DataFrame([], columns=['']),
                        wrap=True,
                        datatype=['str','markdown','str','markdown']
                    )

        with gr.Group():
            gr.Markdown(
                value=TOOL_INFO
            )

        btn_get_w_info.click( 
            fn=connector.get_word_info, 
            inputs=[input_word], 
            outputs=[out_msj,
                    out_context,
                    subsets_freq,
                    dist_plot,
                    wc_plot,
                    subsets_choice
            ]
        )
        
        btn_get_contexts.click(
            fn=connector.get_word_context, 
            inputs=[input_word, n_context, subsets_choice], 
            outputs=[out_msj, out_context]
        )
    
    return iface