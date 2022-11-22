from modules.module_word2Context import Word2Context
from modules.module_logsManager import HuggingFaceDatasetSaver
from examples.examples import examples_datos
from tool_info import TOOL_INFO
import gradio as gr
import pandas as pd
from modules.module_connection import Word2ContextExplorerConnector

def interface(vocabulary, contexts, available_logs, lang="spanish"):

    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs
    )

    # --- Init Class ---
    connector = Word2ContextExplorerConnector(vocabulary=vocabulary, context=contexts)
    labels = pd.read_json(f"language/{lang}.json")["DataExplorer_interface"]

    # --- Interface ---
    iface = gr.Blocks(css=".container { max-width: 90%; margin: auto;}")

    with iface:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown(labels["step1"])
                    with gr.Row(): input_word = gr.Textbox(label=labels["inputWord"]["title"], 
                                                            show_label=False, 
                                                            placeholder=labels["inputWord"]["placeholder"])
                    with gr.Row(): btn_get_w_info = gr.Button(labels["wordInfoButton"])

                with gr.Group():
                    gr.Markdown(labels["step2"])
                    n_context = gr.Slider(label="", 
                                        step=1, minimum=1, maximum=30, value=5, 
                                        visible=True, interactive=True)
                with gr.Group():
                    gr.Markdown(labels["step3"])
                    subsets_choice = gr.CheckboxGroup(label="", 
                                        interactive=True, 
                                        visible=True)
                    with gr.Row(): btn_get_contexts = gr.Button(labels["wordContextButton"], visible=True)
                
                with gr.Row(): out_msj = gr.Markdown(label="", visible=True)

            with gr.Column():
                with gr.Group():
                    gr.Markdown(labels["wordDistributionTitle"])
                    dist_plot = gr.Plot(label="", show_label=False)
                    wc_plot = gr.Plot(label="", show_label=False,)

                with gr.Group():
                    gr.Markdown(labels["frequencyPerSetTitle"])
                    # subsets_freq = gr.Label(label="Frecuencias de aparición p/subconjunto:", 
                    #     num_top_classes=16, visible=True, show_label=False)
                    subsets_freq = gr.HTML(label="")
    
        with gr.Row():
            with gr.Group():
                with gr.Row(): gr.Markdown(labels["contextList"])
                with gr.Row(): out_context = gr.Dataframe(label="", 
                                                interactive=False, 
                                                value=pd.DataFrame([], columns=['']),
                                                wrap=True,
                                                datatype=['str','markdown','str','markdown'])

        with gr.Group():
            gr.Markdown(TOOL_INFO)

        btn_get_w_info.click( 
            fn=connector.get_word_info, 
            inputs=[input_word], 
            outputs=[out_msj,
                    out_context,
                    subsets_freq,
                    dist_plot,
                    wc_plot,
                    subsets_choice]
        )
        
        btn_get_contexts.click(     
            fn=connector.get_word_context, 
            inputs=[input_word, n_context, subsets_choice], 
            outputs=[out_msj, out_context]
        )
        
        # --- Logs ---
        save_field = [input_word, subsets_choice]
        log_callback.setup(components=save_field, flagging_dir="datos")
        
        btn_get_contexts.click(
            fn=lambda *args: log_callback.flag(
                    flag_data=args,
                    flag_option="datos",
                    username="vialibre"
            ),
            inputs=save_field,
            outputs=None, 
            preprocess=False
        )
    
    return iface