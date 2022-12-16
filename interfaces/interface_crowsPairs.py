import gradio as gr
import pandas as pd
from tool_info import TOOL_INFO
from modules.module_logsManager import HuggingFaceDatasetSaver
from modules.module_connection import CrowsPairsExplorerConnector



def interface(
    language_model: str, 
    available_logs: bool, 
    lang: str="es"
) -> gr.Blocks:

    # -- Load examples --
    if lang == 'es':
        from examples.examples_es import examples_crows_pairs
    elif lang == 'en':
        from examples.examples_en import examples_crows_pairs

    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs,
        dataset_name=f"logs_edia_lmodels_{lang}"
    )

    # --- Init vars ---
    connector = CrowsPairsExplorerConnector(
        language_model=language_model
    )
    
    # --- Load language ---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["CrowsPairs_interface"]

    # --- Interface ---
    iface = gr.Blocks(
        css=".container {max-width: 90%; margin: auto;}"
    )

    with iface:
        with gr.Row():
            gr.Markdown(
                value=labels["title"]
            )
        
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    sent0 = gr.Textbox(
                        label=labels["sent0"],
                        placeholder=labels["commonPlacholder"]
                    )
                    sent2 = gr.Textbox(
                        label=labels["sent2"],
                        placeholder=labels["commonPlacholder"]
                    )
                    sent4 = gr.Textbox(
                        label=labels["sent4"],
                        placeholder=labels["commonPlacholder"]
                    )

            with gr.Column():
                with gr.Group():
                    sent1 = gr.Textbox(
                        label=labels["sent1"],
                        placeholder=labels["commonPlacholder"]
                    )
                    sent3 = gr.Textbox(
                        label=labels["sent3"],
                        placeholder=labels["commonPlacholder"]
                    )
                    sent5 = gr.Textbox(
                        label=labels["sent5"],
                        placeholder=labels["commonPlacholder"]
                    )

        with gr.Row():  
            btn = gr.Button(
                value=labels["compareButton"]
            )
        with gr.Row():  
            out_msj = gr.Markdown(
                value=""
            )
        
        with gr.Row():
            with gr.Group():
                gr.Markdown(
                    value=labels["plot"]
                )
                dummy = gr.CheckboxGroup(
                    value="", 
                    show_label=False, 
                    choices=[]
                )
                out = gr.HTML(
                    label=""
                )

        with gr.Row():
            examples = gr.Examples(
                inputs=[sent0, sent1, sent2, sent3, sent4, sent5],
                examples=examples_crows_pairs,
                label=labels["examples"]
            )

        with gr.Row(): 
            gr.Markdown(
                value=TOOL_INFO
            )

        btn.click(  
            fn=connector.compare_sentences,
            inputs=[sent0, sent1, sent2, sent3, sent4, sent5],
            outputs=[out_msj, out, dummy]
        )

        # --- Logs ---
        save_field = [sent0, sent1, sent2, sent3, sent4, sent5]
        log_callback.setup(
            components=save_field, 
            flagging_dir=f"logs_crows_pairs"
        )
        
        btn.click(
            fn=lambda *args: log_callback.flag(
                flag_data=args,
                flag_option="crows_pairs",
                username="vialibre"
            ),
            inputs=save_field,
            outputs=None, 
            preprocess=False
        )
    
    return iface