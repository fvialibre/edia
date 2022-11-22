from modules.module_crowsPairs import CrowsPairs
from modules.module_logsManager import HuggingFaceDatasetSaver
from modules.module_connection import CrowsPairsExplorerConnector
from examples.examples import examples_crows_pairs
from tool_info import TOOL_INFO
import gradio as gr
import pandas as pd


def interface(language_model, available_logs, lang="spanish"):

    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs
    )

    # --- Init vars ---
    connector = CrowsPairsExplorerConnector(language_model=language_model)
    labels = pd.read_json(f"language/{lang}.json")["CrowsPairs_interface"]

    # --- Interface ---
    iface = gr.Blocks(css=".container {max-width: 90%; margin: auto;}")

    with iface:
        with gr.Row():
            gr.Markdown(labels["title"])
        
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    sent0 = gr.Textbox(label=labels["sent0"],
                                placeholder=labels["commonPlacholder"])
                    sent2 = gr.Textbox(label=labels["sent2"],
                                placeholder=labels["commonPlacholder"])
                    sent4 = gr.Textbox(label=labels["sent4"],
                                placeholder=labels["commonPlacholder"])

            with gr.Column():
                with gr.Group():
                    sent1 = gr.Textbox(label=labels["sent1"],
                                    placeholder=labels["commonPlacholder"])
                    sent3 = gr.Textbox(label=labels["sent3"],
                                    placeholder=labels["commonPlacholder"])
                    sent5 = gr.Textbox(label=labels["sent5"],
                                    placeholder=labels["commonPlacholder"])

        with gr.Row():  btn = gr.Button(value=labels["compareButton"])
        with gr.Row():  out_msj = gr.Markdown()
        
        with gr.Row():
            with gr.Group():
                gr.Markdown(labels["plot"])
                dummy = gr.CheckboxGroup(value="", show_label=False, choices=[])
                out = gr.HTML(label="")
                    
        
        with gr.Row():
            examples = gr.Examples(
                inputs = [sent0, sent1, sent2, sent3, sent4, sent5],
                examples = examples_crows_pairs,
                label=labels["examples"]
            )

        with gr.Row(): 
            gr.Markdown(TOOL_INFO)

        btn.click(  
            fn=connector.compare_sentences,
            inputs=[sent0, sent1, sent2, sent3, sent4, sent5],
            outputs=[out_msj, out, dummy]
        )

        # --- Logs ---
        save_field = [sent0, sent1, sent2, sent3, sent4, sent5]
        log_callback.setup(components=save_field, flagging_dir="crows_pairs")
        
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