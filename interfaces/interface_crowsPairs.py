import gradio as gr
import pandas as pd
from tool_info import TOOL_INFO
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


    # --- Init vars ---
    connector = CrowsPairsExplorerConnector(
        language_model=language_model,
        lang=lang,
        logs_file_name=f"logs_edia_lmodels_crowspairs_{lang}" if available_logs else None
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
        token_id = gr.Textbox(
            placeholder=labels['token_id'],
            lines=1,
            show_label=False
        )
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
            with gr.Group():
                btn = gr.Button(
                    value=labels["compareButton"]
                )
                highlight_query = gr.Checkbox(
                    label=labels['highlight_query'],
                    value=False
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
            inputs=[sent0, sent1, sent2, sent3, sent4, sent5, token_id, highlight_query],
            outputs=[out_msj, out, dummy],
            api_name="crows_pairs"
        )

    return iface