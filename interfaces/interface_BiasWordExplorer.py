import gradio as gr
import pandas as pd
from tool_info import TOOL_INFO
from modules.module_connection import BiasWordExplorerConnector


# --- Interface ---
def interface(
    embedding, # Class Embedding instance
    available_logs: bool,
    lang: str="es"
) -> gr.Blocks:

    # -- Load examples ---
    if lang == 'es':
        from examples.examples_es import examples1_explorar_sesgo_en_palabras, examples2_explorar_sesgo_en_palabras
    elif lang == 'en':
        from examples.examples_en import examples1_explorar_sesgo_en_palabras, examples2_explorar_sesgo_en_palabras


    # --- Init vars ---
    connector = BiasWordExplorerConnector(
        embedding=embedding,
        lang=lang,
        logs_file_name = f"logs_edia_we_wordbias_{lang}" if available_logs else None
    )

    # --- Load language ---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["BiasWordExplorer_interface"]

    # --- Interface ---
    interface = gr.Blocks()

    with interface:
        token_id = gr.Textbox(
            label="TokenID",
            placeholder='Ingrese su TokenID',
            lines=1,
            show_label=False
        )
        gr.Markdown(
            value=labels["step1"]
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    diagnose_list = gr.Textbox(
                        lines=2,
                        label=labels["wordListToDiagnose"]
                    )
                with gr.Row():
                    gr.Markdown(
                        value=labels["step2&2Spaces"]
                    )
                with gr.Row():
                    wordlist_1 = gr.Textbox(
                        lines=2,
                        label=labels["wordList1"]
                    )
                    wordlist_2 = gr.Textbox(
                        lines=2, 
                        label=labels["wordList2"]
                    )
                with gr.Row():
                    gr.Markdown(
                        value=labels["step2&4Spaces"]
                    )
                with gr.Row():
                    wordlist_3 = gr.Textbox(
                        lines=2, 
                        label=labels["wordList3"]
                    )
                    wordlist_4 = gr.Textbox(
                        lines=2, 
                        label=labels["wordList4"]
                    )

            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        bias2d = gr.Button(
                            value=labels["plot2SpacesButton"]
                        )
                    with gr.Row():
                        bias4d = gr.Button(
                            value=labels["plot4SpacesButton"]
                        )
                    with gr.Row():
                        highlight_request = gr.Checkbox(
                            label='Destacar consulta',
                            value=False
                        )
                with gr.Row():
                    err_msg = gr.Markdown(
                        label="", 
                        visible=True
                    )
                with gr.Row():
                    bias_plot = gr.Plot(
                        label="", 
                        show_label=False
                    )

        with gr.Row():
            examples = gr.Examples(
                fn=connector.calculate_bias_2d,
                inputs=[wordlist_1, wordlist_2, diagnose_list],
                outputs=[bias_plot, err_msg],
                examples=examples1_explorar_sesgo_en_palabras,
                label=labels["examples2Spaces"]
            )
        with gr.Row():
            examples = gr.Examples(
                fn=connector.calculate_bias_4d,
                inputs=[wordlist_1, wordlist_2,wordlist_3, wordlist_4, diagnose_list],
                outputs=[
                    bias_plot, err_msg
                ],
                examples=examples2_explorar_sesgo_en_palabras,
                label=labels["examples4Spaces"]
            )

        with gr.Row():
            gr.Markdown(
                value=TOOL_INFO
            )

        bias2d.click(
            fn=connector.calculate_bias_2d,
            inputs=[wordlist_1, wordlist_2, diagnose_list, token_id, highlight_request],
            outputs=[bias_plot, err_msg],
            api_name="bias_we_2d"
        )

        bias4d.click(
            fn=connector.calculate_bias_4d,
            inputs=[
                wordlist_1, 
                wordlist_2,
                wordlist_3, 
                wordlist_4, 
                diagnose_list,
                token_id, 
                highlight_request
            ],
            outputs=[bias_plot, err_msg],
            api_name="bias_we_4d"
        )

    return interface
