import gradio as gr
import pandas as pd

from modules.module_logsManager import HuggingFaceDatasetSaver
from modules.module_connection import BiasWordExplorerConnector
from tool_info import TOOL_INFO


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


    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs,
        dataset_name=f"logs_edia_we_{lang}"
    )

    # --- Init vars ---
    connector = BiasWordExplorerConnector(
        embedding=embedding
    )

    # --- Load language ---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["BiasWordExplorer_interface"]

    # --- Interface ---
    interface = gr.Blocks()

    with interface:
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
                with gr.Row():
                    bias2d = gr.Button(
                        value=labels["plot2SpacesButton"]
                    )
                with gr.Row():
                    bias4d = gr.Button(
                        value=labels["plot4SpacesButton"]
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
            inputs=[wordlist_1, wordlist_2, diagnose_list],
            outputs=[bias_plot, err_msg]
        )

        bias4d.click(
            fn=connector.calculate_bias_4d,
            inputs=[wordlist_1, wordlist_2,
                    wordlist_3, wordlist_4, diagnose_list],
            outputs=[bias_plot, err_msg]
        )

        # --- Logs ---
        save_field = [wordlist_1, wordlist_2,wordlist_3, wordlist_4, diagnose_list]
        log_callback.setup(
            components=save_field,
            flagging_dir="logs_word_bias"
        )

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
