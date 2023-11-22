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
            placeholder=labels['token_id'],
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
                        with gr.Row():
                            highlight_query = gr.Checkbox(
                                label=labels['highlight_query'],
                                value=False
                            )
                        with gr.Row():
                            type_of_bias_explored = gr.Textbox(
                                placeholder=labels['type_of_bias_explored'],
                                lines=2,
                                show_label=False
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

        with gr.Group():
            with gr.Row():
                btn_get_logs = gr.Button(
                    value="Ver consultas anteriores"
                )
            with gr.Row():
                df_get_logs = gr.DataFrame(
                    value=pd.DataFrame([], columns=['']),
                    label=None
                )

        with gr.Row():
            examples = gr.Examples(
                inputs=[wordlist_1, wordlist_2, diagnose_list],
                examples=examples1_explorar_sesgo_en_palabras,
                label=labels["examples2Spaces"]
            )
        with gr.Row():
            examples = gr.Examples(
                inputs=[wordlist_1, wordlist_2,wordlist_3, wordlist_4, diagnose_list],
                examples=examples2_explorar_sesgo_en_palabras,
                label=labels["examples4Spaces"]
            )

        with gr.Row():
            gr.Markdown(
                value=TOOL_INFO
            )

        bias2d.click(
            fn=connector.calculate_bias_2d,
            inputs=[wordlist_1, wordlist_2, diagnose_list, token_id, highlight_query, type_of_bias_explored],
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
                highlight_query,
                type_of_bias_explored
            ],
            outputs=[bias_plot, err_msg],
            api_name="bias_we_4d"
        )
        
        btn_get_logs.click(
            fn=connector.get_logs,
            inputs=[
                token_id,
                gr.Textbox(value=f"logs_edia_we_wordbias_{lang}" if available_logs else None)
            ],
            outputs=df_get_logs,
        )

    return interface
