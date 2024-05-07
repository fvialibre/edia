import gradio as gr
import pandas as pd
from tool_info import TOOL_INFO
from modules.module_connection import BiasWordExplorerConnector


# --- Interface ---
def interface(
    embedding, # Class Embedding instance
    available_logs: bool,
    lang: str="es",
    user_email: str="",
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
            value=user_email,
            visible=False
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    value=labels["step1"]
                )
                with gr.Row():
                    diagnose_list = gr.Textbox(
                        lines=2,
                        show_label=False, 
                        placeholder=labels["step1_placeholder"],
                        container=False,
                    )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            value=labels["conceptA"]
                        )
                        wordlist_1 = gr.Textbox(
                            lines=2,
                            label=labels["wordList1"],
                            placeholder=labels["step1_placeholder"],
                            container=False,
                        )
                    with gr.Column():
                        gr.Markdown(
                            value=labels["conceptB"]
                        )
                        wordlist_2 = gr.Textbox(
                            lines=2, 
                            label=labels["wordList2"],
                            placeholder=labels["step1_placeholder"],
                            container=False,
                        )
                with gr.Row():
                    gr.Markdown(
                        value=labels["step2&4Spaces"],
                        visible=False
                    )
                with gr.Row():
                    wordlist_3 = gr.Textbox(
                        lines=2, 
                        label=labels["wordList3"],
                        container=False,
                        visible=False
                    )
                    wordlist_4 = gr.Textbox(
                        lines=2, 
                        label=labels["wordList4"],
                        container=False,
                        visible=False
                    )
                with gr.Row():
                    with gr.Group():
                        with gr.Row():
                            bias2d = gr.Button(
                                value=labels["plot2SpacesButton"]
                            )
                        with gr.Row():
                            bias4d = gr.Button(
                                value=labels["plot4SpacesButton"],
                                visible=False
                            )
                        with gr.Row():
                            with gr.Row():
                                highlight_query = gr.Checkbox(
                                    label=labels['highlight_query'],
                                    value=False,
                                    visible=False
                                )
                                type_of_bias_explored = gr.Dropdown(
                                    choices=[
                                        "Apariencia Física",
                                        "Discapacidad",
                                        "Edad",
                                        "Etnia",
                                        "Estado Socioeconómico",
                                        "Género",
                                        "Nacionalidad",
                                        "Orientación sexual",
                                        "Profesión",
                                        "Religión",
                                    ],
                                    label=labels["type_of_bias_explored"],
                                    multiselect=True,
                                    allow_custom_value=True
                                )
            with gr.Column():
                gr.Markdown(
                    value=labels["plot"]
                )
                err_msg = gr.Markdown(
                    label="", 
                    visible=True
                )
                bias_plot = gr.Plot(
                    label="", 
                    show_label=False
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

        with gr.Group():
            with gr.Row():
                btn_get_logs = gr.Button(
                    value=labels["see_queries_made"]
                )
            with gr.Row():
                df_get_logs = gr.DataFrame(
                    value=pd.DataFrame([], columns=['']),
                    label=None
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
                gr.Textbox(
                    value=f"logs_edia_we_wordbias_{lang}" if available_logs else None,
                    visible=False
                )
            ],
            outputs=[err_msg, df_get_logs],
        )

    return interface
