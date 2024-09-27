import gradio as gr
import pandas as pd
from modules.module_connection import BiasWordExplorerConnector
from auth import school_list


# --- Interface ---
def interface(
    embedding, # Class Embedding instance
    available_logs: bool,
    lang: str="es",
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
        with gr.Row():
            with gr.Column():
                token_id = gr.Textbox(
                    label="Escriba su correo electrónico",
                    lines=1,
                )
            with gr.Column():
                school = gr.Dropdown(
                    choices=school_list,
                    label="Seleccione su escuela",
                    # info="Seleccione su escuela",
                    multiselect=False,
                    allow_custom_value=False,
                )
            with gr.Column():
                age = gr.Dropdown(
                    choices=[str(i) for i in range(1, 100)],
                    label="Seleccione su edad",
                    # info="Seleccione su edad",
                    multiselect=False,
                    allow_custom_value=False,
                )
            with gr.Column():
                gender = gr.Radio(
                    ["M", "F", "X"],
                    label="Seleccione su género",
                    # info="Where did they go?"
                )
            with gr.Column():
                with gr.Row():
                    consent_checkbox = gr.Checkbox(
                        label='He leído y acepto el consentimiento informado ➡️',
                        value=False
                    )
                    _ = gr.HTML(
                        value="<a href='https://docs.google.com/document/d/1yR_spvGWiq9ivuz4iOI8kzwI0Ope6chS/edit'>Link 🔗</a>",
                    )
        with gr.Row():
            with gr.Column():
                model_name = gr.Radio(
                    [
                        "Modelo en español",
                        "Modelo Multilenguaje",
                    ],
                    value="Modelo en español",
                    info="Elegí un modelo de lenguaje.",
                    container=False,
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column():
                        wordlist_1 = gr.Textbox(
                            lines=2,
                            label=labels["conceptA"],
                            info=labels["conceptA_info"],
                            placeholder=labels["conceptA_placeholder"],
                        )
                    with gr.Column():
                        wordlist_2 = gr.Textbox(
                            lines=2,
                            label=labels["conceptB"],
                            info=labels["conceptB_info"],
                            container=True,
                            placeholder=labels["conceptB_placeholder"],
                        )
                with gr.Row():
                    diagnose_list = gr.Textbox(
                        lines=2,
                        label=labels["step3"],
                        info=labels["step3_info"],
                        placeholder=labels["step3_placeholder"],
                    )
                with gr.Row():
                    type_of_bias_explored = gr.Dropdown(
                        choices=[
                            "Apariencia Física",
                            "Discapacidad",
                            "Edad",
                            "Etnia",
                            "Género",
                            "Nacionalidad",
                            "Orientación sexual",
                            "Profesión",
                            "Religión",
                            "Situación Socioeconómica",
                        ],
                        label=labels["type_of_bias_explored_label"],
                        info=labels["type_of_bias_explored_info"],
                        multiselect=True,
                        allow_custom_value=True
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
            with gr.Column():
                gr.Markdown(
                    value=labels["plot"]
                )
                err_msg = gr.HTML(
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
                label=labels["examples2Spaces"],
                elem_id="examples",
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

        bias2d.click(
            fn=connector.calculate_bias_2d,
            inputs=[
                wordlist_1,
                wordlist_2,
                diagnose_list,
                token_id,
                school,
                age,
                gender,
                consent_checkbox,
                highlight_query,
                type_of_bias_explored,
                model_name
            ],
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
                school,
                age,
                gender,
                consent_checkbox,
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
