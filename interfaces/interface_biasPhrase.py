import gradio as gr
import pandas as pd
from modules.module_connection import PhraseBiasExplorerConnector
from auth import school_list


def interface(
    spanish_language_model: str,
    english_language_model: str,
    available_logs: bool, 
    lang: str="es",
) -> gr.Blocks:

    # -- Load examples --
    if lang == 'es':
        from examples.examples_es import examples_sesgos_frases
    elif lang == 'en':
        from examples.examples_en import examples_sesgos_frases


    # --- Init vars ---
    connector = PhraseBiasExplorerConnector(
        spanish_language_model=spanish_language_model,
        english_language_model=english_language_model,
        lang=lang,
        logs_file_name=f"logs_edia_lmodels_biasphrase_{lang}" if available_logs else None
    )

    # --- Get language labels---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["PhraseExplorer_interface"]

    # --- Init Interface ---
    iface = gr.Blocks(
        css=".container {max-width: 90%; margin: auto;}"
    )

    with iface:
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
                    ["Modelo en español", "Modelo en inglés"],
                    value="Modelo en español",
                    info="Elegí un modelo de lenguaje.",
                    container=False,
                    interactive=True,
                )
                with gr.Row():
                    sent = gr.Textbox(
                        lines=2,
                        label=labels["step1"],
                        info=labels["step1_info"],
                        placeholder=labels["step1_placeholder"],
                    )
                with gr.Row():
                    word_list = gr.Textbox( 
                        lines=2,
                        label=labels["step2"],
                        info=labels["step2_info"],
                        placeholder=labels["step2_placeholder"],
                    )

                highlight_query = gr.Checkbox(
                    label=labels['highlight_query'],
                    value=False,
                    visible=False
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
                
                gr.Markdown(
                    value=labels["step3"],
                    visible=False,
                )
                banned_word_list = gr.Textbox( 
                    label=labels["bannedWordList"]["title"], 
                    placeholder=labels["bannedWordList"]["placeholder"],
                    visible=False,
                )
                with gr.Row():
                    with gr.Row(): 
                        articles = gr.Checkbox(
                            label=labels["excludeArticles"], 
                            value=False,
                            visible=False,
                        )
                    with gr.Row(): 
                        prepositions = gr.Checkbox(
                            label=labels["excludePrepositions"], 
                            value=False,
                            visible=False,
                        )
                    with gr.Row(): 
                        conjunctions = gr.Checkbox(
                            label=labels["excludeConjunctions"], 
                            value=False,
                            visible=False,
                        )

                with gr.Row():
                    with gr.Group():
                        btn = gr.Button(
                            value=labels["resultsButton"]
                        )

            with gr.Column():
                gr.Markdown(
                    value=labels["plot"]
                )
                out = gr.HTML(
                    label="",
                )
                out_msj = gr.HTML(
                    value="",
                )

        with gr.Row():
            _ = gr.Examples(
                inputs=[sent, word_list],
                examples=examples_sesgos_frases,
                label=labels["examples"],
                elem_id="examples"
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

        btn.click(
            fn=connector.rank_sentence_options,
            inputs=[
                sent, 
                word_list, 
                banned_word_list, 
                articles, 
                prepositions, 
                conjunctions,
                token_id,
                school,
                age,
                gender,
                consent_checkbox,
                highlight_query,
                type_of_bias_explored,
                model_name,
            ], 
            outputs=[out_msj, out],
            api_name="bias_phrase"
        )
        
        btn_get_logs.click(
            fn=connector.get_logs,
            inputs=[
                token_id,
                gr.Textbox(
                    value=f"logs_edia_lmodels_biasphrase_{lang}" if available_logs else None, visible=False
                )
            ],
            outputs=[out_msj, df_get_logs]
        )

    return iface