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
            with gr.Column(scale=30):
                token_id = gr.Textbox(
                    label=labels["token_id"],
                    lines=1,
                )
            with gr.Column(scale=70):
                with gr.Row():
                    with gr.Column():
                        school = gr.Number(
                            value=0,
                            label=labels["school"],
                        )
                        school_name = gr.HTML(
                            value=labels["notschool"],
                        )
                    with gr.Column():
                        _ = gr.HTML(
                            value=labels["ref"],
                        )
        with gr.Row():
            with gr.Column():
                age = gr.Number(
                    value=0,
                    label=labels["age"],
                )
            with gr.Column():
                gender = gr.Radio(
                    labels["gender_options"],
                    label=labels["gender"],
                )
            with gr.Column():
                with gr.Row():
                    consent_checkbox = gr.Checkbox(
                        label=labels["terms"],
                        value=False
                    )
                    _ = gr.HTML(
                        value="<a href='https://docs.google.com/document/d/1v7XTX7pFJ8SUv0JbwY5yXsISH61k5GRWdDqWz6PFrls/edit'>Link 🔗</a>",
                    )
                
        with gr.Row():
            with gr.Column():
                model_name = gr.Radio(
                    labels["language_options"],
                    value="Modelo en español",
                    info=labels["languagemodel"],
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
                        choices=labels["choices"],
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

        def update_school_name(school):
            if school is None or school == 0:
                return (
                    gr.HTML(
                        value=f"<p>No seleccionaste ningún colegio</p>",
                    )
                )
            elif school not in school_list:
                return (
                    gr.HTML(
                        value=f"<p>El colegio seleccionado no existe</p>",
                    )
                )
            else:
                return (
                    gr.HTML(
                        value=f"<p>Seleccionaste: {school_list[school]}</p>",
                    )
                )
        school.change(
            fn=update_school_name,
            inputs=[
                school
            ],
            outputs=[school_name])

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