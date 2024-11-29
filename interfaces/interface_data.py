import gradio as gr
import pandas as pd
from modules.module_connection import Word2ContextExplorerConnector
from auth import school_list

def interface(
    vocabulary, # Vocabulary class instance
    contexts: str,
    available_logs: bool,
    available_wordcloud: bool,
    lang: str="es",
) -> gr.Blocks:

    # --- Init Class ---
    connector = Word2ContextExplorerConnector(
        vocabulary=vocabulary, 
        context=contexts,
        lang=lang,
        logs_file_name=f"logs_edia_datos_{lang}" if available_logs else None
    )

    # --- Load language ---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["DataExplorer_interface"]

    # --- Interface ---
    iface = gr.Blocks(
        css=".container { max-width: 90%; margin: auto;}"
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
                    input_word = gr.Textbox(
                        label=labels["step1"], 
                        info=labels["step1_info"],
                        placeholder=labels["inputWord"]["placeholder"],
                    )
                with gr.Row(): 
                    btn_get_w_info = gr.Button(
                        value=labels["wordInfoButton"]
                    )
                with gr.Row():
                    n_context = gr.Slider(
                        label=labels["step2"], 
                        info=labels["step2_info"],
                        step=1, minimum=1, maximum=30, value=5, 
                        visible=True, 
                        interactive=True,
                    )
                with gr.Row():
                    subsets_choice = gr.CheckboxGroup(
                        label=labels["step3"],
                        info=labels["step3_info"],
                        interactive=True, 
                        visible=True
                    )
                with gr.Group():
                    with gr.Row():
                        btn_get_contexts = gr.Button(
                            value=labels["wordContextButton"], 
                            visible=True
                        )
                    with gr.Row():
                        highlight_query = gr.Checkbox(
                            label=labels['highlight_query'],
                            value=False,
                            visible=False
                        )

                with gr.Row(): 
                    out_msj = gr.HTML(
                        label="", 
                        visible=True
                    )

                with gr.Row():
                    gr.Markdown(
                        value=labels["contextList"]
                    )
                with gr.Row():
                    out_context = gr.Dataframe(
                        label="", 
                        interactive=False, 
                        value=pd.DataFrame([], columns=['']),
                        wrap=True,
                        datatype=['str','markdown','str','markdown']
                    )

            with gr.Column():
                gr.Markdown(
                    value=labels["wordDistributionTitle"]
                )
                dist_plot = gr.Plot(
                    label="", 
                    show_label=False
                )
                wc_plot = gr.Plot(
                    label="", 
                    show_label=False, 
                    visible=available_wordcloud
                )

                gr.Markdown(
                    value=labels["frequencyPerSetTitle"]
                )
                subsets_freq = gr.HTML(
                    label=""
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


        btn_get_w_info.click( 
            fn=connector.get_word_info, 
            inputs=[input_word], 
            outputs=[out_msj,
                    out_context,
                    subsets_freq,
                    dist_plot,
                    wc_plot,
                    subsets_choice
            ],
            api_name='word_info'
        )
        
        btn_get_contexts.click(
            fn=connector.get_word_context, 
            inputs=[
                input_word,
                n_context,
                subsets_choice,
                token_id,
                school,
                age,
                gender,
                consent_checkbox,
                highlight_query,
                model_name
            ], 
            outputs=[out_msj, out_context],
            api_name='word_contexts'
        )

        btn_get_logs.click(
            fn=connector.get_logs,
            inputs=[
                token_id,
                gr.Textbox(
                    value=f"logs_edia_datos_{lang}" if available_logs else None,
                    visible=False
                )
            ],
            outputs=[out_msj, df_get_logs],
        )

    return iface