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
        with gr.Row():
            with gr.Column():
                model_name = gr.Radio(
                    ["Modelo en español"],
                    value="Modelo en español",
                    info="Elegí un modelo de lenguaje.",
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