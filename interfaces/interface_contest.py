import gradio as gr
import pandas as pd
from auth import school_list


def interface(
    available_logs: bool, 
    lang: str="es",
) -> gr.Blocks:

    # -- Load examples --
    if lang == 'es':
        from examples.examples_es import examples_sesgos_frases
    elif lang == 'en':
        from examples.examples_en import examples_sesgos_frases

    # --- Get language labels---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["PhraseExplorer_interface"]

    # --- Init Interface ---
    iface = gr.Blocks(
        css=".container {max-width: 90%; margin: auto;}"
    )

    # def load_contest_data(token_id):

    # --- Load CSV into pandas DataFrame ---
    df = pd.read_csv("/home/givetta/edia/logs/logs_edia_lmodels_biasphrase_es_async3.csv")
    token_id_counts = df['token_id'].value_counts().reset_index()
    token_id_counts.columns = ['token_id', 'count']
    print(token_id_counts)
    top_10 = token_id_counts.head(10)
    print(top_10)

    
    with iface:
        with gr.Row():
            with gr.Column():
                token_id = gr.Textbox(
                    label="Escriba su correo electrónico",
                    lines=1,
                )

        with gr.Row():
            btn_get_contest_data = gr.Button(
                value=labels["see_queries_made"]
            )

        temp_by_time = gr.BarPlot(
            top_10,
            x="token_id",
            y="count",
        )



        # btn_get_contest_data.click(
        #     fn=load_contest_data,
        #     inputs=[
        #         token_id,
        #     ], 
        #     outputs=[out_msj, out],
        #     api_name="contest_data"
        # )

    return iface