import gradio as gr 
import pandas as pd
import matplotlib.pyplot as plt
from modules.module_connection import WordExplorerConnector
from auth import school_list

plt.rcParams.update({'font.size': 14})

def interface(
    embedding, # Class Embedding instance
    available_logs: bool, 
    max_neighbors: int,
    lang: str="es",
) -> gr.Blocks:

    # -- Load examples ---
    if lang == 'es':
        from examples.examples_es import examples_explorar_relaciones_entre_palabras
    elif lang == 'en':
        from examples.examples_en import examples_explorar_relaciones_entre_palabras


    # --- Init vars ---
    connector = WordExplorerConnector(
        embedding=embedding,
        lang=lang,
        logs_file_name=f"logs_edia_we_wordexplorer_{lang}" if available_logs else None
    )

    # --- Load language ---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["WordExplorer_interface"]

    # --- Interface ---
    interface = gr.Blocks()

    with interface:
        with gr.Row():
            with gr.Column(scale=30):
                token_id = gr.Textbox(
                    label="Escriba su correo electrónico",
                    lines=1,
                )
            with gr.Column(scale=70):
                with gr.Row():
                    with gr.Column():
                        school = gr.Number(
                            value=0,
                            label="Seleccione el identificador de su escuela (Ver ➡️)",
                        )
                        school_name = gr.HTML(
                            value=f"<p>No seleccionaste ningún colegio</p>",
                        )
                    with gr.Column():
                        _ = gr.HTML(
                            value="<a href='https://docs.google.com/spreadsheets/d/1SQaQqXh46_J_VrcHo3YJUfPSfKIjbKi73EEtaImzk9c/edit'>Lista de escuelas 🔗</a>",
                        )
        with gr.Row():
            with gr.Column():
                age = gr.Number(
                    value=0,
                    label="Seleccione su edad",
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
                        value="<a href='https://docs.google.com/document/d/17Feum83dTqjcicgJxuWdZ3qLuL3emmVY2idGym_usLU/edit?usp=sharing'>Link 🔗</a>",
                    )
        with gr.Row():
            with gr.Column(scale=3):
                model_name = gr.Radio(
                    ["Modelo en español"],
                    value="Modelo en español",
                    info="Elegí un modelo de lenguaje.",
                    container=False,
                    interactive=True,
                )
                gr.Markdown(
                    value=labels["title"]
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        diagnose_list = gr.Textbox(
                            lines=2, 
                            label=labels["wordList0"],
                            info=labels["wordLists_info"],
                            placeholder=labels["wordList0_placeholder"],
                        )
                    with gr.Column(scale=1,min_width=10):
                        color_wordlist = gr.ColorPicker(
                            label="",
                            value='#000000'
                        )

                with gr.Row():
                    with gr.Column(scale=5): 
                        wordlist_1 = gr.Textbox(
                            lines=2, 
                            label=labels["wordList1"],
                            info=labels["wordLists_info"],
                            placeholder=labels["wordList1_placeholder"],
                        )
                    with gr.Column(scale=1,min_width=10): 
                        color_wordlist_1 = gr.ColorPicker(
                            label="",
                            value='#1f78b4'
                        )
                with gr.Row():
                    with gr.Column(scale=5): 
                        wordlist_2 = gr.Textbox(
                            lines=2, 
                            label=labels["wordList2"],
                            info=labels["wordLists_info"],
                            placeholder=labels["wordList2_placeholder"],
                        )
                    with gr.Column(scale=1,min_width=10):
                        color_wordlist_2 = gr.ColorPicker(
                            label="",
                            value='#33a02c'
                        )
                with gr.Row():
                    with gr.Column(scale=5):    
                        wordlist_3 = gr.Textbox(
                            lines=2, 
                            label=labels["wordList3"],
                            info=labels["wordLists_info"],
                            placeholder=labels["wordList3_placeholder"],
                        )
                    with gr.Column(scale=1,min_width=10): 
                        color_wordlist_3 = gr.ColorPicker(
                            label="",
                            value='#e31a1c',
                        )
                with gr.Row():
                    with gr.Column(scale=5):    
                        wordlist_4 = gr.Textbox(
                            lines=2, 
                            label=labels["wordList4"],
                            info=labels["wordLists_info"],
                            placeholder=labels["wordList4_placeholder"],
                            visible=False,

                        )
                    with gr.Column(scale=1,min_width=10): 
                        color_wordlist_4 = gr.ColorPicker(
                            label="",
                            value='#6a3d9a',
                            visible=False,
                        )
            with gr.Column(scale=4):
                gr.Markdown(
                    value=labels["plotNeighbours"]["title"]
                )
                with gr.Row():
                    n_neighbors = gr.Slider(
                        minimum=0,
                        maximum=max_neighbors,
                        step=1,
                        label=labels["plotNeighbours"]["quantity"]
                    )
                    alpha = gr.Slider(
                        minimum=0.1,
                        maximum=0.9, 
                        value=0.3, 
                        step=0.1,
                        label=labels["options"]["transparency"]
                    )
                    fontsize=gr.Number(
                        value=25, 
                        label=labels["options"]["font-size"]
                    )
                with gr.Group():
                    with gr.Row():
                        btn_plot = gr.Button(
                            value=labels["plot_button"]
                        )
                    with gr.Row():
                        highlight_query = gr.Checkbox(
                            label=labels['highlight_query'],
                            value=False,
                            visible=False
                        )
                with gr.Row(): 
                    err_msg = gr.HTML(
                        label="", 
                        visible=True
                    )
                with gr.Row():
                    word_proyections = gr.Plot(
                        label="", 
                        show_label=False
                    )


        with gr.Row():
            gr.Examples(
                fn=connector.plot_proyection_2d,
                inputs=[diagnose_list,wordlist_1,wordlist_2,wordlist_3],
                outputs=[word_proyections,err_msg],
                examples=examples_explorar_relaciones_entre_palabras,
                label=labels["examples"],
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

        btn_plot.click(
            fn=connector.plot_proyection_2d,
            inputs=[
                diagnose_list,
                wordlist_1,
                wordlist_2,
                wordlist_3,
                wordlist_4,
                color_wordlist,
                color_wordlist_1,
                color_wordlist_2,
                color_wordlist_3,
                color_wordlist_4,
                alpha,
                fontsize,
                n_neighbors,
                token_id,
                school,
                age,
                gender,
                consent_checkbox,
                highlight_query,
                model_name
            ],
            outputs=[word_proyections, err_msg],
            api_name="word_explorer"
        )

        btn_get_logs.click(
            fn=connector.get_logs,
            inputs=[
                token_id,
                gr.Textbox(
                    value=f"logs_edia_we_wordexplorer_{lang}" if available_logs else None,
                    visible=False
                )
            ],
            outputs=[err_msg, df_get_logs],
        )

        return interface