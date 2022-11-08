from modules.module_word2Context import Word2Context
from modules.module_logsManager import HuggingFaceDatasetSaver
from examples.examples import examples_datos
from tool_info import TOOL_INFO
import gradio as gr
import pandas as pd



def interface(vocabulary, contexts, available_logs):

    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs
    )

    # --- Init Class ---
    w2c = Word2Context(
        context_ds_name=contexts,
        vocabulary=vocabulary
    )

    # --- Interface ---
    iface = gr.Blocks(css=".container { max-width: 90%; margin: auto;}")

    with iface:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("1. Ingrese una palabra de interés")
                    with gr.Row(): input_word = gr.Textbox(label="", show_label=False, placeholder="Ingresar aquí la palabra ...")
                    with gr.Row(): btn_get_w_info = gr.Button("Obtener información de palabra")   

                with gr.Group():
                    gr.Markdown("2. Seleccione cantidad máxima de contextos a recuperar")
                    n_context = gr.Slider(label="", 
                                        step=1, minimum=1, maximum=30, value=5, 
                                        visible=True, interactive=True)
                with gr.Group():
                    gr.Markdown("3. Seleccione conjuntos de interés")
                    subsets_choice = gr.CheckboxGroup(label="", 
                                        interactive=True, 
                                        visible=True)
                    with gr.Row(): btn_get_contexts = gr.Button("Buscar contextos", visible=True)
                
                with gr.Row(): out_msj = gr.Markdown(label="", visible=True)

            with gr.Column():
                with gr.Group():
                    gr.Markdown("Distribución de palabra en vocabulario")
                    dist_plot = gr.Plot(label="", show_label=False)
                    wc_plot = gr.Plot(label="", show_label=False,)

                with gr.Group():
                    gr.Markdown("Frecuencias de aparición por conjunto")
                    # subsets_freq = gr.Label(label="Frecuencias de aparición p/subconjunto:", 
                    #     num_top_classes=16, visible=True, show_label=False)
                    subsets_freq = gr.HTML(label="")
    
        with gr.Row():
            with gr.Group():
                with gr.Row(): gr.Markdown("Lista de contextos")
                with gr.Row(): out_context = gr.Dataframe(label="", 
                                                interactive=False, 
                                                value=pd.DataFrame([], columns=['']),
                                                wrap=True,
                                                datatype=['str','markdown','str','markdown'])

        with gr.Group():
            gr.Markdown(TOOL_INFO)

        btn_get_w_info.click( 
            fn=w2c.getWordInfo, 
            inputs=[input_word], 
            outputs=[out_msj,
                    out_context,
                    subsets_freq,
                    dist_plot,
                    wc_plot,
                    subsets_choice]
        )
        
        btn_get_contexts.click(     
            fn=w2c.getWordContext, 
            inputs=[input_word, n_context, subsets_choice], 
            outputs=[out_msj, out_context]
        )
        
        # --- Logs ---
        save_field = [input_word, subsets_choice]
        log_callback.setup(components=save_field, flagging_dir="datos")
        
        btn_get_contexts.click(
            fn=lambda *args: log_callback.flag(
                    flag_data=args,
                    flag_option="datos",
                    username="vialibre"
            ),
            inputs=save_field,
            outputs=None, 
            preprocess=False
        )
    
    return iface