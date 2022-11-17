from modules.module_rankSents import RankSents
from modules.module_logsManager import HuggingFaceDatasetSaver
from modules.module_connection import PhraseBiasExplorerConnector
from examples.examples import examples_sesgos_frases
from tool_info import TOOL_INFO
import gradio as gr


def interface(language_model, available_logs):

    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs
    )

    # --- Init vars ---
    connector = PhraseBiasExplorerConnector(language_model=language_model)

    # --- Interface ---
    iface = gr.Blocks(css=".container {max-width: 90%; margin: auto;}")

    with iface:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("1. Ingrese una frase")
                    sent = gr.Textbox(label="",
                                    placeholder="Utilice * para enmascarar la palabra de interés")
                    
                    gr.Markdown("2. Ingrese palabras de interés (Opcional)")
                    word_list = gr.Textbox( label="", placeholder="La lista de palabras deberán estar separadas por ,")
                    
                    with gr.Group():
                        gr.Markdown("3. Ingrese palabras no deseadas (En caso de no completar punto 2)")    
                        banned_word_list = gr.Textbox( label="", placeholder="La lista de palabras deberán estar separadas por ,")
                        with gr.Row():
                            with gr.Row(): articles = gr.Checkbox(label="Excluir Artículos", value=False)
                            with gr.Row(): prepositions = gr.Checkbox(label="Excluir Preposiciones", value=False)
                            with gr.Row(): conjunctions = gr.Checkbox(label="Excluir Conjunciones", value=False)

                with gr.Row(): btn = gr.Button(value="Obtener")

            with gr.Column():
                with gr.Group():
                    gr.Markdown("Visualización de proporciones")
                    dummy = gr.CheckboxGroup(value="", show_label=False, choices=[])
                    out = gr.HTML(label="")
                    out_msj = gr.Markdown()
        
        with gr.Row():
            examples = gr.Examples(
                fn=connector.rank_sentence_options,
                inputs = [sent, word_list],
                outputs=[out, out_msj],
                examples = examples_sesgos_frases
            )

        with gr.Row(): 
            gr.Markdown(TOOL_INFO)

        btn.click(  
            fn=connector.rank_sentence_options,
            inputs=[sent, word_list, banned_word_list, articles, prepositions, conjunctions], 
            outputs=[out_msj, out, dummy]
        )

        # --- Logs ---
        save_field = [sent, word_list]
        log_callback.setup(components=save_field, flagging_dir="sesgo_en_frases")
        
        btn.click(
            fn=lambda *args: log_callback.flag(
                    flag_data=args,
                    flag_option="sesgo_en_frases",
                    username="vialibre"
            ),
            inputs=save_field,
            outputs=None, 
            preprocess=False
        )
    
    return iface