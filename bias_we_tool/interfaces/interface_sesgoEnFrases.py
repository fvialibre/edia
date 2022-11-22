from modules.module_rankSents import RankSents
from modules.module_logsManager import HuggingFaceDatasetSaver
from modules.module_connection import PhraseBiasExplorerConnector
from examples.examples import examples_sesgos_frases
from tool_info import TOOL_INFO
import gradio as gr
import pandas as pd


def interface(language_model, available_logs, lang="spanish"):

    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs
    )

    # --- Init vars ---
    connector = PhraseBiasExplorerConnector(language_model=language_model)
    labels = pd.read_json(f"language/{lang}.json")["PhraseExplorer_interface"]

    # --- Interface ---
    iface = gr.Blocks(css=".container {max-width: 90%; margin: auto;}")

    with iface:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown(labels["step1"])
                    sent = gr.Textbox(label=labels["sent"]["title"],
                                    placeholder=labels["sent"]["placeholder"])
                    
                    gr.Markdown(labels["step2"])
                    word_list = gr.Textbox( label=labels["wordList"]["title"], 
                                            placeholder=labels["wordList"]["placeholder"])
                    
                    with gr.Group():
                        gr.Markdown(labels["step3"])
                        banned_word_list = gr.Textbox( label=labels["bannedWordList"]["title"], 
                                                        placeholder=labels["bannedWordList"]["placeholder"])
                        with gr.Row():
                            with gr.Row(): articles     = gr.Checkbox(label=labels["excludeArticles"], value=False)
                            with gr.Row(): prepositions = gr.Checkbox(label=labels["excludePrepositions"], value=False)
                            with gr.Row(): conjunctions = gr.Checkbox(label=labels["excludeConjunctions"], value=False)

                with gr.Row(): btn = gr.Button(value=labels["resultsButton"])

            with gr.Column():
                with gr.Group():
                    gr.Markdown(labels["plot"])
                    dummy = gr.CheckboxGroup(value="", show_label=False, choices=[])
                    out = gr.HTML(label="")
                    out_msj = gr.Markdown()
        
        with gr.Row():
            examples = gr.Examples(
                fn=connector.rank_sentence_options,
                inputs = [sent, word_list],
                outputs=[out, out_msj],
                examples = examples_sesgos_frases,
                label=labels["examples"]
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