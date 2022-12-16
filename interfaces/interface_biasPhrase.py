import gradio as gr
import pandas as pd
from tool_info import TOOL_INFO
from modules.module_logsManager import HuggingFaceDatasetSaver
from modules.module_connection import PhraseBiasExplorerConnector



def interface(
    language_model: str, 
    available_logs: bool, 
    lang: str="es"
) -> gr.Blocks:

    # -- Load examples --
    if lang == 'es':
        from examples.examples_es import examples_sesgos_frases
    elif lang == 'en':
        from examples.examples_en import examples_sesgos_frases

    # --- Init logs ---
    log_callback = HuggingFaceDatasetSaver(
        available_logs=available_logs,
        dataset_name=f"logs_edia_lmodels_{lang}"
    )

    # --- Init vars ---
    connector = PhraseBiasExplorerConnector(
        language_model=language_model,
        lang=lang
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
                with gr.Group():
                    gr.Markdown(
                        value=labels["step1"]
                    )
                    sent = gr.Textbox(
                        label=labels["sent"]["title"],
                        placeholder=labels["sent"]["placeholder"],
                        show_label=False
                    )
                    
                    gr.Markdown(
                        value=labels["step2"]
                    )
                    word_list = gr.Textbox( 
                        label=labels["wordList"]["title"], 
                        placeholder=labels["wordList"]["placeholder"],
                        show_label=False
                    )
                    
                    with gr.Group():
                        gr.Markdown(
                            value=labels["step3"]
                        )
                        banned_word_list = gr.Textbox( 
                            label=labels["bannedWordList"]["title"], 
                            placeholder=labels["bannedWordList"]["placeholder"]
                        )
                        with gr.Row():
                            with gr.Row(): 
                                articles = gr.Checkbox(
                                    label=labels["excludeArticles"], 
                                    value=False
                                )
                            with gr.Row(): 
                                prepositions = gr.Checkbox(
                                    label=labels["excludePrepositions"], 
                                    value=False
                                )
                            with gr.Row(): 
                                conjunctions = gr.Checkbox(
                                    label=labels["excludeConjunctions"], 
                                    value=False
                                )

                with gr.Row(): 
                    btn = gr.Button(
                        value=labels["resultsButton"]
                    )

            with gr.Column():
                with gr.Group():
                    gr.Markdown(
                        value=labels["plot"]
                    )
                    dummy = gr.CheckboxGroup(
                        value="", 
                        show_label=False, 
                        choices=[]
                    )
                    out = gr.HTML(
                        label=""
                    )
                    out_msj = gr.Markdown(
                        value=""
                    )
        
        with gr.Row():
            examples = gr.Examples(
                fn=connector.rank_sentence_options,
                inputs=[sent, word_list],
                outputs=[out, out_msj],
                examples=examples_sesgos_frases,
                label=labels["examples"]
            )

        with gr.Row(): 
            gr.Markdown(
                value=TOOL_INFO
            )

        btn.click(  
            fn=connector.rank_sentence_options,
            inputs=[sent, word_list, banned_word_list, articles, prepositions, conjunctions], 
            outputs=[out_msj, out, dummy]
        )

        # --- Logs ---
        save_field = [sent, word_list]
        log_callback.setup(
            components=save_field, 
            flagging_dir="logs_phrase_bias"
        )
        
        btn.click(
            fn=lambda *args: log_callback.flag(
                flag_data=args,
                flag_option="phrase_bias",
                username="vialibre"
            ),
            inputs=save_field,
            outputs=None, 
            preprocess=False
        )
    
    return iface