import base64
import json
import os
from datetime import datetime
# from langchain_openai import ChatOpenAI
# from langchain_cohere import ChatCohere
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from modules.module_ollama import ModelWrapper
import gradio as gr
from gradio_i18n import Translate, gettext as i18n
import pandas as pd
from data.nationalities import nationalities
from dotenv import dotenv_values
import random
import re
from PIL import Image
import numpy as np
import concurrent.futures

# When user selects Argentina
df = pd.read_csv("data/Diccionario de Lunfardo V4.csv", dtype=str)

secrets = dotenv_values("./.env")
os.environ["OLLAMA_API_KEY"] = secrets["OLLAMA_API_KEY"]

models = {
    "vllm/ministral3-8b": ModelWrapper(
        token=secrets["OLLAMA_API_KEY"], model="vllm/ministral3-8b"
    ),
    "vllm/qwen3.5-4b": ModelWrapper(
        token=secrets["OLLAMA_API_KEY"], model="vllm/qwen3.5-4b"
    ),
    "vllm/gemma4-26b": ModelWrapper(
        token=secrets["OLLAMA_API_KEY"], model="vllm/gemma4-26b"
    ),
    "vllm/gemma3-4b": ModelWrapper(
        token=secrets["OLLAMA_API_KEY"], model="vllm/gemma3-4b"
    ),
}

# --- Interface ---
def interface(
    token_id,
    age,
    gender,
    nationality,
    region,
    school,
    consent_checkbox
) -> gr.Blocks:
    
    def log_result(
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        consent_checkbox,
        new_phrase,
        new_phrase_definition,
        new_phrase_sentence_example,
        system_prompt,
        model_names,
        model_responses
    ):
        result = {
            "timestamp": datetime.now().isoformat(),
            "token_id": token_id,
            "age": age,
            "gender": gender,
            "nationality": nationality,
            "region": region,
            "school": school,
            "consent_checkbox": consent_checkbox,
            "new_phrase": new_phrase,
            "new_phrase_definition": new_phrase_definition,
            "new_phrase_sentence_example": new_phrase_sentence_example,
            "system_prompt": system_prompt,
            "model_names": model_names,
            "model_responses": model_responses
        }
        with open("logs/logs_typicalPhrases_model_definitions.jsonl", "a+", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def log_skip(
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        consent_checkbox,
        new_phrase,
        new_phrase_definition,
        new_phrase_sentence_example,
        model_names,
        model_a_response,
        model_a_likert,
        model_b_response,
        model_b_likert,
        model_c_response,
        model_c_likert
    ):
        result = {
            "timestamp": datetime.now().isoformat(),
            "token_id": token_id,
            "age": age,
            "gender": gender,
            "nationality": nationality,
            "region": region,
            "school": school,
            "consent_checkbox": consent_checkbox,
            "new_phrase": new_phrase,
            "new_phrase_definition": new_phrase_definition,
            "new_phrase_sentence_example": new_phrase_sentence_example,
            "model_names": model_names,
            "model_a_response": model_a_response[0]['token'] if model_a_response else None,
            "model_a_likert": model_a_likert,
            "model_b_response": model_b_response[0]['token'] if model_b_response else None,
            "model_b_likert": model_b_likert,
            "model_c_response": model_c_response[0]['token'] if model_c_response else None,
            "model_c_likert": model_c_likert
        }
        with open("logs/logs_typicalPhrases.jsonl", "a+", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Gradio interface
    with gr.Blocks() as interface:
        _ = gr.Markdown(
            "# " + i18n("TypicalPhrasesAnnotationTitle") + "\n\n" +
            i18n("TypicalPhrasesAnnotationGoal") + "\n\n" +
            i18n("TypicalPhrasesAnnotationInstructions")
        )
        with gr.Row():
            with gr.Column():
                new_phrase = gr.Textbox(
                    label=i18n("TypicalPhrasesNewPhraseLabel"),
                    placeholder=i18n("TypicalPhrasesNewPhrasePlaceholder"),
                )
                new_phrase_definition = gr.Textbox(
                    label=i18n("TypicalPhrasesNewPhraseDefinitionLabel"),
                    placeholder=i18n("TypicalPhrasesNewPhraseDefinitionPlaceholder"),
                )
                new_phrase_sentence_example = gr.Textbox(
                    label=i18n("TypicalPhrasesNewPhraseSentenceLabel"),
                    placeholder=i18n("TypicalPhrasesNewPhraseSentencePlaceholder"),
                )
                llm_responses_button = gr.Button(i18n("TypicalPhrasesLLMResponsesButton"), interactive=False, variant="primary")
            with gr.Column():
                gr.Markdown(
                    "### " + i18n("TypicalPhrasesLLMHeader")
                )
                with gr.Row(variant="panel"):
                    model_a_response = gr.HighlightedText(
                        label=i18n("ModelALabel"),
                        value=[],
                        combine_adjacent=True,
                        show_legend=False,
                        interactive=False,
                    )
                    model_a_likert = gr.Radio(
                        [1, 2, 3, 4, 5],
                        label=i18n("TypicalPhrasesLikertLabel"),
                        info=i18n("TypicalPhrasesLikertInfo"),
                        value=None,
                        interactive=True,
                        visible=False,
                    )
                with gr.Row(variant="panel"):
                    model_b_response = gr.HighlightedText(
                        label=i18n("ModelBLabel"),
                        value=[],
                        combine_adjacent=True,
                        show_legend=False,
                        interactive=False,
                    )
                    model_b_likert = gr.Radio(
                        [1, 2, 3, 4, 5],
                        label=i18n("TypicalPhrasesLikertLabel"),
                        info=i18n("TypicalPhrasesLikertInfo"),
                        value=None,
                        interactive=True,
                        visible=False,
                    )
                with gr.Row(variant="panel"):
                    model_c_response = gr.HighlightedText(
                        label=i18n("ModelCLabel"),
                        value=[],
                        combine_adjacent=True,
                        show_legend=False,
                        interactive=False,
                    )
                    model_c_likert = gr.Radio(
                        [1, 2, 3, 4, 5],
                        label=i18n("TypicalPhrasesLikertLabel"),
                        info=i18n("TypicalPhrasesLikertInfo"),
                        value=None,
                        interactive=True,
                        visible=False,
                    )
                skip_button = gr.Button(i18n("NextButton"), visible=False, interactive=False,  variant="primary")

        def on_llm_responses_button(
            token_id,
            age,
            gender,
            nationality,
            region,
            school,
            consent_checkbox,
            new_phrase,
            new_phrase_definition,
            new_phrase_sentence_example,
        ):

            # Prompts
            system_prompt = i18n("TypicalPhrasesSystemPrompt")
            model_list = list(models.items())
            primary_models = model_list[:3]
            fallback_models = model_list[3:]

            def try_invoke(model):
                name, m = model
                try:
                    if isinstance(m, ModelWrapper):
                        response = m.invoke(
                            system_prompt,
                            new_phrase,
                        )
                        return name, response['content'].strip()
                    else:
                        response = m.invoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=[
                                {"type": "text", "text": new_phrase},
                            ])
                        ])
                        return name, response.content.strip()
                except Exception as e:
                    error_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "tab": "typicalPhrases",
                        "model": name,
                        "error": str(e),
                    }
                    with open("logs/api_errors.jsonl", "a+", encoding="utf-8") as ef:
                        ef.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                    return name, None

            # Run primaries in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                primary_futures = [executor.submit(try_invoke, m) for m in primary_models]
                results = [f.result() for f in primary_futures]
            model_names = [name for name, _ in results]
            model_responses = [content for _, content in results]

            # Fill failed slots with fallbacks (no model used more than once)
            fallback_iter = iter(fallback_models)
            for i, response in enumerate(model_responses):
                if response is None:
                    content = None
                    while content is None:
                        fallback = next(fallback_iter, None)
                        if fallback is None:
                            model_names[i] = None
                            content = i18n("ModelNotWorkingError")
                            break
                        fb_name, content = try_invoke(fallback)
                        if content is not None:
                            model_names[i] = fb_name
                    model_responses[i] = content

            log_result(
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                new_phrase,
                new_phrase_definition,
                new_phrase_sentence_example,
                system_prompt,
                model_names,
                model_responses
            )

            return (
                gr.update(value=[(model_responses[0], None)]),
                gr.update(value=[(model_responses[1], None)]),
                gr.update(value=[(model_responses[2], None)]),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        llm_responses_button.click(
            on_llm_responses_button,
            inputs=[
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                new_phrase,
                new_phrase_definition,
                new_phrase_sentence_example,
            ],
            outputs=[
                model_a_response,
                model_b_response,
                model_c_response,
                model_a_likert,
                model_b_likert,
                model_c_likert,
                llm_responses_button,
                skip_button
            ]
        )

        def on_skip(
            token_id,
            age,
            gender,
            nationality,
            region,
            school,
            consent_checkbox,
            new_phrase,
            new_phrase_definition,
            new_phrase_sentence_example,
            model_a_response,
            model_a_likert,
            model_b_response,
            model_b_likert,
            model_c_response,
            model_c_likert
        ):

            log_skip(
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                new_phrase,
                new_phrase_definition,
                new_phrase_sentence_example,
                list(models.keys()),
                model_a_response,
                model_a_likert,
                model_b_response,
                model_b_likert,
                model_c_response,
                model_c_likert
            )
            return (
                "",
                "",
                "",
                gr.update(value=[]),
                gr.update(value=[]),
                gr.update(value=[]),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=True),
                gr.update(visible=False)
            )


        skip_button.click(
            on_skip,
            inputs=[
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                new_phrase,
                new_phrase_definition,
                new_phrase_sentence_example,
                model_a_response,
                model_a_likert,
                model_b_response,
                model_b_likert,
                model_c_response,
                model_c_likert
            ],
            outputs=[
                new_phrase,
                new_phrase_definition,
                new_phrase_sentence_example,
                model_a_response,
                model_b_response,
                model_c_response,
                model_a_likert,
                model_b_likert,
                model_c_likert,
                llm_responses_button,
                skip_button
            ],
        )

        # LLM Responses Toggle

        def toggle_llm_responses(
            new_phrase,
            new_phrase_definition,
            new_phrase_sentence_example,
        ):
            if all([
                new_phrase,
                new_phrase_definition,
                new_phrase_sentence_example,
            ]):
                return (
                    gr.update(interactive=True)
                )
            else:
                return (
                    gr.update(interactive=False)
                )

        toggle_llm_responses_inputs = [
            new_phrase,
            new_phrase_definition,
            new_phrase_sentence_example,
        ]
        toggle_llm_responses_outputs = [
            llm_responses_button
        ]

        for component in toggle_llm_responses_inputs:
            component.change(
                fn=toggle_llm_responses,
                inputs=toggle_llm_responses_inputs,
                outputs=toggle_llm_responses_outputs
            )

        # Skip Button Toggle

        def toggle_skip_button(
            model_a_likert,
            model_b_likert,
            model_c_likert
        ):
            if all([
                model_a_likert is not None,
                model_b_likert is not None,
                model_c_likert is not None,
            ]):
                return (
                    gr.update(interactive=True)
                )
            else:
                return (
                    gr.update(interactive=False)
                )

        toggle_skip_button_inputs = [
            model_a_likert,
            model_b_likert,
            model_c_likert
        ]
        toggle_skip_button_outputs = [
            skip_button
        ]

        for component in toggle_skip_button_inputs:
            component.change(
                fn=toggle_skip_button,
                inputs=toggle_skip_button_inputs,
                outputs=toggle_skip_button_outputs
            )

    return interface
