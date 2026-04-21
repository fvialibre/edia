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
    "vllm/gemma3-4b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="vllm/gemma3-4b"),
    "vllm/qwen3-vl-2b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="vllm/qwen3-vl-2b"),
    "vllm/mistral-7b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="vllm/mistral-7b"),
}

# --- Interface ---
def interface(
    token_id,
    age,
    gender,
    nationality,
    region,
    consent_checkbox
) -> gr.Blocks:
    def get_random_data_point():
        data_point = df.sample().iloc[0]
        return {
            "ID": data_point["ID"],
            "phrase": data_point["phrase"],
            "meaning": data_point["meaning"],
        }

    def log_result(
        token_id,
        age,
        gender,
        nationality,
        region,
        consent_checkbox,
        data_point_phrase,
        data_point_definition,
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
            "consent_checkbox": consent_checkbox,
            "data_point_phrase": data_point_phrase,
            "data_point_definition": data_point_definition,
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
        consent_checkbox,
        data_point_phrase,
        data_point_definition,
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
            "consent_checkbox": consent_checkbox,
            # "data_point_phrase": data_point_phrase,
            # "data_point_definition": data_point_definition,
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

    initial_data_point = get_random_data_point()

    # Gradio interface
    with gr.Blocks() as interface:
        _ = gr.Markdown(
            "# " + i18n("TypicalPhrasesAnnotationTitle") + "\n\n" +
            "### " + i18n("TypicalPhrasesAnnotationGoal") + "\n\n" +
            "### " + i18n("TypicalPhrasesAnnotationInstructions")
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(i18n("ExampleLabel"), visible=False)
                data_point_phrase = gr.Markdown(f"{i18n('TypicalPhrasesDataPointPhrasePrefix')} `{initial_data_point['phrase']}`", visible=False)
                data_point_definition = gr.Markdown(f"{i18n('TypicalPhrasesDataPointMeaningPrefix')} {initial_data_point['meaning']}", visible=False)
                gr.Markdown(i18n("YourTurnHeading"), visible=False)
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
                llm_responses_button = gr.Button(i18n("LLMResponsesButton"), interactive=False, variant="primary")
            with gr.Column():
                gr.Markdown(
                    "### " + i18n("LLMHeader")
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
            consent_checkbox,
            data_point_phrase,
            data_point_definition,
            new_phrase,
            new_phrase_definition,
            new_phrase_sentence_example,
        ):

            # Prompts
            system_prompt = i18n("TypicalPhrasesSystemPrompt")
            model_responses = []
            
            def invoke_model(model_name, model):
                if isinstance(model, ModelWrapper):
                    response = model.invoke(
                        system_prompt,
                        new_phrase,
                    )
                    content = response['content'].strip()
                else:
                    response = model.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=[
                            {"type": "text", "text": new_phrase},
                        ])
                    ])
                    content = response.content.strip()
                
                # Remove everything between ◣
                content = re.sub(r'◣.*?ground', '', content, flags=re.DOTALL).strip()
                return content
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(invoke_model, name, model): name for name, model in models.items()}
                for future in concurrent.futures.as_completed(futures):
                    model_responses.append(future.result())

            log_result(
                token_id,
                age,
                gender,
                nationality,
                region,
                consent_checkbox,
                data_point_phrase,
                data_point_definition,
                new_phrase,
                new_phrase_definition,
                new_phrase_sentence_example,
                system_prompt,
                list(models.keys()),
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
                consent_checkbox,
                data_point_phrase,
                data_point_definition,
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
            consent_checkbox,
            data_point_phrase,
            data_point_definition,
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
                consent_checkbox,
                data_point_phrase,
                data_point_definition,
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
            new_data_point = get_random_data_point()
            return (
                f"{i18n('TypicalPhrasesDataPointPhrasePrefix')} `{new_data_point['phrase']}`",
                f"{i18n('TypicalPhrasesDataPointMeaningPrefix')} {new_data_point['meaning']}",
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
                consent_checkbox,
                data_point_phrase,
                data_point_definition,
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
                data_point_phrase,
                data_point_definition,
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
