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


# When user selects Argentina
df = pd.read_csv("data/Diccionario de Lunfardo V4.csv", dtype=str)

secrets = dotenv_values("./.env")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
os.environ["OLLAMA_API_KEY"] = secrets["OLLAMA_API_KEY"]
os.environ['COHERE_API_KEY'] = secrets['COHERE_API_KEY']
os.environ['GOOGLE_API_KEY'] = secrets['GOOGLE_API_KEY']

models = {
    "google/gemma-3-12b-it": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="google/gemma-3-12b-it"),
    # "gemma3:4b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="gemma3:4b"),
    # "qwen3:32b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="qwen3:32b"),
    "mistralai/Mistral-7B-Instruct-v0.3": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="mistralai/Mistral-7B-Instruct-v0.3"),
    # "mistral-small3.2:24b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="mistral-small3.2:24b"),
    # "ministral-3:14b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="ministral-3:14b"),
    # "mistral:7b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="mistral:7b"),
    # "openai-gpt-4.1": ChatOpenAI(api_key=secrets["OPENAI_API_KEY"], model="gpt-4.1", temperature=1, max_retries=3),
    # "cohere": ChatCohere(cohere_api_key=secrets["COHERE_API_KEY"], model="command-r", temperature=1, max_retries=3),
    # "google": ChatGoogleGenerativeAI(google_api_key=secrets["GOOGLE_API_KEY"], model="gemini-2.0-flash", temperature=1, max_retries=3),
    "meta-llama/Llama-3.2-3B-Instruct": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="meta-llama/Llama-3.2-3B-Instruct"),
    # "llama3.1:8b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="llama3.1:8b"),
    # "llava:34b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="llava:34b"),
    # "gpt-oss:20b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="gpt-oss:20b"),

}

# --- Interface ---
def interface(DEFAULT_LANG: str) -> gr.Blocks:
    def get_random_data_point():
        data_point = df.sample().iloc[0]
        return {
            "ID": data_point["ID"],
            "phrase": data_point["phrase"],
            "meaning": data_point["meaning"],
        }

    def get_administrative_divisions(selected_countries):
        """Fetches administrative divisions for selected countries."""
        try:
            df = pd.read_json("data/global_administrative_division.json")
            filtered_df = df[df["name"].isin(selected_countries)]
            # Format as "Division Name (Country Name)"
            divisions = [
                f"{division['name']} ({row['name']})"
                for _, row in filtered_df.iterrows()
                for division in row["AD"]
            ]
            return sorted(list(set(divisions))) # Sort and remove duplicates
        except FileNotFoundError:
            print("Error: data/global_administrative_division.json not found.")
            return []
        except Exception as e:
            print(f"Error reading or processing administrative divisions: {e}")
            return []

    def log_result(
        token_id,
        age,
        gender,
        nationality_personal_info,
        personal_region_dropdown,
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
            "nationality_personal_info": nationality_personal_info,
            "personal_region_dropdown": personal_region_dropdown,
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
        nationality_personal_info,
        personal_region_dropdown,
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
            "nationality_personal_info": nationality_personal_info,
            "personal_region_dropdown": personal_region_dropdown,
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
        lang = gr.Radio(
            choices=[
                (i18n("English"), "en"),
                (i18n("Spanish"), "es"),
                (i18n("Portuguese"), "pt"),
            ],
            value=DEFAULT_LANG,
            label=i18n("LanguageLabel"),
        )
        with Translate(
            "language/i18n_typicalPhrases.json",
            lang,
            placeholder_langs=["en", "pt", "es"],
        ):
            with gr.Row(equal_height=False, variant="panel"):
                with gr.Column(scale=1, min_width=200):
                    with gr.Group():
                        token_id = gr.Textbox(
                            label=i18n("Identifier"),
                            lines=1,
                        )
                        age = gr.Number(
                            value=0,
                            label=i18n("AgeLabel"),
                            visible=True,
                        )
                        gender = gr.Radio(
                            ["M", "F", "X"],
                            label=i18n("GenderLabel"),
                            value=None,
                            visible=True,
                        )
                with gr.Column(scale=2):
                    with gr.Group():
                        nationality_personal_info = gr.Dropdown(
                            label=i18n("NationalityLabel"),
                            info=i18n("NationalityInfo"),
                            choices=nationalities,
                            multiselect=True,
                            allow_custom_value=False,
                        )
                        personal_region_dropdown = gr.Dropdown(
                            label=i18n("PersonalRegionLabel"),
                            choices=[], # Initially empty
                            allow_custom_value=True,
                            multiselect=True,
                            interactive=False, # Initially disabled
                        )
                with gr.Column(scale=1, min_width=200):
                    consent_checkbox = gr.Checkbox(
                        label=i18n("ConsentLabel"),
                        value=False,
                    )
                    _ = gr.HTML(
                        value=f"<a href='https://docs.google.com/document/d/17Feum83dTqjcicgJxuWdZ3qLuL3emmVY2idGym_usLU/edit?usp=sharing'>Link 🔗</a>",
                    )
            _ = gr.HTML(
                value="<hr>",
            )
            with gr.Column(visible=True) as personal_data_missing:
                gr.Markdown(
                    i18n("PersonalDataMissingHeading")
                )
            with gr.Row():
                with gr.Column(visible=False, elem_id="annotation_col") as annotation_col:
                    _ = gr.Markdown(
                        "# " + i18n("AnnotationTitle") + "\n\n" +
                        "### " + i18n("AnnotationGoal") + "\n\n" +
                        "### " + i18n("AnnotationInstructions")
                    )
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(i18n("ExampleLabel"), visible=False)
                            data_point_phrase = gr.Markdown(f"{i18n('DataPointPhrasePrefix')} `{initial_data_point['phrase']}`", visible=False)
                            data_point_definition = gr.Markdown(f"{i18n('DataPointMeaningPrefix')} {initial_data_point['meaning']}", visible=False)
                            gr.Markdown(i18n("YourTurnHeading"), visible=False)
                            new_phrase = gr.Textbox(
                                label=i18n("NewPhraseLabel"),
                                placeholder=i18n("NewPhrasePlaceholder"),
                            )
                            new_phrase_definition = gr.Textbox(
                                label=i18n("NewPhraseDefinitionLabel"),
                                placeholder=i18n("NewPhraseDefinitionPlaceholder"),
                            )
                            new_phrase_sentence_example = gr.Textbox(
                                label=i18n("NewPhraseSentenceLabel"),
                                placeholder=i18n("NewPhraseSentencePlaceholder"),
                            )
                            llm_responses_button = gr.Button(i18n("LLMResponsesButton"), interactive=False, variant="primary")

                with gr.Column(visible=False, elem_id="llm_responses_col") as llm_responses_col:
                    gr.Markdown(
                        "### " + i18n("LLMHeader")
                    )
                    with gr.Row(variant="panel"):
                        model_a_response = gr.HighlightedText(
                            label=i18n("ModelSmallLabel"),
                            value=[],
                            combine_adjacent=True,
                            show_legend=False,
                            interactive=False,
                        )
                        model_a_likert = gr.Radio(
                            [1, 2, 3, 4, 5],
                            label=i18n("LikertLabel"),
                            info=i18n("LikertInfo"),
                            value=None,
                            interactive=True,
                            visible=False,
                        )
                    with gr.Row(variant="panel"):
                        model_b_response = gr.HighlightedText(
                            label=i18n("ModelMediumLabel"),
                            value=[],
                            combine_adjacent=True,
                            show_legend=False,
                            interactive=False,
                        )
                        model_b_likert = gr.Radio(
                            [1, 2, 3, 4, 5],
                            label=i18n("LikertLabel"),
                            info=i18n("LikertInfo"),
                            value=None,
                            interactive=True,
                            visible=False,
                        )
                    with gr.Row(variant="panel"):
                        model_c_response = gr.HighlightedText(
                            label=i18n("ModelLargeLabel"),
                            value=[],
                            combine_adjacent=True,
                            show_legend=False,
                            interactive=False,
                        )
                        model_c_likert = gr.Radio(
                            [1, 2, 3, 4, 5],
                            label=i18n("LikertLabel"),
                            info=i18n("LikertInfo"),
                            value=None,
                            interactive=True,
                            visible=False,
                        )
                    skip_button = gr.Button(i18n("NextButton"), visible=False, interactive=False,  variant="primary")

            # Function to update the PERSONAL region dropdown based on selected PERSONAL nationalities
            def update_personal_regions(selected_personal_nationalities):
                if not selected_personal_nationalities:
                    # Disable and clear if no nationalities are selected
                    return gr.update(choices=[], value=[], interactive=False)
                else:
                    # Get divisions using the helper function
                    region_choices = get_administrative_divisions(selected_personal_nationalities)
                    # Enable and update choices, keep existing selection if possible (Gradio handles this)
                    return gr.update(choices=region_choices, interactive=True)

            # Connect the personal nationality dropdown to update the personal region dropdown
            nationality_personal_info.change(
                fn=update_personal_regions,
                inputs=[nationality_personal_info],
                outputs=[personal_region_dropdown]
            )

            def on_llm_responses_button(
                token_id,
                age,
                gender,
                nationality_personal_info,
                personal_region_dropdown,
                consent_checkbox,
                data_point_phrase,
                data_point_definition,
                new_phrase,
                new_phrase_definition,
                new_phrase_sentence_example,
            ):

                # Prompts
                system_prompt = i18n("SystemPromptDefine")
                model_responses = []
                for model_name, model in models.items():

                    if isinstance(model, ModelWrapper):
                        response = model.invoke(
                            system_prompt,
                            new_phrase,
                        )
                        response_content = response['content'].strip()
                    else:

                        response = model.invoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=[
                                {"type": "text", "text": new_phrase},
                            ])
                        ])
                        response_content = response.content.strip()

                    model_responses.append(response_content)

                log_result(
                    token_id,
                    age,
                    gender,
                    nationality_personal_info,
                    personal_region_dropdown,
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
                    nationality_personal_info,
                    personal_region_dropdown,
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
                nationality_personal_info,
                personal_region_dropdown,
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
                    nationality_personal_info,
                    personal_region_dropdown,
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
                    f"{i18n('DataPointPhrasePrefix')} `{new_data_point['phrase']}`",
                    f"{i18n('DataPointMeaningPrefix')} {new_data_point['meaning']}",
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
                    nationality_personal_info,
                    personal_region_dropdown,
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

            # Annotation Toggle

            def toggle_annotation(
                token_id, age, gender, nationality_personal_info, consent_checkbox
            ):
                if any([
                    token_id is None,
                    age is None,
                    gender is None,
                    nationality_personal_info is None or len(nationality_personal_info) == 0,
                    consent_checkbox is None,
                    age < 0,
                    age > 100,
                    len(token_id) == 0,
                    not consent_checkbox
                ]):
                    return (
                        gr.update(),
                        gr.update(),
                        gr.Column(visible=False),
                        gr.Column(visible=False),
                        gr.Column(visible=True)
                    )
                else:
                    new_data_point = get_random_data_point()
                    return (
                        f"{i18n('DataPointPhrasePrefix')} `{new_data_point['phrase']}`",
                        f"{i18n('DataPointMeaningPrefix')} {new_data_point['meaning']}",
                        gr.Column(visible=True),
                        gr.Column(visible=True),
                        gr.Column(visible=False),
                    )

            toggle_annotation_inputs = [
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox
            ]
            toggle_annotation_outputs = [
                data_point_phrase,
                data_point_definition,
                annotation_col,
                llm_responses_col,
                personal_data_missing
            ]

            for component in toggle_annotation_inputs:
                component.change(
                    fn=toggle_annotation,
                    inputs=toggle_annotation_inputs,
                    outputs=toggle_annotation_outputs
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
