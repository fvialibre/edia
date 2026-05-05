import base64
import json
import os
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from modules.module_ollama import ModelWrapper
import gradio as gr
from gradio_i18n import Translate, gettext as i18n
from gradio_modal import Modal
import pandas as pd
from data.nationalities import nationalities
from dotenv import dotenv_values
import random
import re
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

df = pd.read_csv(
    "/home/givetta/edia/data/dollar_street/dataset_dollarstreet/images_v2_imagenet_train.csv"
)
df["income_quartile"] = pd.qcut(df["income"], q=4, labels=[1, 2, 3, 4]).astype(int)
df["topics"] = df["topics"].apply(eval)
df = df.explode("topics")
df["imageRelPath"] = (
    "/home/givetta/edia/data/dollar_street/dataset_dollarstreet/" + df["imageRelPath"]
)

secrets = dotenv_values("./.env")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
os.environ["OLLAMA_API_KEY"] = secrets["OLLAMA_API_KEY"]
os.environ["COHERE_API_KEY"] = secrets["COHERE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]

models = {
    # "gemma3:27b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="ollama/gemma3:27b"),
    "phi4-multi": ModelWrapper(
        token=secrets["OLLAMA_API_KEY"], model="vllm/phi4-multi"
    ),
    "gemma3-4b": ModelWrapper(
        token=secrets["OLLAMA_API_KEY"], model="vllm/gemma3-4b"
    ),
    "qwen3-vl-2b": ModelWrapper(
        token=secrets["OLLAMA_API_KEY"], model="vllm/qwen3-vl-2b"
    ),
}

# --- Interface ---
def interface(
    token_id, age, gender, nationality, region, school, consent_checkbox
) -> gr.Blocks:
    def get_random_data_points():
        """Sample two random images from the dollar street dataset (same topic preferred, different enough income quartiles)."""
        valid_quartile_pairs = [(1, 3), (1, 4), (2, 4)]
        q_left, q_right = random.choice(valid_quartile_pairs)
        topic = df["topics"].sample(1).iloc[0]
        topic_df = df[df["topics"] == topic]
        left_pool = topic_df[topic_df["income_quartile"] == q_left]
        right_pool = topic_df[topic_df["income_quartile"] == q_right]
        if left_pool.empty:
            left_pool = df[df["income_quartile"] == q_left]
        if right_pool.empty:
            right_pool = df[df["income_quartile"] == q_right]
        left = left_pool.sample(1).iloc[0]
        right = right_pool.sample(1).iloc[0]
        return {
            "left": {
                "id": left["id"],
                "image_path": left["imageRelPath"],
            },
            "right": {
                "id": right["id"],
                "image_path": right["imageRelPath"],
            },
        }

    def log_result(
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        consent_checkbox,
        current_data_points,
        question_input,
        biased_answer_input,
        bias_type_input,
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
            "current_data_points": current_data_points,
            "question_input": question_input,
            "biased_answer_input": biased_answer_input,
            "bias_type_input": bias_type_input,
        }
        with open("logs/logs_ambiguous_references.jsonl", "a+", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def log_validation(
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        consent_checkbox,
        validated_submission,
        validation_answer,
        validation_quality,
        validation_labels,
    ):
        result = {
            "timestamp": datetime.now().isoformat(),
            "validator_token_id": token_id,
            "validator_age": age,
            "validator_gender": gender,
            "validator_nationality": nationality,
            "validator_region": region,
            "validator_school": school,
            "validator_consent_checkbox": consent_checkbox,
            "validated_submission": validated_submission,
            "validation_answer": validation_answer,
            "validation_quality": validation_quality,
            "validation_labels": validation_labels,
        }
        with open(
            "logs/logs_ambiguous_references_validations.jsonl", "a+", encoding="utf-8"
        ) as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    initial_data_points = get_random_data_points()

    # Gradio interface
    with gr.Blocks() as interface:
        current_data_points = gr.State(value=initial_data_points)
        validated_submission = gr.State(value=None)

        _ = gr.Markdown(
            "# " + i18n("AmbiguousReferencesAnnotationTitle") + "\n\n" +
            i18n("AmbiguousReferencesAnnotationInstructions") + "\n\n" +
            i18n("AmbiguousReferencesAnnotationInstructions2")
        )
        with gr.Row():
            with gr.Column(scale=1):
                data_point_image_left = gr.Image(
                    label=i18n("AmbiguousReferencesImageLeftLabel"),
                    value=initial_data_points["left"]["image_path"],
                    interactive=False,
                )
            with gr.Column(scale=1):
                data_point_image_right = gr.Image(
                    label=i18n("AmbiguousReferencesImageRightLabel"),
                    value=initial_data_points["right"]["image_path"],
                    interactive=False,
                )
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label=i18n("AmbiguousReferencesQuestionLabel"),
                )
                biased_answer_input = gr.Radio(
                    label=i18n("AmbiguousReferencesBiasedAnswerLabel"),
                    choices=[i18n("AmbiguousReferencesLeft"), i18n("AmbiguousReferencesRight")],
                    interactive=True,
                )
                bias_type_input = gr.Dropdown(
                    choices=[
                        i18n("AmbiguousReferencesValidationBiasPhysicalAppearance"),
                        i18n("AmbiguousReferencesValidationBiasDisability"),
                        i18n("AmbiguousReferencesValidationBiasAge"),
                        i18n("AmbiguousReferencesValidationBiasEthnicity"),
                        i18n("AmbiguousReferencesValidationBiasGender"),
                        i18n("AmbiguousReferencesValidationBiasNationality"),
                        i18n("AmbiguousReferencesValidationBiasSexualOrientation"),
                        i18n("AmbiguousReferencesValidationBiasProfession"),
                        i18n("AmbiguousReferencesValidationBiasReligion"),
                        i18n("AmbiguousReferencesValidationBiasSocioeconomicStatus"),
                    ],
                    label=i18n("AmbiguousReferencesValidationBiasLabel"),
                    info=i18n("AmbiguousReferencesValidationBiasInfo"),
                    multiselect=True,
                    allow_custom_value=True,
                )
                with gr.Row():
                    llm_responses_button = gr.Button(
                        i18n("AmbiguousReferencesLLMResponsesButton"),
                        interactive=False,
                        variant="primary",
                        scale=75,
                    )
                    skip_images_button = gr.Button(
                        i18n("SkipButton"),
                        variant="secondary",
                        scale=25,
                    )

        gr.Markdown(
            i18n("AmbiguousReferencesLLMHeader")
        )
        with gr.Row():
            with gr.Row():
                model_a_response = gr.HighlightedText(
                    label=i18n("ModelALabel"),
                    value=[],
                    combine_adjacent=True,
                    show_legend=False,
                    interactive=False,
                    color_map={"✓": "green", "X": "red"},
                )
                model_b_response = gr.HighlightedText(
                    label=i18n("ModelBLabel"),
                    value=[],
                    combine_adjacent=True,
                    show_legend=False,
                    interactive=False,
                    color_map={"✓": "green", "X": "red"},
                )
                model_c_response = gr.HighlightedText(
                    label=i18n("ModelCLabel"),
                    value=[],
                    combine_adjacent=True,
                    show_legend=False,
                    interactive=False,
                    color_map={"✓": "green", "X": "red"},
                )

        with gr.Row():
            next_button = gr.Button(i18n("NextButton"), variant="primary", visible=False)

        ### MODAL
        with Modal(visible=False) as validation_modal:
            _ = gr.Markdown("# " + i18n("AmbiguousReferencesModalHeader"))
            with gr.Row():
                with gr.Column():
                    modal_data_point_image_left = gr.Image(
                        label=i18n("AmbiguousReferencesImageLeftLabel"),
                        value=initial_data_points["left"]["image_path"],
                        interactive=False,
                    )
                with gr.Column():
                    modal_data_point_image_right = gr.Image(
                        label=i18n("AmbiguousReferencesImageRightLabel"),
                        value=initial_data_points["right"]["image_path"],
                        interactive=False,
                    )
                with gr.Column():
                    validation_1_multiple_choice = gr.Radio(
                        label=i18n("AmbiguousReferencesValidationQuestionLabel"),
                        choices=[i18n("AmbiguousReferencesLeft"), i18n("AmbiguousReferencesRight")],
                        value=None,
                        interactive=True,
                    )
                    validation_1_q1 = gr.Radio(
                        choices=[1, 2, 3, 4, 5],
                        value=None,
                        label=i18n("AmbiguousReferencesValidationQualityLabel"),
                        info=i18n("AmbiguousReferencesValidationQualityInfo"),
                        interactive=True,
                    )
                    validation_1_label = gr.Dropdown(
                        choices=[
                            i18n("AmbiguousReferencesValidationBiasPhysicalAppearance"),
                            i18n("AmbiguousReferencesValidationBiasDisability"),
                            i18n("AmbiguousReferencesValidationBiasAge"),
                            i18n("AmbiguousReferencesValidationBiasEthnicity"),
                            i18n("AmbiguousReferencesValidationBiasGender"),
                            i18n("AmbiguousReferencesValidationBiasNationality"),
                            i18n("AmbiguousReferencesValidationBiasSexualOrientation"),
                            i18n("AmbiguousReferencesValidationBiasProfession"),
                            i18n("AmbiguousReferencesValidationBiasReligion"),
                            i18n("AmbiguousReferencesValidationBiasSocioeconomicStatus"),
                        ],
                        label=i18n("AmbiguousReferencesValidationBiasLabel"),
                        info=i18n("AmbiguousReferencesValidationBiasInfo"),
                        multiselect=True,
                        allow_custom_value=True,
                    )
            with gr.Row():
                modal_skip_button = gr.Button(i18n("SkipButton"), variant="secondary", scale=75)
                modal_next_button = gr.Button(
                    i18n("NextButton"), variant="primary", scale=25, interactive=False
                )

        # --- Event handlers ---

        def image_to_base64(image_array):
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
            image_path = f"/tmp/edia_{timestamp_str}.png"
            img = Image.fromarray(image_array)
            img.thumbnail((800, 800), Image.LANCZOS)
            img.save(image_path, quality=75, optimize=True)
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        def on_llm_responses_button(
            token_id,
            age,
            gender,
            nationality,
            region,
            school,
            consent_checkbox,
            current_data_points,
            data_point_image_left,
            data_point_image_right,
            question_input,
            biased_answer_input,
            bias_type_input,
        ):                
            system_prompt = (
                f"{i18n('AmbiguousReferencesSystemPrompt')}\n"
                f"{i18n('AmbiguousReferencesBiasTypesLabel')}: {', '.join(bias_type_input)}"
            )
            multiple_choice_prompt = (
                f"{question_input}\n" + i18n("AmbiguousReferencesMultipleChoicePromptSuffix")
            )

            base64_left = image_to_base64(data_point_image_left)
            base64_right = image_to_base64(data_point_image_right)

            def call_model(model_name, model):
                if isinstance(model, ModelWrapper):
                    response = model.invoke(
                        system_prompt,
                        multiple_choice_prompt,
                        base64_image=base64_left,
                        base64_image_2=base64_right,
                    )
                    response_content = response["content"].strip()
                else:
                    response = model.invoke(
                        [
                            SystemMessage(content=system_prompt),
                            HumanMessage(
                                content=[
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_left}"
                                        },
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_right}"
                                        },
                                    },
                                    {"type": "text", "text": multiple_choice_prompt},
                                ]
                            ),
                        ]
                    )
                    response_content = response.content.strip()
                return response_content

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(call_model, name, model): name
                    for name, model in models.items()
                }
                model_responses = [future.result() for future in futures]

            def highlight_response(response):
                lines = response.strip().split("\n", 1)
                answer_line = lines[0].strip()
                rest = lines[1].strip() if len(lines) > 1 else ""
                label = "✓" if i18n(biased_answer_input).strip().lower() in answer_line.lower() else "X"
                result = [(answer_line, label)]
                if rest:
                    result.append((f"\n{rest}", None))
                return result

            log_result(
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                current_data_points,
                question_input,
                biased_answer_input,
                bias_type_input,
            )

            return (
                gr.update(value=highlight_response(model_responses[0])),
                gr.update(value=highlight_response(model_responses[1])),
                gr.update(value=highlight_response(model_responses[2])),
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
                current_data_points,
                data_point_image_left,
                data_point_image_right,
                question_input,
                biased_answer_input,
                bias_type_input,
            ],
            outputs=[model_a_response, model_b_response, model_c_response, next_button],
        )

        def on_next_button(token_id):
            # Load a random submission from logs for validation
            df_logs = pd.read_json(
                "logs/logs_ambiguous_references.jsonl", lines=True, convert_dates=False
            )
            other_submissions = df_logs[df_logs["token_id"] != token_id]

            # Exclude submissions that the validator has already validated
            df_validations = pd.read_json(
                "logs/logs_ambiguous_references_validations.jsonl",
                lines=True,
                convert_dates=False,
            )
            my_validations = df_validations[
                df_validations["validator_token_id"] == token_id
            ]["validated_submission"]
            already_validated = my_validations.apply(lambda x: x["timestamp"]).tolist()
            other_submissions = other_submissions[
                ~other_submissions["timestamp"].isin(already_validated)
            ]

            # If there are no other submissions, sample a random one from the logs (including the validator's own submissions)
            if other_submissions.empty:
                other_submission = df_logs.sample(1).iloc[0]
            else:
                other_submission = other_submissions.sample(1).iloc[0]

            # Update modal with other user's submission
            return (
                gr.update(visible=True),
                gr.update(
                    value=other_submission["current_data_points"]["left"]["image_path"]
                ),
                gr.update(
                    value=other_submission["current_data_points"]["right"]["image_path"]
                ),
                gr.Radio(
                    label=other_submission["question_input"],
                    choices=[i18n("AmbiguousReferencesLeft"), i18n("AmbiguousReferencesRight")],
                    value=None,
                    interactive=True,
                ),
                gr.update(value=other_submission["biased_answer_input"]),
                other_submission.to_dict(),
            )

        next_button.click(
            on_next_button,
            inputs=[
                token_id,
            ],
            outputs=[
                validation_modal,
                modal_data_point_image_left,
                modal_data_point_image_right,
                validation_1_multiple_choice,
                validation_1_q1,
                validated_submission,
            ],
        )

        def load_new_data_points():
            new_data_points = get_random_data_points()
            return (
                new_data_points,
                new_data_points["left"]["image_path"],
                new_data_points["right"]["image_path"],
                "",
                gr.Radio(
                    label=i18n("AmbiguousReferencesBiasedAnswerLabel"),
                    choices=[i18n("AmbiguousReferencesLeft"), i18n("AmbiguousReferencesRight")],
                    interactive=True,
                    value=None,
                ),
                gr.update(value=[]),
                gr.update(visible=False),
                gr.update(value=[]),
                gr.update(value=[]),
                gr.update(value=[]),
                Modal(visible=False),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=[]),
                gr.update(interactive=False),
            )

        def on_modal_next_button(
            token_id,
            age,
            gender,
            nationality,
            region,
            school,
            consent_checkbox,
            validated_submission,
            validation_answer,
            validation_quality,
            validation_labels,
        ):
            log_validation(
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                validated_submission,
                validation_answer,
                validation_quality,
                validation_labels,
            )
            return load_new_data_points()

        def on_modal_skip_button(*args):
            return load_new_data_points()

        modal_next_and_skip_outputs = [
            current_data_points,
            data_point_image_left,
            data_point_image_right,
            question_input,
            biased_answer_input,
            bias_type_input,
            next_button,
            model_a_response,
            model_b_response,
            model_c_response,
            validation_modal,
            modal_data_point_image_left,
            modal_data_point_image_right,
            validation_1_multiple_choice,
            validation_1_q1,
            validation_1_label,
            modal_next_button,
        ]

        modal_next_button.click(
            on_modal_next_button,
            inputs=[
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                validated_submission,
                validation_1_multiple_choice,
                validation_1_q1,
                validation_1_label,
            ],
            outputs=modal_next_and_skip_outputs,
        )

        modal_skip_button.click(
            on_modal_skip_button,
            inputs=[
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                question_input,
                biased_answer_input,
                bias_type_input,
            ],
            outputs=modal_next_and_skip_outputs,
        )

        def on_skip_images_button():
            return load_new_data_points()

        skip_images_button.click(
            on_skip_images_button,
            inputs=[],
            outputs=modal_next_and_skip_outputs,
        )

        def toggle_modal_next_button(
            validation_answer, validation_quality, validation_labels
        ):
            if all([validation_answer, validation_quality, validation_labels]):
                return gr.update(interactive=True)
            else:
                return gr.update(interactive=False)

        for component in [
            validation_1_multiple_choice,
            validation_1_q1,
            validation_1_label,
        ]:
            component.change(
                fn=toggle_modal_next_button,
                inputs=[
                    validation_1_multiple_choice,
                    validation_1_q1,
                    validation_1_label,
                ],
                outputs=[modal_next_button],
            )

        # --- LLM Responses Toggle ---

        def toggle_llm_responses(question_input, biased_answer_input, bias_type_input):
            if all([question_input, biased_answer_input, bias_type_input]):
                return gr.update(
                    interactive=True,
                )
            else:
                return gr.update(
                    interactive=False,
                )

        for component in [question_input, biased_answer_input, bias_type_input]:
            component.change(
                fn=toggle_llm_responses,
                inputs=[question_input, biased_answer_input, bias_type_input],
                outputs=[llm_responses_button],
            )

    return interface
