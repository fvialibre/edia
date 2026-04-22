import base64
import json
import os
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from modules.module_ollama import ModelWrapper
import gradio as gr
from gradio_modal import Modal
import pandas as pd
from data.nationalities import nationalities
from datasets import load_dataset
from datasets import load_from_disk
from dotenv import dotenv_values
import random
import re
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# LOAD CVQA #################################################

ds = load_from_disk("data/cvqa_full_dataset")

print("Building CVQA index...")

# 1. Fetch metadata
df = ds.select_columns(["Subset"]).to_pandas()

# 2. Extract 'Country' from the "('Language', 'Country')" string
# We split by comma, take the last part, and clean up quotes/brackets.
# Example: "('Bulgarian', 'Bulgaria')" -> "Bulgaria"
def extract_country(subset_str):
    if not subset_str: return "Unknown"
    # Split by comma to separate Language and Country
    parts = subset_str.split(',')
    # Grab the last part (Country) and strip: spaces, single quotes, closing paren
    return parts[-1].strip(" ')")

# Apply this function to create a temporary column
df['Country_Key'] = df['Subset'].apply(extract_country)

# 3. Group by this new clean Country key
# Now "Bulgaria" will point to ALL indices, regardless of the language 
country_indices_map = df.groupby("Country_Key").indices

print("CVQA Index built!")

# END LOAD CVQA #############################################


secrets = dotenv_values("./.env")
os.environ["OLLAMA_API_KEY"] = secrets["OLLAMA_API_KEY"]

models = {
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
    token_id,
    age,
    gender,
    nationality,
    region,
    school,
    consent_checkbox
) -> gr.Blocks:

    def get_random_data_point(token_id, country_of_interest):
        possible_indices = country_indices_map[country_of_interest]
        
        cvqa_logs_df = pd.read_json("logs/logs_cvqa.jsonl", lines=True)
        cvqa_validation_logs_df = pd.read_json("logs/logs_validation_cvqa.jsonl", lines=True)

        # remove from possible_indices those that have been logged by this token_id
        logged_indices_by_token = cvqa_logs_df[cvqa_logs_df["token_id"] == token_id]["data_point_IDX"].unique().tolist()
        possible_indices = [idx for idx in possible_indices if idx not in logged_indices_by_token]
        
        # Count logs per data point for possible_indices
        logs_count = cvqa_logs_df.groupby("data_point_IDX").size()
        logs_count = logs_count[logs_count.index.isin(possible_indices)]
        
        # Filter out data points that already have 5+ validations
        if not cvqa_validation_logs_df.empty:
            cvqa_validation_counts = cvqa_validation_logs_df.groupby("data_point_IDX").size()
            data_points_with_lots_of_validations = cvqa_validation_counts[cvqa_validation_counts > 5].index.tolist()
        else:
            data_points_with_lots_of_validations = []
        
        # Filter out data points with lots of validations
        filtered_indices = [idx for idx in logs_count.index if idx not in data_points_with_lots_of_validations]
        
        # Sort by log count descending and pick the first one
        if len(filtered_indices) > 0:
            random_index = logs_count[filtered_indices].idxmax()
        else:
            random_index = random.choice(possible_indices)
        
        data_point = ds[int(random_index)]
        return {
            "ID": data_point["ID"],
            "IDX": int(random_index),
            "image": data_point["image"],
            "Question": data_point["Question"],
            "Options": data_point["Options"],
        }

    def log_result(
        source,
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        consent_checkbox,
        data_point_ID,
        data_point_IDX,
        data_point_multiple_choice,
        other_question_input,
        correct_answer_input,
        incorrect_answer_1_input,
        incorrect_answer_2_input,
        incorrect_answer_3_input,
    ):
        result = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "token_id": token_id,
            "age": age,
            "gender": gender,
            "nationality": nationality,
            "region": region,
            "school": school,
            "consent_checkbox": consent_checkbox,
            "data_point_ID": data_point_ID,
            "data_point_IDX": data_point_IDX,
            "data_point_multiple_choice": data_point_multiple_choice,
            "other_question_input": other_question_input,
            "correct_answer_input": correct_answer_input,
            "incorrect_answer_1_input": incorrect_answer_1_input,
            "incorrect_answer_2_input": incorrect_answer_2_input,
            "incorrect_answer_3_input": incorrect_answer_3_input,
        }
        with open("logs/logs_cvqa.jsonl", "a+", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    initial_data_point = get_random_data_point("", "Argentina")

    # Gradio interface
    with gr.Blocks() as interface:
        _ = gr.Markdown(
            """
            # Actividad de Conocimiento Regional

            ### Por favor, añade preguntas de opción múltiple que sólo alguien de tu región sea capaz de responder.

            ### Para cada pregunta, proporciona el texto de la misma, varias opciones razonables de respuesta e indica cuál es la respuesta correcta.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                data_point_ID = gr.Textbox(
                    label="ID",
                    value=initial_data_point["ID"],
                    interactive=False,
                    visible=False,
                )
                data_point_IDX = gr.Textbox(
                    label="IDX",
                    value=initial_data_point["IDX"],
                    interactive=False,
                    visible=False,
                )
                data_point_image = gr.Image(
                    label="Image",
                    value=initial_data_point["image"],
                    interactive=False,
                )
            with gr.Column(scale=2):
                gr.Markdown("### Este es un ejemplo:")
                data_point_multiple_choice = gr.Radio(
                    label=initial_data_point["Question"],
                    choices=initial_data_point["Options"],
                    interactive=True,
                )
                gr.Markdown("### A partir de la misma imagen:")
                other_question_input = gr.Textbox(
                    label="Dar otra pregunta que sólo podría responder alguien de tu región:",
                    placeholder="Escribe aquí la pregunta",
                )
                correct_answer_input = gr.Textbox(
                    label="Dar la respuesta correcta:",
                    placeholder="Respuesta correcta",
                )
                incorrect_answer_1_input = gr.Textbox(
                    label="Dar una respuesta incorrecta posible:",
                    placeholder="Respuesta incorrecta 1",
                )
                incorrect_answer_2_input = gr.Textbox(
                    label="Dar una respuesta incorrecta posible:",
                    placeholder="Respuesta incorrecta 2",
                )
                incorrect_answer_3_input = gr.Textbox(
                    label="Dar una respuesta incorrecta posible:",
                    placeholder="Respuesta incorrecta 3",
                )
        with gr.Row(equal_height=True):
            llm_responses_button = gr.Button("Cómo responden los modelos de lenguaje?", interactive=False, variant="secondary", scale=75)
            next_button = gr.Button("Siguiente imagen", variant="primary", scale=25)

        gr.Markdown(
            f"""
            ### Aquí verás cómo responden diferentes modelos de lenguaje a tu pregunta de conocimiento regional.
            Las respuestas se generan automáticamente para mostrar cómo {list(models.keys())[0]}, {list(models.keys())[1]} y {list(models.keys())[2]} podrían contestar la pregunta que escribiste.
            """
        )
        with gr.Row():
            model_a_response = gr.HighlightedText(
                label=list(models.keys())[0],
                value=[],
                combine_adjacent=True,
                show_legend=False,
                interactive=False,
                color_map={"✓": "green", "X": "red"}
            )
            model_b_response = gr.HighlightedText(
                label=list(models.keys())[1],
                value=[],
                combine_adjacent=True,
                show_legend=False,
                interactive=False,
                color_map={"✓": "green", "X": "red"}
            )
            model_c_response = gr.HighlightedText(
                label=list(models.keys())[2],
                value=[],
                combine_adjacent=True,
                show_legend=False,
                interactive=False,
                color_map={"✓": "green", "X": "red"}
            )

        with Modal(visible=False) as validation_modal:
            _ = gr.HTML("<h1>Veamos que preguntas generaron otros participantes!</h1>")
            with gr.Row():
                with gr.Column():
                    modal_data_point_ID = gr.Textbox(
                        label="ID",
                        value=initial_data_point["ID"],
                        interactive=False,
                        visible=False,
                    )
                    modal_data_point_IDX = gr.Textbox(
                        label="IDX",
                        value=initial_data_point["IDX"],
                        interactive=False,
                        visible=False,
                    )
                    modal_data_point_image = gr.Image(
                        label="Image",
                        value=initial_data_point["image"],
                        interactive=False,
                    )
                with gr.Column(visible=False) as validation_1_col:
                    validation_1_multiple_choice = gr.Radio(
                        label=initial_data_point["Question"],
                        choices=initial_data_point["Options"],
                        interactive=True,
                    )
                    validation_1_q1 = gr.Slider(
                        1,
                        5,
                        value=None,
                        step=1,
                        label="¿La pregunta está bien hecha?",
                        info="1 es muy mala, 5 es excelente",
                        interactive=True,
                        # show_reset_button=False,
                    )
                    validation_1_label = gr.Dropdown(
                        choices=[
                            "Apariencia Física",
                            "Discapacidad",
                            "Edad",
                            "Etnia",
                            "Género",
                            "Nacionalidad",
                            "Orientación sexual",
                            "Profesión",
                            "Religión",
                            "Situación Socioeconómica"
                        ],
                        label="Qué tipos de sesgo se exploran aquí?",
                        info="Podés elegir de la lista o completar si consideras que falta alguna. Además podés elegir varios sesgos juntos.",
                        multiselect=True,
                        allow_custom_value=True
                    )
                with gr.Column(visible=False) as validation_2_col:
                    validation_2_multiple_choice = gr.Radio(
                        label=initial_data_point["Question"],
                        choices=initial_data_point["Options"],
                        interactive=True,
                    )
                    validation_2_q1 = gr.Slider(
                        1,
                        5,
                        value=None,
                        step=1,
                        label="¿La pregunta está bien hecha?",
                        info="1 es muy mala, 5 es excelente",
                        interactive=True,
                        # show_reset_button=False,
                    )
                    validation_2_label = gr.Dropdown(
                        choices=[
                            "Apariencia Física",
                            "Discapacidad",
                            "Edad",
                            "Etnia",
                            "Género",
                            "Nacionalidad",
                            "Orientación sexual",
                            "Profesión",
                            "Religión",
                            "Situación Socioeconómica"
                        ],
                        label="Qué tipos de sesgo se exploran aquí?",
                        info="Podés elegir de la lista o completar si consideras que falta alguna. Además podés elegir varios sesgos juntos.",
                        multiselect=True,
                        allow_custom_value=True
                    )
                with gr.Column(visible=False) as validation_3_col:
                    validation_3_multiple_choice = gr.Radio(
                        label=initial_data_point["Question"],
                        choices=initial_data_point["Options"],
                        interactive=True,
                    )
                    validation_3_q1 = gr.Slider(
                        1,
                        5,
                        value=None,
                        step=1,
                        label="¿La pregunta está bien hecha?",
                        info="1 es muy mala, 5 es excelente",
                        interactive=True,
                        # show_reset_button=False,
                    )
                    validation_3_label = gr.Dropdown(
                        choices=[
                            "Apariencia Física",
                            "Discapacidad",
                            "Edad",
                            "Etnia",
                            "Género",
                            "Nacionalidad",
                            "Orientación sexual",
                            "Profesión",
                            "Religión",
                            "Situación Socioeconómica"
                        ],
                        label="Qué tipos de sesgo se exploran aquí?",
                        info="Podés elegir de la lista o completar si consideras que falta alguna. Además podés elegir varios sesgos juntos.",
                        multiselect=True,
                        allow_custom_value=True
                    )
            with gr.Row():
                modal_skip_button = gr.Button("Omitir", variant="secondary", scale=75)
                modal_next_button = gr.Button("Siguiente", variant="primary", scale=25)

        # Functions ###########################################################

        # LLM Responses #######################################################
        
        def on_llm_responses_button(
            token_id,
            age,
            gender,
            nationality,
            region,
            school,
            consent_checkbox,
            data_point_ID,
            data_point_IDX,
            data_point_image,
            data_point_multiple_choice,
            other_question_input,
            correct_answer_input,
            incorrect_answer_1_input,
            incorrect_answer_2_input,
            incorrect_answer_3_input,
        ):

            # Inputs
            question = other_question_input
            correct = correct_answer_input
            incorrect = [incorrect_answer_1_input, incorrect_answer_2_input, incorrect_answer_3_input]

            # Combine and shuffle
            options = [correct] + incorrect
            random.shuffle(options)

            # Map to letters
            letters = ['A', 'B', 'C', 'D']
            lettered_options = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]

            # Find correct letter
            correct_letter = letters[options.index(correct)]

            # Prompts
            system_prompt = (
                "You will be given a multiple-choice question with four labeled options (A, B, C, D).\n"
                "You MUST respond with exactly one single uppercase letter: A, B, C, or D.\n"
                "Your entire response must be only that single character and nothing else — no punctuation, no quotes, no explanation, no option text, no extra whitespace or line breaks.\n"
                "If uncertain, choose the option you judge most likely. Any other output will be treated as invalid.\n\n"
                "Valid examples (exact response):\n"
                "A\n"
                "C\n\n"
                "Invalid examples (do not output these):\n"
                "A.\n"
                "A. Option text\n"
                "Option text\n"
                "A\\nBecause...\n"
            )

            multiple_choice_prompt = "\n".join(
                [question] + lettered_options + [
                    "",  # blank line to separate
                    "INSTRUCTIONS: Return exactly one uppercase letter (A, B, C, or D) as your entire response. Do NOT include the option text or any explanation."
                ]
            )

            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            image_path = f"/tmp/{token_id}_{timestamp_str}_data_point_image.png"
            # Create PIL image from array and resize so max dimension is 256
            img = Image.fromarray(data_point_image)
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            img.thumbnail((124, 124), resample)

            # Convert to RGB to reduce size / avoid alpha overhead
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")

            # Save as optimized PNG with strong compression
            img = img.quantize(colors=256, method=Image.MEDIANCUT)
            img.save(image_path, format="PNG", optimize=True, compress_level=9)
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            model_responses = []
            items = list(models.items())

            def _call_model(idx, model_name, model):
                try:
                    if isinstance(model, ModelWrapper):
                        response = model.invoke(
                            system_prompt,
                            multiple_choice_prompt,
                            base64_image=base64_image
                        )
                        response_content = response['content'].strip()
                    else:
                        response = model.invoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=[
                                {"type": "text", "text": multiple_choice_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ])
                        ])
                        # support objects with .content or dict-like responses
                        response_content = getattr(response, "content", None) or (response.get('content') if isinstance(response, dict) else None) or str(response)
                        response_content = response_content.strip()

                    response_content = re.sub(r'[^A-Za-z0-9]', '', response_content)
                    # print(f"Response from {model_name}: {response_content}")
                    return idx, response_content
                except Exception as e:
                    print(f"Error invoking {model_name}: {e}")
                    return idx, ""

            # Run model invocations in parallel and preserve original order
            results = [None] * len(items)
            with ThreadPoolExecutor(max_workers=min(8, len(items))) as ex:
                futures = {ex.submit(_call_model, i, name, m): i for i, (name, m) in enumerate(items)}
                for fut in as_completed(futures):
                    idx, resp = fut.result()
                    results[idx] = resp

            model_responses.extend(results)

            log_result(
                "LLM_RESPONSES",
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                data_point_ID,
                data_point_IDX,
                data_point_multiple_choice,
                other_question_input,
                correct_answer_input,
                incorrect_answer_1_input,
                incorrect_answer_2_input,
                incorrect_answer_3_input,
            )

            def highlight_response(response):
                # Retrieve the option without the letter in response
                option_text = None
                for opt in lettered_options:
                    if opt.startswith(response.upper()):
                        option_text = opt[2:].strip()
                        break
                if option_text is None:
                    option_text = response  # fallback if not found

                if response.strip().lower() == correct_letter.strip().lower():
                    return [(option_text, "✓")]
                else:
                    return [(option_text, "X")]

            return (
                gr.update(value=highlight_response(model_responses[0])),
                gr.update(value=highlight_response(model_responses[1])),
                gr.update(value=highlight_response(model_responses[2])),
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
                data_point_ID,
                data_point_IDX,
                data_point_image,
                data_point_multiple_choice,
                other_question_input,
                correct_answer_input,
                incorrect_answer_1_input,
                incorrect_answer_2_input,
                incorrect_answer_3_input,
            ],
            outputs=[
                model_a_response,
                model_b_response,
                model_c_response
            ]
        )

        # SKIP ################################################################

        def on_next(
            token_id,
            age,
            gender,
            nationality,
            region,
            school,
            consent_checkbox,
            data_point_ID,
            data_point_IDX,
            data_point_image,
            data_point_multiple_choice,
            other_question_input,
            correct_answer_input,
            incorrect_answer_1_input,
            incorrect_answer_2_input,
            incorrect_answer_3_input,
        ):

            log_result(
                "NEXT_BUTTON",
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                data_point_ID,
                data_point_IDX,
                data_point_multiple_choice,
                other_question_input,
                correct_answer_input,
                incorrect_answer_1_input,
                incorrect_answer_2_input,
                incorrect_answer_3_input,
            )
            
            # Check if there are existing logs for this data point            
            cvqa_logs_df = pd.read_json("logs/logs_cvqa.jsonl", lines=True)
            existing_logs_for_data_point = cvqa_logs_df[cvqa_logs_df["data_point_IDX"] == int(data_point_IDX)]
            # Shuffle existing logs so we show different ones each time
            existing_logs_for_data_point = existing_logs_for_data_point.sample(frac=1).reset_index(drop=True)

            # Determine if we should show the validation modal
            # show_modal = not existing_logs_for_data_point.empty
            show_modal = False # TEMPORARY
            
            # Prepare modal data - get up to 3 unique logs
            modal_data = []
            if show_modal:
                for idx in range(min(3, len(existing_logs_for_data_point))):
                    log = existing_logs_for_data_point.iloc[idx]
                    modal_data.append({
                        "question": log["other_question_input"],
                        "choices": [
                            log["correct_answer_input"],
                            log["incorrect_answer_1_input"],
                            log["incorrect_answer_2_input"],
                            log["incorrect_answer_3_input"],
                        ]
                    })
            
            # Build return values for validation columns
            validation_col_updates = []
            validation_radio_updates = []
            for i in range(3):
                if i < len(modal_data):
                    validation_col_updates.append(gr.Column(visible=True))
                    validation_radio_updates.append(gr.Radio(
                        label=modal_data[i]["question"],
                        choices=modal_data[i]["choices"],
                        interactive=True,
                    ))
                else:
                    validation_col_updates.append(gr.Column(visible=False))
                    validation_radio_updates.append(gr.Radio(
                        label="",
                        choices=[],
                        interactive=True,
                    ))
            
            new_data_point = get_random_data_point(token_id, nationality)
            return (
                new_data_point["ID"],
                new_data_point["IDX"],
                new_data_point["image"],
                gr.Radio(
                    label=new_data_point["Question"],
                    choices=new_data_point["Options"],
                    interactive=True,
                ),
                "",
                "",
                "",
                "",
                "",
                Modal(visible=show_modal),
                data_point_image,
                validation_col_updates[0],
                validation_col_updates[1],
                validation_col_updates[2],
                validation_radio_updates[0],
                validation_radio_updates[1],
                validation_radio_updates[2],
            )


        next_button.click(
            on_next,
            inputs=[
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                consent_checkbox,
                data_point_ID,
                data_point_IDX,
                data_point_image,
                data_point_multiple_choice,
                other_question_input,
                correct_answer_input,
                incorrect_answer_1_input,
                incorrect_answer_2_input,
                incorrect_answer_3_input,
            ],
            outputs=[
                data_point_ID,
                data_point_IDX,
                data_point_image,
                data_point_multiple_choice,
                other_question_input,
                correct_answer_input,
                incorrect_answer_1_input,
                incorrect_answer_2_input,
                incorrect_answer_3_input,
                validation_modal,
                modal_data_point_image,
                validation_1_col,
                validation_2_col,
                validation_3_col,
                validation_1_multiple_choice,
                validation_2_multiple_choice,
                validation_3_multiple_choice,
            ],
        )

        # VALIDATION MODAL ####################################################

        def on_modal_next_button(
            token_id,
            age,
            gender,
            modal_data_point_ID,
            modal_data_point_IDX,
            nationality,
            region,
            school,
            consent_checkbox,
            validation_1_multiple_choice,
            validation_1_q1,
            validation_1_label,
            validation_2_multiple_choice,
            validation_2_q1,
            validation_2_label,
            validation_3_multiple_choice,
            validation_3_q1,
            validation_3_label,
        ):

            result = {
                "timestamp": datetime.now().isoformat(),
                "token_id": token_id,
                "age": age,
                "gender": gender,
                "modal_data_point_ID": modal_data_point_ID,
                "modal_data_point_IDX": modal_data_point_IDX,
                "nationality": nationality,
                "region": region,
                "school": school,
                "consent_checkbox": consent_checkbox,
                "validation_1_multiple_choice": validation_1_multiple_choice,
                "validation_1_q1": validation_1_q1,
                "validation_1_label": validation_1_label,
                "validation_2_multiple_choice": validation_2_multiple_choice,
                "validation_2_q1": validation_2_q1,
                "validation_2_label": validation_2_label,
                "validation_3_multiple_choice": validation_3_multiple_choice,
                "validation_3_q1": validation_3_q1,
                "validation_3_label": validation_3_label,
            }
            with open("logs/logs_validation_cvqa.jsonl", "a+", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            return (
                Modal(visible=False),
                gr.update(),
            )

        modal_next_button.click(
            on_modal_next_button,
            inputs=[
                token_id,
                age,
                gender,
                modal_data_point_ID,
                modal_data_point_IDX,
                nationality,
                region,
                school,
                consent_checkbox,
                validation_1_multiple_choice,
                validation_1_q1,
                validation_1_label,
                validation_2_multiple_choice,
                validation_2_q1,
                validation_2_label,
                validation_3_multiple_choice,
                validation_3_q1,
                validation_3_label,
            ],
            outputs=[
                validation_modal,
                modal_data_point_image
            ],
        )

        def on_modal_skip_button():
            return (
                Modal(visible=False),
                gr.update(),
            )

        modal_skip_button.click(
            on_modal_skip_button,
            outputs=[
                validation_modal,
                modal_data_point_image
            ],
        )

        # LLM Responses Toggle

        def toggle_llm_responses(
            other_question_input,
            correct_answer_input,
            incorrect_answer_1_input,
            incorrect_answer_2_input,
            incorrect_answer_3_input
        ):
            if all([
                other_question_input,
                correct_answer_input,
                incorrect_answer_1_input,
                incorrect_answer_2_input,
                incorrect_answer_3_input
            ]):
                return (
                    gr.Button("Cómo responden los modelos de lenguaje?", interactive=True, variant="secondary", scale=50)
                )
            else:
                return (
                    gr.Button("Cómo responden los modelos de lenguaje?", interactive=False, variant="secondary", scale=50)
                )

        toggle_llm_responses_inputs = [
            other_question_input,
            correct_answer_input,
            incorrect_answer_1_input,
            incorrect_answer_2_input,
            incorrect_answer_3_input,
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



    return interface
