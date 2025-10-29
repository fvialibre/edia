import base64
import json
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
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

ds = load_from_disk("cvqa_argentina")

secrets = dotenv_values("./.env")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
os.environ["OLLAMA_API_KEY"] = secrets["OLLAMA_API_KEY"]
os.environ['COHERE_API_KEY'] = secrets['COHERE_API_KEY']
os.environ['GOOGLE_API_KEY'] = secrets['GOOGLE_API_KEY']

models = {
    "openai": ChatOpenAI(api_key=secrets["OPENAI_API_KEY"], model="gpt-5-nano", temperature=1, max_retries=3),
    # "cohere": ChatCohere(cohere_api_key=secrets["COHERE_API_KEY"], model="command-r", temperature=1, max_retries=3),
    # "google": ChatGoogleGenerativeAI(google_api_key=secrets["GOOGLE_API_KEY"], model="gemini-2.0-flash", temperature=1, max_retries=3),
    "gemma3:4b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="gemma3:4b"),
    # "llama3.1:8b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="llama3.1:8b"),
    "llava:34b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="llava:34b"),
    # "mistral:7b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="mistral:7b"),
    # "gpt-oss:20b": ModelWrapper(token=secrets["OLLAMA_API_KEY"], model="gpt-oss:20b"),

}

# --- Interface ---
def interface(lang: str) -> gr.Blocks:

    def get_random_data_point():
        data_point = ds.shuffle().select(range(1))[0]
        return {
            "ID": data_point["ID"],
            "image": data_point["image"],
            "Question": data_point["Question"],
            "Options": data_point["Options"],
        }

    def log_result(
        source,
        token_id,
        age,
        gender,
        nationality_personal_info,
        consent_checkbox,
        data_point_ID,
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
            "nationality_personal_info": nationality_personal_info,
            "consent_checkbox": consent_checkbox,
            "data_point_ID": data_point_ID,
            "data_point_multiple_choice": data_point_multiple_choice,
            "other_question_input": other_question_input,
            "correct_answer_input": correct_answer_input,
            "incorrect_answer_1_input": incorrect_answer_1_input,
            "incorrect_answer_2_input": incorrect_answer_2_input,
            "incorrect_answer_3_input": incorrect_answer_3_input,
        }
        with open("logs/logs_cvqa.jsonl", "a+", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    initial_data_point = get_random_data_point()

    # Gradio interface
    with gr.Blocks() as interface:
        with gr.Row():
            token_id = gr.Textbox(
                label="Identifier",
                info="Enter the identifier provided in the workshop",
                lines=1,
            )
            age = gr.Number(
                value=0,
                label="Enter your age",
                visible=False,
            )
            gender = gr.Radio(
                ["M", "F", "X"],
                label="Select your gender",
                value="X",
                visible=False,
            )
            nationality_personal_info = gr.Dropdown(
                label="Where are you from?",
                info="Select the nationality that represents your cultural, personal, or national identity",
                choices=nationalities,
                multiselect=True,
                allow_custom_value=False,
            )
            with gr.Column():
                consent_checkbox = gr.Checkbox(
                    label="I have read and accept the informed consent ⬇️", value=False
                )
                _ = gr.HTML(
                    value="<a href='https://docs.google.com/document/d/1YEi0QpFYJwFBSIAjGplPc0VkOxJwnME29dWWyfp37XY/edit?usp=sharing'>Link 🔗</a>",
                )
        _ = gr.HTML(
            value="<hr>",
        )
        with gr.Column(visible=True) as personal_data_missing:
            gr.Markdown(
                """
                # Enter your personal data and confirm your consent to proceed with the survey!
            """
            )
        with gr.Column(visible=False, elem_id="annotation_col") as annotation_col:
            _ = gr.Markdown(
                """
                # Regional Knowledge Activity

                ### Please add multiple choice questions that only someone from your region should be able to answer.

                ### For each question, provide the question text, several answer options, and indicate the correct answer.
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
                llm_responses_button = gr.Button("Cómo responden los modelos de lenguaje?", interactive=False, variant="secondary", scale=50)
                next_button = gr.Button("Siguiente", variant="primary", scale=25)

        with gr.Column(visible=False, elem_id="llm_responses_col") as llm_responses_col:
            gr.Markdown(
                f"""
                ### Aquí verás cómo responden diferentes modelos de lenguaje a tu pregunta regional.
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
                    modal_data_point_image = gr.Image(
                        label="Image",
                        value=initial_data_point["image"],
                        interactive=False,
                    )
                with gr.Column():
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
                        show_reset_button=False,
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
                with gr.Column():
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
                        show_reset_button=False,
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
                with gr.Column():
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
                        show_reset_button=False,
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
            nationality_personal_info,
            consent_checkbox,
            data_point_ID,
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
                    print(f"Response from {model_name}: {response_content}")
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
                nationality_personal_info,
                consent_checkbox,
                data_point_ID,
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
                nationality_personal_info,
                consent_checkbox,
                data_point_ID,
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
            nationality_personal_info,
            consent_checkbox,
            data_point_ID,
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
                nationality_personal_info,
                consent_checkbox,
                data_point_ID,
                data_point_multiple_choice,
                other_question_input,
                correct_answer_input,
                incorrect_answer_1_input,
                incorrect_answer_2_input,
                incorrect_answer_3_input,
            )
            
            modal_data_point_image = data_point_image 
            # TODO: Add other datapoints for validation
            
            
            new_data_point = get_random_data_point()
            return (
                new_data_point["ID"],
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
                Modal(visible=True),
                modal_data_point_image
            )


        next_button.click(
            on_next,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                data_point_ID,
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
                data_point_image,
                data_point_multiple_choice,
                other_question_input,
                correct_answer_input,
                incorrect_answer_1_input,
                incorrect_answer_2_input,
                incorrect_answer_3_input,
                validation_modal,
                modal_data_point_image
            ],
        )

        # VALIDATION MODAL ####################################################

        def on_modal_next_button(
            token_id,
            age,
            gender,
            nationality_personal_info,
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
                "nationality_personal_info": nationality_personal_info,
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
                nationality_personal_info,
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

        # Annotation Toggle

        def toggle_annotation(
            token_id, age, gender, nationality_personal_info, consent_checkbox
        ):
            if any([
                token_id is None,
                age is None,
                gender is None,
                nationality_personal_info is None,
                consent_checkbox is None,
                age < 0,
                age > 100,
                len(nationality_personal_info) == 0,
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
                    new_data_point["ID"],
                    new_data_point["image"],
                    gr.Radio(
                        label=new_data_point["Question"],
                        choices=new_data_point["Options"],
                        interactive=True,
                    ),
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
            data_point_ID,
            data_point_image,
            data_point_multiple_choice,
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
