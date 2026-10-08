import uuid
import re
import gradio as gr
from gradio_modal import Modal
from gradio_i18n import gettext as i18n
import pandas as pd
from modules.module_ollama import ModelWrapper
import json
from datetime import datetime
import os
from dotenv import dotenv_values
from data.clinical_patients.clinical_prompts import clinical_prompts, clinical_extra_studies
from html_constants import HTML_FEEDBACK_TITLE
from modules.module_patientAssignment import assign_patient

# Empty consulting room, shown behind the chat before a patient is picked or after a conversation ends.
CLINICAL_CHATBOT_BACKGROUND_URI = "https://i.imgur.com/NQfqr2f.jpeg"

# Shown once a patient is assigned but the conversation hasn't started yet.
CLINICAL_CHATBOT_PATIENT_WAITING_URI = "https://i.imgur.com/WOe9nX0.jpeg"

# Roles allowed to manually override the sampled patient.
CLINICAL_ADMIN_ROLES = ("ClinicalChatbotOther")

PATIENT_NAME_PATTERN = re.compile(r"Nombre y Apellido:\s*(.+)")

def get_patient_name(patient_id):
    """Pulls the patient's display name out of their prompt, falling back to the ID."""
    match = PATIENT_NAME_PATTERN.search(clinical_prompts.get(patient_id, ""))
    return match.group(1).strip().rstrip(".") if match else patient_id

def build_chat_placeholder(has_patient):
    """Builds the Chatbot placeholder HTML, swapping the image/text once a patient is assigned."""
    image_uri = CLINICAL_CHATBOT_PATIENT_WAITING_URI if has_patient else CLINICAL_CHATBOT_BACKGROUND_URI
    text_key = "ClinicalChatbotChatPlaceholderPatientReady" if has_patient else "ClinicalChatbotChatPlaceholder"
    return (
        f'<img src="{image_uri}" '
        'style="max-width:min(100%,480px); border-radius:12px; margin-bottom:1em;" /><div>'
        + i18n(text_key)
        + "</div>"
    )

# --- Interface ---
def interface(
    token_id,
    age,
    gender,
    nationality,
    region,
    school,
    consent_checkbox,
    participant_area,
    contact_with_real_patients=None,
    **kwargs,
) -> gr.Blocks:
    if contact_with_real_patients is None:
        for alias in (
            "real_patient_contact",
            "contact_with_real_patients_checkbox",
            "real_patient_contact_checkbox",
            "has_contact_with_real_patients",
        ):
            if alias in kwargs:
                contact_with_real_patients = kwargs[alias]
                break
        else:
            contact_with_real_patients = gr.State(False)

    secrets = dotenv_values("./.env")
    os.environ["OLLAMA_API_KEY"] = secrets["OLLAMA_API_KEY"]
    llm = ModelWrapper(
        token=secrets["OLLAMA_API_KEY"],
        model="vllm/gemma4-26b",
    )

    def predict(
        message,
        history,
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        patient_id,
        participant_area,
        contact_with_real_patients,
    ):
        temperature = 1.0
        if patient_id not in clinical_prompts:
            return "Please select a patient before starting the consultation."

        prompt = clinical_prompts[patient_id]
        history_text = []
        for human, ai in history:
            history_text.append(f"User: {human}")
            history_text.append(f"Assistant: {ai}")
        history_text.append(f"User: {message}")
        user_prompt = "\n".join(history_text)
        model_response = llm.invoke(prompt, user_prompt)
        response_content = model_response["content"]

        with open("./logs/logs_clinical_chatbot.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "token_id": token_id,
                "age": age,
                "gender": gender,
                "nationality": nationality,
                "region": region,
                "school": school,
                "patient_id": patient_id,
                "participant_area": participant_area,
                "contact_with_real_patients": contact_with_real_patients,
                "message": message,
                "response": response_content,
                "history": history,
                "current_prompt": prompt,
                "temperature": temperature
            }, ensure_ascii=False) + "\n")
        return response_content
    
    def open_turn_feedback_modal(
        x: gr.LikeData,
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        patient_id,
        participant_area,
        contact_with_real_patients,
    ):
        return {
            "selected_message": x.value,
            "is_like": x.liked,
            "turn_index": x.index,
            "token_id": token_id,
            "age": age,
            "gender": gender,
            "nationality": nationality,
            "region": region,
            "school": school,
            "participant_area": participant_area,
            "contact_with_real_patients": contact_with_real_patients,
            "patient_id": patient_id,
        }, Modal(visible=True)

    def send_turn_feedback_modal(turn_info_for_feedback, text_feedback):
        with open("./logs/logs_clinical_chatbot_feedback.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "selected_message": turn_info_for_feedback["selected_message"],
                "is_like": turn_info_for_feedback["is_like"],
                "turn_index": turn_info_for_feedback["turn_index"],
                "text_feedback": text_feedback,
                "token_id": turn_info_for_feedback["token_id"],
                "age": turn_info_for_feedback["age"],
                "gender": turn_info_for_feedback["gender"],
                "nationality": turn_info_for_feedback["nationality"],
                "region": turn_info_for_feedback["region"],
                "school": turn_info_for_feedback["school"],
                "participant_area": turn_info_for_feedback["participant_area"],
                "contact_with_real_patients": turn_info_for_feedback.get("contact_with_real_patients"),
                "patient_id": turn_info_for_feedback["patient_id"],
                "prompt": clinical_prompts[turn_info_for_feedback["patient_id"]],
            }, ensure_ascii=False) + "\n")
        gr.Info("Feedback enviado con éxito")
        return (Modal(visible=False), "")

    def toggle_complementary_studies(is_open, patient_id):
        is_open = not is_open
        if is_open:
            text = clinical_extra_studies.get(patient_id, i18n("ClinicalChatbotComplementaryStudiesPlaceholder"))
            text_box = gr.Textbox(value=text, visible=True)
            button = gr.Button(i18n("ClinicalChatbotComplementaryStudiesCloseButton"))
        else:
            text_box = gr.Textbox(visible=False)
            button = gr.Button(i18n("ClinicalChatbotComplementaryStudiesOpenButton"))
        return is_open, text_box, button

    def reset_complementary_studies():
        return False, gr.Textbox(visible=False), gr.Button(i18n("ClinicalChatbotComplementaryStudiesOpenButton"))

    def reset_on_patient_change(patient_id):
        is_open, text_box, button = reset_complementary_studies()
        has_patient = bool(patient_id)
        sample_patient_visibility = gr.update(visible=not has_patient)
        end_conversation_visibility = gr.update(visible=has_patient)
        chatbot_update = gr.Chatbot(value=[], placeholder=build_chat_placeholder(has_patient))
        return is_open, text_box, button, chatbot_update, sample_patient_visibility, end_conversation_visibility

    def sample_new_patient(token_id, participant_area, patient_id):
        next_patient_id = assign_patient(token_id, participant_area, exclude_patient_id=patient_id)
        gr.Info(i18n("ClinicalChatbotNewPatientToast").format(patient=get_patient_name(next_patient_id)))
        return next_patient_id

    def toggle_patient_override(participant_area):
        return gr.update(interactive=participant_area in CLINICAL_ADMIN_ROLES)

    def send_end_conversation_form(
        acute_problems,
        chronic_problems,
        diagnosis,
        general_feedback,
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        patient_id,
        participant_area,
        contact_with_real_patients,
    ):
        with open("./logs/logs_clinical_chatbot_end_conversation.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "acute_problems": acute_problems,
                "chronic_problems": chronic_problems,
                "diagnosis": diagnosis,
                "general_feedback": general_feedback,
                "token_id": token_id,
                "age": age,
                "gender": gender,
                "nationality": nationality,
                "region": region,
                "school": school,
                "patient_id": patient_id,
                "participant_area": participant_area,
                "contact_with_real_patients": contact_with_real_patients,
            }, ensure_ascii=False) + "\n")
        gr.Info("Feedback enviado con éxito")
        is_open, text_box, button = reset_complementary_studies()
        # Leave the room empty; the student has to call in a new patient explicitly.
        return (Modal(visible=False), "", "", "", "", [], is_open, text_box, button, None)

    chat_placeholder = build_chat_placeholder(has_patient=False)

    with gr.Blocks(css=".contain { display: flex !important; flex-direction: column !important; }"
    "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
    "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
    f"#chatbot .bubble-wrap, #chatbot .wrap {{ background-image: url('{CLINICAL_CHATBOT_BACKGROUND_URI}') !important; background-size: cover !important; background-position: center !important; background-repeat: no-repeat !important; }}"
    "#col { height: 100vh !important; }") as interface:
        turn_info_for_feedback = gr.State({
            "selected_message": None,
            "is_like": None,
            "turn_index": None,
            "token_id": None,
            "age": None,
            "gender": None,
            "nationality": None,
            "region": None,
            "school": None,
            "patient_id": None,
            "participant_area": None,
            "contact_with_real_patients": None,
        })
        gr.HTML("<h1 style='text-align: center;'>" + i18n("ClinicalChatbotTitle") + "</h1>")
        gr.Markdown(i18n("ClinicalChatbotDescription"))
        with gr.Row():
            patient_id = gr.Dropdown(
                choices=list(clinical_prompts),
                label=i18n("ClinicalChatbotPatientLabel"),
                interactive=False,
                scale=3,
            )
            sample_patient_button = gr.Button(
                i18n("ClinicalChatbotSamplePatientButton"),
                variant="stop",
                elem_id="sample-patient-button",
                scale=1,
            )
        
        gr.HTML("<hr>")

        with gr.Row(visible=True, elem_id='col') as chat_col:
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    show_copy_button=True,
                    placeholder=chat_placeholder,
                    # likeable=True,
                )
                chat_interface = gr.ChatInterface(
                        predict,
                        additional_inputs=[
                            token_id,
                            age,
                            gender,
                            nationality,
                            region,
                            school,
                            patient_id,
                            participant_area,
                            contact_with_real_patients,
                        ],
                        chatbot=chatbot,
                        submit_btn=i18n("ClinicalChatbotSubmitButton"),
                        stop_btn=None,
                    )
            with gr.Column(scale=1, elem_id="complementary-studies-col"):
                gr.Markdown("## " + i18n("ClinicalChatbotComplementaryStudiesTitle"))
                complementary_studies_is_open = gr.State(False)
                complementary_studies_open_button = gr.Button(i18n("ClinicalChatbotComplementaryStudiesOpenButton"))
                complementary_studies_text = gr.Textbox(
                    show_label=False,
                    lines=15,
                    interactive=False,
                    visible=False,
                )

        with gr.Row():
            end_conversation_button = gr.Button(
                i18n("ClinicalChatbotEndConversationButton"),
                variant="stop",
                elem_id="end-conversation-button",
                visible=False,
            )

        ### MODAL
        with Modal(visible=False) as turn_feedback_modal:
            _ = gr.HTML(HTML_FEEDBACK_TITLE)

            with gr.Row():
                text_feedback = gr.Textbox(
                    label=i18n("ClinicalChatbotFeedbackLabel"),
                    lines=2,
                    placeholder=i18n("ClinicalChatbotFeedbackPlaceholder"),
                )

            modal_submit_button = gr.Button(i18n("ClinicalChatbotSubmitButton"))

        with Modal(visible=False) as end_conversation_modal:
            gr.HTML(
                "<h1 style='text-align: center; margin-bottom: 0em;'>"
                + i18n("ClinicalChatbotEndConversationModalTitle")
                + "</h1><h3 style='text-align: center; margin-bottom: 1em;'>"
                + i18n("ClinicalChatbotEndConversationModalDescription")
                + "</h3>"
            )
            acute_problems = gr.Textbox(
                label=i18n("ClinicalChatbotAcuteProblemsLabel"),
                placeholder=i18n("ClinicalChatbotAcuteProblemsPlaceholder"),
                lines=2,
            )
            chronic_problems = gr.Textbox(
                label=i18n("ClinicalChatbotChronicProblemsLabel"),
                placeholder=i18n("ClinicalChatbotChronicProblemsPlaceholder"),
                lines=2,
            )
            diagnosis = gr.Textbox(
                label=i18n("ClinicalChatbotDiagnosisLabel"),
                placeholder=i18n("ClinicalChatbotDiagnosisPlaceholder"),
                lines=2,
            )
            general_feedback = gr.Textbox(
                label=i18n("ClinicalChatbotGeneralFeedbackLabel"),
                placeholder=i18n("ClinicalChatbotGeneralFeedbackPlaceholder"),
                lines=3,
            )
            end_conversation_submit_button = gr.Button(
                i18n("ClinicalChatbotEndConversationSubmitButton"),
                variant="stop",
            )

        chatbot.like(
            open_turn_feedback_modal,
            [token_id, age, gender, nationality, region, school, patient_id, participant_area, contact_with_real_patients],
            [turn_info_for_feedback, turn_feedback_modal],
        )
        modal_submit_button.click(send_turn_feedback_modal, [turn_info_for_feedback, text_feedback], [turn_feedback_modal, text_feedback])
        complementary_studies_open_button.click(
            toggle_complementary_studies,
            [complementary_studies_is_open, patient_id],
            [complementary_studies_is_open, complementary_studies_text, complementary_studies_open_button],
        )
        patient_id.change(
            reset_on_patient_change,
            [patient_id],
            [complementary_studies_is_open, complementary_studies_text, complementary_studies_open_button, chatbot, sample_patient_button, end_conversation_button],
        )
        sample_patient_button.click(
            sample_new_patient,
            [token_id, participant_area, patient_id],
            [patient_id],
        )
        participant_area.change(
            toggle_patient_override,
            [participant_area],
            [patient_id],
        )
        end_conversation_button.click(lambda: Modal(visible=True), None, end_conversation_modal)
        end_conversation_submit_button.click(
            send_end_conversation_form,
            [
                acute_problems,
                chronic_problems,
                diagnosis,
                general_feedback,
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                patient_id,
                participant_area,
                contact_with_real_patients,
            ],
            [end_conversation_modal, acute_problems, chronic_problems, diagnosis, general_feedback, chatbot, complementary_studies_is_open, complementary_studies_text, complementary_studies_open_button, patient_id],
        )
    return interface
