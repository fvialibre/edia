import uuid
import gradio as gr
from gradio_modal import Modal
from gradio_i18n import gettext as i18n
import pandas as pd
from modules.module_ollama import ModelWrapper
import json
from datetime import datetime
import os
from dotenv import dotenv_values
from data.clinical_patients.clinical_prompts import clinical_patients
from html_constants import HTML_FEEDBACK_TITLE
from modules.module_patientAssignment import assign_patient

# Empty consulting room, shown behind the chat before a patient is picked or after a conversation ends.
CLINICAL_CHATBOT_BACKGROUND_URI = "https://i.imgur.com/NQfqr2f.jpeg"

# Shown once a patient is assigned but the conversation hasn't started yet.
CLINICAL_CHATBOT_PATIENT_WAITING_URI = "https://i.imgur.com/JDSFVDs.jpeg"

def get_patient_name(patient_id):
    """Pulls the patient's display name, falling back to the ID."""
    patient = clinical_patients.get(patient_id)
    return patient.name if patient else patient_id

def build_patient_card(patient_id):
    """Builds a styled Markdown card displaying the active patient's name or waiting room status."""
    if not patient_id:
        return (
            '<div style="display: flex; align-items: center; gap: 14px; padding: 12px 18px; '
            'border-radius: 10px; border: 1.5px dashed rgba(156, 163, 175, 0.5); '
            'background: rgba(156, 163, 175, 0.07); min-height: 52px; box-sizing: border-box;">'
            '<span style="font-size: 1.7em; line-height: 1;">🚪</span>'
            '<div>'
            '<div style="font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7;">'
            + i18n("ClinicalChatbotPatientLabel")
            + '</div>'
            '<div style="font-size: 1.05em; font-weight: 500; opacity: 0.65; font-style: italic;">'
            + i18n("ClinicalChatbotNoPatientAssigned")
            + '</div>'
            '</div>'
            '</div>'
        )
    patient_name = get_patient_name(patient_id)
    return (
        '<div style="display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 12px 18px; '
        'border-radius: 10px; border: 1.5px solid rgba(59, 130, 246, 0.55); '
        'background: rgba(59, 130, 246, 0.08); min-height: 52px; box-sizing: border-box;">'
        '<div style="display: flex; align-items: center; gap: 14px;">'
        '<span style="font-size: 1.7em; line-height: 1;">🧑‍⚕️</span>'
        '<div>'
        '<div style="font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #3b82f6;">'
        + i18n("ClinicalChatbotPatientInConsultation")
        + '</div>'
        f'<div style="font-size: 1.3em; font-weight: 700; letter-spacing: -0.01em;">{patient_name}</div>'
        '</div>'
        '</div>'
        '</div>'
    )

def build_chat_placeholder(has_patient):
    """Builds the Chatbot placeholder HTML, swapping the image/text once a patient is assigned."""
    image_uri = CLINICAL_CHATBOT_PATIENT_WAITING_URI if has_patient else CLINICAL_CHATBOT_BACKGROUND_URI
    text_key = "ClinicalChatbotChatPlaceholderPatientReady" if has_patient else "ClinicalChatbotChatPlaceholder"
    return (
        '<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; width: 100%; margin: 0 auto; padding: 1.5em 1em;">'
        f'<img src="{image_uri}" '
        'style="max-width: min(100%, 480px); height: auto; border-radius: 12px; margin: 0 auto 1.25em auto; display: block; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);" />'
        '<div style="text-align: center; max-width: 480px; margin: 0 auto;">'
        + i18n(text_key)
        + "</div></div>"
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
        contact_with_real_patients = gr.State(False)

    secrets = dotenv_values("./.env")
    os.environ["OLLAMA_API_KEY"] = secrets["OLLAMA_API_KEY"]
    llm = ModelWrapper(
        token=secrets["OLLAMA_API_KEY"],
        model="vllm/gemma4-26b",
    )

    def log_clinical_action(action, conversation_id, token_id, patient_id):
        try:
            with open("./logs/logs_clinical_chatbot_actions.jsonl", "a+", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": str(conversation_id) if conversation_id else None,
                    "action": action,
                    "token_id": token_id,
                    "patient_id": patient_id,
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error logging clinical action: {e}")

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
        conversation_id,
    ):
        temperature = 1.0
        patient = clinical_patients.get(patient_id)
        if not patient:
            return i18n("ClinicalChatbotNoPatientAssigned")

        prompt = patient.prompt
        messages = [{"role": "system", "content": prompt}]
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "user")
                if role == "human":
                    role = "user"
                elif role in ("ai", "model"):
                    role = "assistant"
                messages.append({"role": role, "content": item.get("content", "")})
            else:
                human, ai = item
                if human:
                    messages.append({"role": "user", "content": human})
                if ai:
                    messages.append({"role": "assistant", "content": ai})
        messages.append({"role": "user", "content": message})

        model_response = llm.invoke(messages, temperature=temperature)
        response_content = model_response["content"]

        conv_id = str(conversation_id) if conversation_id else str(uuid.uuid4())
        turn_id = (len(history) // 2) + 1

        with open("./logs/logs_clinical_chatbot.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conv_id,
                "turn_id": turn_id,
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
        conversation_id,
    ):
        turn_idx = x.index
        if isinstance(turn_idx, (list, tuple)):
            turn_num = turn_idx[0] + 1
        elif isinstance(turn_idx, int):
            turn_num = (turn_idx // 2) + 1
        else:
            turn_num = None

        return {
            "conversation_id": str(conversation_id) if conversation_id else None,
            "turn_id": turn_num,
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
        patient = clinical_patients.get(turn_info_for_feedback["patient_id"])
        with open("./logs/logs_clinical_chatbot_feedback.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "conversation_id": turn_info_for_feedback.get("conversation_id"),
                "turn_id": turn_info_for_feedback.get("turn_id"),
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
                "prompt": patient.prompt if patient else "",
            }, ensure_ascii=False) + "\n")
        gr.Info("Feedback enviado con éxito")
        return (Modal(visible=False), "")

    def toggle_physical_exam(is_open, patient_id, conversation_id=None, token_id=None):
        if not patient_id:
            return False, gr.Textbox(visible=False), gr.Button(i18n("ClinicalChatbotPhysicalExamOpenButton"), interactive=False)
        is_open = not is_open
        if is_open:
            patient = clinical_patients.get(patient_id)
            physical_exam = patient.physical_exam if patient else ""
            text = physical_exam or i18n("ClinicalChatbotPhysicalExamPlaceholder")
            text_box = gr.Textbox(value=text, visible=True)
            button = gr.Button(i18n("ClinicalChatbotPhysicalExamCloseButton"), interactive=True)
            log_clinical_action("open_physical_exam", conversation_id, token_id, patient_id)
        else:
            text_box = gr.Textbox(visible=False)
            button = gr.Button(i18n("ClinicalChatbotPhysicalExamOpenButton"), interactive=True)
            log_clinical_action("close_physical_exam", conversation_id, token_id, patient_id)
        return is_open, text_box, button

    def reset_physical_exam(has_patient=False):
        return False, gr.Textbox(visible=False), gr.Button(i18n("ClinicalChatbotPhysicalExamOpenButton"), interactive=has_patient)

    def toggle_complementary_studies(is_open, patient_id, conversation_id=None, token_id=None):
        if not patient_id:
            return False, gr.Textbox(visible=False), gr.Button(i18n("ClinicalChatbotComplementaryStudiesOpenButton"), interactive=False)
        is_open = not is_open
        if is_open:
            patient = clinical_patients.get(patient_id)
            extra_studies = patient.extra_studies if patient else ""
            text = extra_studies or i18n("ClinicalChatbotComplementaryStudiesPlaceholder")
            text_box = gr.Textbox(value=text, visible=True)
            button = gr.Button(i18n("ClinicalChatbotComplementaryStudiesCloseButton"), interactive=True)
            log_clinical_action("open_complementary_studies", conversation_id, token_id, patient_id)
        else:
            text_box = gr.Textbox(visible=False)
            button = gr.Button(i18n("ClinicalChatbotComplementaryStudiesOpenButton"), interactive=True)
            log_clinical_action("close_complementary_studies", conversation_id, token_id, patient_id)
        return is_open, text_box, button

    def reset_complementary_studies(has_patient=False):
        return False, gr.Textbox(visible=False), gr.Button(i18n("ClinicalChatbotComplementaryStudiesOpenButton"), interactive=has_patient)

    def reset_on_patient_change(patient_id):
        has_patient = bool(patient_id)
        exam_is_open, exam_text, exam_btn = reset_physical_exam(has_patient)
        studies_is_open, studies_text, studies_btn = reset_complementary_studies(has_patient)
        sample_patient_visibility = gr.update(visible=not has_patient)
        end_conversation_visibility = gr.update(visible=has_patient)
        chatbot_update = gr.Chatbot(value=[], type="messages", placeholder=build_chat_placeholder(has_patient))
        textbox_update = gr.update(interactive=has_patient, value="")
        cleared_chat_state = []
        new_conv_id = str(uuid.uuid4()) if has_patient else None
        patient_display_update = build_patient_card(patient_id)
        return (
            exam_is_open,
            exam_text,
            exam_btn,
            studies_is_open,
            studies_text,
            studies_btn,
            chatbot_update,
            sample_patient_visibility,
            end_conversation_visibility,
            textbox_update,
            cleared_chat_state,
            new_conv_id,
            patient_display_update,
        )

    def sample_new_patient(token_id, participant_area, patient_id):
        next_patient_id = assign_patient(token_id, participant_area, exclude_patient_id=patient_id)
        gr.Info(i18n("ClinicalChatbotNewPatientToast").format(patient=get_patient_name(next_patient_id)))
        return next_patient_id

    def send_end_conversation_form(
        acute_problems,
        chronic_problems,
        diagnosis,
        requested_studies,
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        patient_id,
        participant_area,
        contact_with_real_patients,
        conversation_id,
        chat_history,
    ):
        timestamp = datetime.now().isoformat()
        conv_id = str(conversation_id) if conversation_id else None
        patient = clinical_patients.get(patient_id)
        prompt = patient.prompt if patient else ""

        with open("./logs/logs_clinical_chatbot_end_conversation.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": timestamp,
                "conversation_id": conv_id,
                "acute_problems": acute_problems,
                "chronic_problems": chronic_problems,
                "diagnosis": diagnosis,
                "requested_studies": requested_studies,
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

        with open("./logs/clinical_encounters.jsonl", "a+", encoding='utf-8') as f:
            f.write(json.dumps({
                "conversation_id": conv_id,
                "timestamp": timestamp,
                "token_id": token_id,
                "patient_id": patient_id,
                "prompt": prompt,
                "participant": {
                    "age": age,
                    "gender": gender,
                    "nationality": nationality,
                    "region": region,
                    "school": school,
                    "participant_area": participant_area,
                    "contact_with_real_patients": contact_with_real_patients,
                },
                "messages": chat_history or [],
                "evaluation": {
                    "acute_problems": acute_problems,
                    "chronic_problems": chronic_problems,
                    "diagnosis": diagnosis,
                    "requested_studies": requested_studies,
                }
            }, ensure_ascii=False) + "\n")

        gr.Info("Feedback enviado con éxito")
        exam_is_open, exam_text, exam_btn = reset_physical_exam(has_patient=False)
        studies_is_open, studies_text, studies_btn = reset_complementary_studies(has_patient=False)
        chatbot_update = gr.Chatbot(value=[], type="messages", placeholder=build_chat_placeholder(has_patient=False))
        # Leave the room empty; the student has to call in a new patient explicitly.
        return (
            Modal(visible=False),
            "",
            "",
            "",
            "",
            chatbot_update,
            exam_is_open,
            exam_text,
            exam_btn,
            studies_is_open,
            studies_text,
            studies_btn,
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(interactive=False, value=""),
            [],
            None,
            build_patient_card(None),
        )

    chat_placeholder = build_chat_placeholder(has_patient=False)

    with gr.Blocks(css=".contain { display: flex !important; flex-direction: column !important; }"
    "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
    "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
    "#chatbot .placeholder-content { display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; height: 100% !important; text-align: center !important; }"
    "#chatbot .placeholder { display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; height: 100% !important; text-align: center !important; margin: auto !important; }"
    "#chatbot .placeholder * { text-align: center !important; margin-left: auto !important; margin-right: auto !important; }"
    f"#chatbot .bubble-wrap, #chatbot .wrap {{ background-image: url('{CLINICAL_CHATBOT_BACKGROUND_URI}') !important; background-size: cover !important; background-position: center !important; background-repeat: no-repeat !important; }}"
    "#col { height: 100vh !important; }") as interface:
        turn_info_for_feedback = gr.State({
            "conversation_id": None,
            "turn_id": None,
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
        conversation_id = gr.State(None)
        gr.HTML("<h1 style='text-align: center;'>" + i18n("ClinicalChatbotTitle") + "</h1>")
        gr.Markdown(i18n("ClinicalChatbotDescription"))
        patient_id = gr.State(None)
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                patient_display = gr.Markdown(
                    value=build_patient_card(None),
                    sanitize_html=False,
                    elem_id="patient-card-display",
                )
            with gr.Column(scale=1):
                sample_patient_button = gr.Button(
                    i18n("ClinicalChatbotSamplePatientButton"),
                    variant="stop",
                    elem_id="sample-patient-button",
                )
        
        gr.HTML("<hr>")

        with gr.Row(visible=True, elem_id='col') as chat_col:
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    elem_id="chatbot",
                    show_copy_button=True,
                    placeholder=chat_placeholder,
                    # likeable=True,
                )
                chat_textbox = gr.Textbox(
                    show_label=False,
                    placeholder="",
                    scale=7,
                    autofocus=False,
                    interactive=False,
                    submit_btn=True,
                )
                chat_interface = gr.ChatInterface(
                        predict,
                        type="messages",
                        textbox=chat_textbox,
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
                            conversation_id,
                        ],
                        chatbot=chatbot,
                        submit_btn=True,
                        stop_btn=None,
                    )
            with gr.Column(scale=1, elem_id="complementary-studies-col"):
                gr.Markdown("## " + i18n("ClinicalChatbotPhysicalExamTitle"))
                physical_exam_is_open = gr.State(False)
                physical_exam_open_button = gr.Button(
                    i18n("ClinicalChatbotPhysicalExamOpenButton"),
                    interactive=False,
                )
                physical_exam_text = gr.Textbox(
                    show_label=False,
                    lines=10,
                    interactive=False,
                    visible=False,
                )

                gr.Markdown("## " + i18n("ClinicalChatbotComplementaryStudiesTitle"))
                complementary_studies_is_open = gr.State(False)
                complementary_studies_open_button = gr.Button(
                    i18n("ClinicalChatbotComplementaryStudiesOpenButton"),
                    interactive=False,
                )
                complementary_studies_text = gr.Textbox(
                    show_label=False,
                    lines=10,
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
            requested_studies = gr.Textbox(
                label=i18n("ClinicalChatbotRequestedStudiesLabel"),
                placeholder=i18n("ClinicalChatbotRequestedStudiesPlaceholder"),
                lines=2,
            )
            end_conversation_submit_button = gr.Button(
                i18n("ClinicalChatbotEndConversationSubmitButton"),
                variant="stop",
            )

        chatbot.like(
            open_turn_feedback_modal,
            [token_id, age, gender, nationality, region, school, patient_id, participant_area, contact_with_real_patients, conversation_id],
            [turn_info_for_feedback, turn_feedback_modal],
        )
        modal_submit_button.click(send_turn_feedback_modal, [turn_info_for_feedback, text_feedback], [turn_feedback_modal, text_feedback])
        physical_exam_open_button.click(
            toggle_physical_exam,
            [physical_exam_is_open, patient_id, conversation_id, token_id],
            [physical_exam_is_open, physical_exam_text, physical_exam_open_button],
        )
        complementary_studies_open_button.click(
            toggle_complementary_studies,
            [complementary_studies_is_open, patient_id, conversation_id, token_id],
            [complementary_studies_is_open, complementary_studies_text, complementary_studies_open_button],
        )
        patient_id.change(
            reset_on_patient_change,
            [patient_id],
            [
                physical_exam_is_open,
                physical_exam_text,
                physical_exam_open_button,
                complementary_studies_is_open,
                complementary_studies_text,
                complementary_studies_open_button,
                chatbot,
                sample_patient_button,
                end_conversation_button,
                chat_interface.textbox,
                chat_interface.chatbot_state,
                conversation_id,
                patient_display,
            ],
        )
        sample_patient_button.click(
            sample_new_patient,
            [token_id, participant_area, patient_id],
            [patient_id],
        )
        end_conversation_button.click(lambda: Modal(visible=True), None, end_conversation_modal)
        end_conversation_submit_button.click(
            send_end_conversation_form,
            [
                acute_problems,
                chronic_problems,
                diagnosis,
                requested_studies,
                token_id,
                age,
                gender,
                nationality,
                region,
                school,
                patient_id,
                participant_area,
                contact_with_real_patients,
                conversation_id,
                chat_interface.chatbot_state,
            ],
            [
                end_conversation_modal,
                acute_problems,
                chronic_problems,
                diagnosis,
                requested_studies,
                chatbot,
                physical_exam_is_open,
                physical_exam_text,
                physical_exam_open_button,
                complementary_studies_is_open,
                complementary_studies_text,
                complementary_studies_open_button,
                patient_id,
                sample_patient_button,
                end_conversation_button,
                chat_interface.textbox,
                chat_interface.chatbot_state,
                conversation_id,
                patient_display,
            ],
        )
    return interface
