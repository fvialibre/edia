import uuid
import gradio as gr
from gradio_modal import Modal
from gradio_i18n import gettext as i18n
from gradio_i18n.i18n import TranslateContext, LocalContext
import yaml
import pandas as pd
from modules.module_ollama import ModelWrapper
import json
from datetime import datetime
import os
from dotenv import dotenv_values
from data.clinical_patients.clinical_prompts import clinical_patients
from html_constants import HTML_FEEDBACK_TITLE
from modules.module_patientAssignment import assign_patient, is_psychology_participant

# Empty consulting room, shown behind the chat before a patient is picked or after a conversation ends.
CLINICAL_CHATBOT_BACKGROUND_URI = "https://i.imgur.com/NQfqr2f.jpeg"

# Shown once a patient is assigned but the conversation hasn't started yet.
CLINICAL_CHATBOT_PATIENT_WAITING_URI = "https://i.imgur.com/JDSFVDs.jpeg"

_I18N_FALLBACK = None

def get_i18n_text(key, default_lang="es"):
    """Fetches text string from gradio_i18n dictionary so it serializes properly in dynamic callbacks."""
    global _I18N_FALLBACK
    lang = default_lang
    try:
        req = LocalContext.request.get(None)
        if req:
            lang = TranslateContext.get_current_language(req)
        tr = TranslateContext.dictionary.get(lang, {}).get(key)
        if tr:
            return tr
        tr = TranslateContext.dictionary.get(default_lang, {}).get(key)
        if tr:
            return tr
    except Exception:
        pass

    if _I18N_FALLBACK is None:
        try:
            with open("language/i18n.yaml", "r", encoding="utf-8") as f:
                _I18N_FALLBACK = yaml.safe_load(f)
        except Exception:
            _I18N_FALLBACK = {}

    return _I18N_FALLBACK.get(lang, {}).get(key, _I18N_FALLBACK.get(default_lang, {}).get(key, key))

def get_patient_name(patient_id):
    """Pulls the patient's display name, falling back to the ID."""
    patient = clinical_patients.get(patient_id)
    return patient.name if patient else patient_id

def build_patient_card(patient_id):
    """Builds a styled Markdown card displaying the active patient's name or waiting room status."""
    patient_label = get_i18n_text("ClinicalChatbotPatientLabel")
    if not patient_id:
        no_patient_text = get_i18n_text("ClinicalChatbotNoPatientAssigned")
        return (
            '<div style="display: flex; align-items: center; gap: 14px; padding: 12px 18px; '
            'border-radius: 10px; border: 1.5px dashed rgba(156, 163, 175, 0.5); '
            'background: rgba(156, 163, 175, 0.07); min-height: 52px; box-sizing: border-box;">'
            '<span style="font-size: 1.7em; line-height: 1;">🚪</span>'
            '<div>'
            f'<div style="font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7;">{patient_label}</div>'
            f'<div style="font-size: 1.05em; font-weight: 500; opacity: 0.65; font-style: italic;">{no_patient_text}</div>'
            '</div>'
            '</div>'
        )
    patient_name = get_patient_name(patient_id)
    in_consultation_text = get_i18n_text("ClinicalChatbotPatientInConsultation")
    in_consultation_status = get_i18n_text("ClinicalChatbotPatientInConsultationStatus")
    return (
        '<div style="display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 12px 18px; '
        'border-radius: 10px; border: 1.5px solid rgba(59, 130, 246, 0.55); '
        'background: rgba(59, 130, 246, 0.08); min-height: 52px; box-sizing: border-box;">'
        '<div style="display: flex; align-items: center; gap: 14px;">'
        '<span style="font-size: 1.7em; line-height: 1;">🧑‍⚕️</span>'
        '<div>'
        f'<div style="font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #3b82f6;">{in_consultation_text}</div>'
        f'<div style="font-size: 1.3em; font-weight: 700; letter-spacing: -0.01em;">{patient_name}</div>'
        '</div>'
        '</div>'
        '<div style="display: flex; align-items: center; gap: 6px; background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.35); color: #15803d; padding: 6px 12px; border-radius: 20px; font-size: 0.82em; font-weight: 600;">'
        '<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #22c55e; box-shadow: 0 0 6px #22c55e;"></span>'
        f'<span>{in_consultation_status}</span>'
        '</div>'
        '</div>'
    )

def build_chat_placeholder(has_patient, patient_id=None):
    """Builds the Chatbot placeholder HTML, swapping the image/text once a patient is assigned."""
    image_uri = CLINICAL_CHATBOT_PATIENT_WAITING_URI if has_patient else CLINICAL_CHATBOT_BACKGROUND_URI
    text_key = "ClinicalChatbotChatPlaceholderPatientReady" if has_patient else "ClinicalChatbotChatPlaceholder"
    text_val = get_i18n_text(text_key)
    if has_patient and patient_id:
        patient_name = get_patient_name(patient_id)
        try:
            text_val = text_val.format(patient=patient_name)
        except Exception:
            pass
    return (
        '<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; width: 100%; margin: 0 auto; padding: 1.5em 1em;">'
        f'<img src="{image_uri}" '
        'style="max-width: min(100%, 480px); height: auto; border-radius: 12px; margin: 0 auto 1.25em auto; display: block; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);" />'
        f'<div style="text-align: center; max-width: 480px; margin: 0 auto;">{text_val}</div>'
        '</div>'
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
    contact_with_real_patients,
    **kwargs,
) -> gr.Blocks:
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
        gr.Info(get_i18n_text("ChatbotFeedbackSentInfo"))
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
        chatbot_update = gr.Chatbot(value=[], type="messages", placeholder=build_chat_placeholder(has_patient, patient_id=patient_id))
        if has_patient:
            patient_name = get_patient_name(patient_id)
            input_placeholder = get_i18n_text("ClinicalChatbotInputPlaceholderActive").format(patient=patient_name)
        else:
            input_placeholder = get_i18n_text("ClinicalChatbotInputPlaceholderNoPatient")
        textbox_update = gr.update(
            interactive=has_patient,
            value="",
            placeholder=input_placeholder,
            autofocus=has_patient,
        )
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

    def validate_end_conversation_fields(
        area,
        acute,
        chronic,
        diag,
        studies,
        psych_reason,
        psych_psychosocial,
        psych_diagnosis,
        psych_intervention,
    ):
        if is_psychology_participant(area):
            is_ready = bool(
                psych_reason and str(psych_reason).strip()
                and psych_psychosocial and str(psych_psychosocial).strip()
                and psych_diagnosis and str(psych_diagnosis).strip()
                and psych_intervention and str(psych_intervention).strip()
            )
        else:
            is_ready = bool(
                acute and str(acute).strip()
                and chronic and str(chronic).strip()
                and diag and str(diag).strip()
                and studies and str(studies).strip()
            )
        return gr.update(interactive=is_ready)

    def sample_new_patient(token_id, participant_area, patient_id):
        next_patient_id = assign_patient(token_id, participant_area, exclude_patient_id=patient_id)
        patient_name = get_patient_name(next_patient_id)
        gr.Info(get_i18n_text("ClinicalChatbotNewPatientToast").format(patient=patient_name))
        return next_patient_id

    def send_end_conversation_form(
        acute_problems,
        chronic_problems,
        diagnosis,
        requested_studies,
        psych_reason,
        psych_psychosocial,
        psych_diagnosis,
        psych_intervention,
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

        is_psych = is_psychology_participant(participant_area)
        if is_psych:
            eval_data = {
                "consultation_reason": psych_reason,
                "psychosocial_factors": psych_psychosocial,
                "diagnostic_hypothesis": psych_diagnosis,
                "therapeutic_plan": psych_intervention,
            }
        else:
            eval_data = {
                "acute_problems": acute_problems,
                "chronic_problems": chronic_problems,
                "diagnosis": diagnosis,
                "requested_studies": requested_studies,
            }

        with open("./logs/logs_clinical_chatbot_end_conversation.jsonl", "a+", encoding='utf-8') as f:
            log_entry = {
                "timestamp": timestamp,
                "conversation_id": conv_id,
                "token_id": token_id,
                "age": age,
                "gender": gender,
                "nationality": nationality,
                "region": region,
                "school": school,
                "patient_id": patient_id,
                "participant_area": participant_area,
                "contact_with_real_patients": contact_with_real_patients,
                "acute_problems": acute_problems if not is_psych else None,
                "chronic_problems": chronic_problems if not is_psych else None,
                "diagnosis": diagnosis if not is_psych else None,
                "requested_studies": requested_studies if not is_psych else None,
                "consultation_reason": psych_reason if is_psych else None,
                "psychosocial_factors": psych_psychosocial if is_psych else None,
                "diagnostic_hypothesis": psych_diagnosis if is_psych else None,
                "therapeutic_plan": psych_intervention if is_psych else None,
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

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
                "evaluation": eval_data,
            }, ensure_ascii=False) + "\n")

        gr.Info(get_i18n_text("ChatbotFeedbackSentInfo"))
        exam_is_open, exam_text, exam_btn = reset_physical_exam(has_patient=False)
        studies_is_open, studies_text, studies_btn = reset_complementary_studies(has_patient=False)
        chatbot_update = gr.Chatbot(value=[], type="messages", placeholder=build_chat_placeholder(has_patient=False))
        no_patient_placeholder = get_i18n_text("ClinicalChatbotInputPlaceholderNoPatient")
        # Leave the room empty; the student has to call in a new patient explicitly.
        return (
            Modal(visible=False),
            "",
            "",
            "",
            "",
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
            gr.update(interactive=False, value="", placeholder=no_patient_placeholder),
            [],
            None,
            build_patient_card(None),
            gr.update(interactive=False),
        )

    chat_placeholder = build_chat_placeholder(has_patient=False)

    custom_css = (
        ".contain { display: flex !important; flex-direction: column !important; }\n"
        "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }\n"
        "#chatbot { flex-grow: 1 !important; overflow: auto !important;}\n"
        "#chatbot .placeholder-content { display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; height: 100% !important; text-align: center !important; }\n"
        "#chatbot .placeholder { display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; height: 100% !important; text-align: center !important; margin: auto !important; }\n"
        "#chatbot .placeholder * { text-align: center !important; margin-left: auto !important; margin-right: auto !important; }\n"
        f"#chatbot .bubble-wrap, #chatbot .wrap {{ background-image: url('{CLINICAL_CHATBOT_BACKGROUND_URI}') !important; background-size: cover !important; background-position: center !important; background-repeat: no-repeat !important; }}\n"
        "#col { height: 100vh !important; }\n"
        "@keyframes clinical-bounce-arrow {\n"
        "  0%, 100% { transform: translateY(0); }\n"
        "  50% { transform: translateY(4px); }\n"
        "}\n"
        ".clinical-bounce-arrow {\n"
        "  animation: clinical-bounce-arrow 1.3s infinite ease-in-out;\n"
        "}\n"
        "@keyframes pulse-chat-input {\n"
        "  0% {\n"
        "    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.55);\n"
        "    border-color: rgba(59, 130, 246, 0.7);\n"
        "  }\n"
        "  50% {\n"
        "    box-shadow: 0 0 0 6px rgba(59, 130, 246, 0.18);\n"
        "    border-color: #3b82f6;\n"
        "  }\n"
        "  100% {\n"
        "    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);\n"
        "    border-color: rgba(59, 130, 246, 0.7);\n"
        "  }\n"
        "}\n"
        "#clinical-chat-textbox textarea:not(:disabled), #clinical-chat-textbox input:not(:disabled) {\n"
        "  animation: pulse-chat-input 2.4s infinite ease-in-out;\n"
        "  border: 1.5px solid #3b82f6 !important;\n"
        "}\n"
        "#clinical-chat-textbox textarea:not(:disabled):focus, #clinical-chat-textbox input:not(:disabled):focus {\n"
        "  animation: none !important;\n"
        "  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.35) !important;\n"
        "}\n"
    )

    with gr.Blocks(css=custom_css) as interface:
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
                    placeholder=get_i18n_text("ClinicalChatbotInputPlaceholderNoPatient"),
                    scale=7,
                    autofocus=False,
                    interactive=False,
                    submit_btn=True,
                    elem_id="clinical-chat-textbox",
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
            with gr.Group(visible=True) as medical_form_group:
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
            with gr.Group(visible=False) as psychology_form_group:
                psych_reason = gr.Textbox(
                    label=i18n("ClinicalChatbotPsychologyConsultationReasonLabel"),
                    placeholder=i18n("ClinicalChatbotPsychologyConsultationReasonPlaceholder"),
                    lines=2,
                )
                psych_psychosocial = gr.Textbox(
                    label=i18n("ClinicalChatbotPsychologyPsychosocialFactorsLabel"),
                    placeholder=i18n("ClinicalChatbotPsychologyPsychosocialFactorsPlaceholder"),
                    lines=2,
                )
                psych_diagnosis = gr.Textbox(
                    label=i18n("ClinicalChatbotPsychologyDiagnosticHypothesisLabel"),
                    placeholder=i18n("ClinicalChatbotPsychologyDiagnosticHypothesisPlaceholder"),
                    lines=2,
                )
                psych_intervention = gr.Textbox(
                    label=i18n("ClinicalChatbotPsychologyTherapeuticPlanLabel"),
                    placeholder=i18n("ClinicalChatbotPsychologyTherapeuticPlanPlaceholder"),
                    lines=2,
                )
            end_conversation_submit_button = gr.Button(
                i18n("ClinicalChatbotEndConversationSubmitButton"),
                variant="stop",
                interactive=False,
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
            js="""() => {
                setTimeout(() => {
                    const el = document.querySelector('#clinical-chat-textbox textarea, #clinical-chat-textbox input');
                    if (el && !el.disabled) {
                        el.focus();
                        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                }, 200);
            }""",
        )
        sample_patient_button.click(
            sample_new_patient,
            [token_id, participant_area, patient_id],
            [patient_id],
        )

        end_conversation_fields = [
            participant_area,
            acute_problems,
            chronic_problems,
            diagnosis,
            requested_studies,
            psych_reason,
            psych_psychosocial,
            psych_diagnosis,
            psych_intervention,
        ]

        def open_end_conversation_modal(area, acute, chronic, diag, studies, pr, pps, pd, pint):
            is_psych = is_psychology_participant(area)
            return (
                Modal(visible=True),
                gr.update(visible=not is_psych),
                gr.update(visible=is_psych),
                validate_end_conversation_fields(area, acute, chronic, diag, studies, pr, pps, pd, pint),
            )

        end_conversation_button.click(
            open_end_conversation_modal,
            inputs=end_conversation_fields,
            outputs=[
                end_conversation_modal,
                medical_form_group,
                psychology_form_group,
                end_conversation_submit_button,
            ],
        )

        def update_form_visibility_and_validation(area, acute, chronic, diag, studies, pr, pps, pd, pint):
            is_psych = is_psychology_participant(area)
            return (
                gr.update(visible=not is_psych),
                gr.update(visible=is_psych),
                validate_end_conversation_fields(area, acute, chronic, diag, studies, pr, pps, pd, pint),
            )

        if hasattr(participant_area, "change"):
            participant_area.change(
                update_form_visibility_and_validation,
                inputs=end_conversation_fields,
                outputs=[medical_form_group, psychology_form_group, end_conversation_submit_button],
            )

        end_conversation_submit_button.click(
            send_end_conversation_form,
            [
                acute_problems,
                chronic_problems,
                diagnosis,
                requested_studies,
                psych_reason,
                psych_psychosocial,
                psych_diagnosis,
                psych_intervention,
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
                psych_reason,
                psych_psychosocial,
                psych_diagnosis,
                psych_intervention,
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
                end_conversation_submit_button,
            ],
        )
        for field in [
            acute_problems,
            chronic_problems,
            diagnosis,
            requested_studies,
            psych_reason,
            psych_psychosocial,
            psych_diagnosis,
            psych_intervention,
        ]:
            field.change(
                validate_end_conversation_fields,
                inputs=end_conversation_fields,
                outputs=[end_conversation_submit_button],
            )
    return interface
