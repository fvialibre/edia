"""Balances how often each clinical patient is assigned across workshop participants.

Derives history straight from the clinical chatbot conversation log (which already
records token_id + patient_id per message) instead of keeping a separate assignment
log, so counts always reflect real conversations and survive restarts for free.
"""
import json
import os
import random
from collections import defaultdict
import yaml

from data.clinical_patients.clinical_prompts import clinical_prompts

CLINICAL_CHATBOT_LOG_PATH = "./logs/logs_clinical_chatbot.jsonl"

# Psychology participants only practice with the cardiovascular/respiratory cases.
PSYCHOLOGY_PATIENT_POOL = (
    "PEV-BETA-DM-01",
    "PEV-BETA-GERD-02",
    "PEV-BETA-IC-03",
    "PEV-BETA-RENAL-04",
    "PEV-BETA-REUMA-05",
    "PEV-BETA-MET-06",
    "PEV-BETA-HEM-07",
    "PEV-BETA-HEM-08",
    "PEV-BETA-CV-09",
    "PEV-BETA-RESP-10",
)

# Medical participants practice with patients 01 through 07.
MEDICAL_PATIENT_POOL = (
    "PEV-BETA-DM-01",
    "PEV-BETA-GERD-02",
    "PEV-BETA-IC-03",
    "PEV-BETA-RENAL-04",
    "PEV-BETA-REUMA-05",
    "PEV-BETA-MET-06",
    "PEV-BETA-HEM-07",
    "PEV-BETA-HEM-08",
    "PEV-BETA-CV-09",
    "PEV-BETA-RESP-10",
)

PSYCHOLOGY_AREAS = {
    "ClinicalChatbotPsychologyStudent",
    "ClinicalChatbotPsychologyInTraining",
}

MEDICAL_AREAS = {
    "ClinicalChatbotMedicalStudent",
    "ClinicalChatbotMedicalInTraining",
    "ClinicalChatbotOther",
}

def _load_i18n_area_labels():
    try:
        with open("language/i18n.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for lang_dict in data.values():
            if isinstance(lang_dict, dict):
                for key in list(PSYCHOLOGY_AREAS):
                    if key in lang_dict:
                        PSYCHOLOGY_AREAS.add(lang_dict[key])
                for key in list(MEDICAL_AREAS):
                    if key in lang_dict:
                        MEDICAL_AREAS.add(lang_dict[key])
    except Exception:
        pass

_load_i18n_area_labels()

def is_psychology_participant(participant_area):
    if not participant_area:
        return False
    return str(participant_area).strip() in PSYCHOLOGY_AREAS

ROLE_PATIENT_POOLS = {
    **{area: PSYCHOLOGY_PATIENT_POOL for area in PSYCHOLOGY_AREAS},
    **{area: MEDICAL_PATIENT_POOL for area in MEDICAL_AREAS},
}

def _patient_pool_for_role(participant_area):
    if is_psychology_participant(participant_area):
        pool = PSYCHOLOGY_PATIENT_POOL
    elif participant_area and str(participant_area).strip() in MEDICAL_AREAS:
        pool = MEDICAL_PATIENT_POOL
    else:
        return list(clinical_prompts)
    return [patient_id for patient_id in pool if patient_id in clinical_prompts]


def _read_patient_history():
    """Returns (patients_seen_by_token, conversation_counts) built from past conversations."""
    patients_seen_by_token = defaultdict(set)
    conversation_counts = {patient_id: 0 for patient_id in clinical_prompts}

    if not os.path.exists(CLINICAL_CHATBOT_LOG_PATH):
        return patients_seen_by_token, conversation_counts

    with open(CLINICAL_CHATBOT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            token_id = entry.get("token_id")
            patient_id = entry.get("patient_id")
            if patient_id not in clinical_prompts:
                continue
            if patient_id not in patients_seen_by_token[token_id]:
                conversation_counts[patient_id] += 1
            patients_seen_by_token[token_id].add(patient_id)

    return patients_seen_by_token, conversation_counts


def assign_patient(token_id, participant_area=None, exclude_patient_id=None) -> str:
    """Picks a patient this token hasn't talked to yet (within their role's pool).

    Candidates are weighted inversely to how often they've been seen so far, so
    balancing is favored without ever fully starving out a patient that's a bit behind.
    """
    patients_seen_by_token, conversation_counts = _read_patient_history()
    already_seen = patients_seen_by_token.get(token_id, set())
    pool = _patient_pool_for_role(participant_area)

    candidates = [patient_id for patient_id in pool if patient_id not in already_seen]
    if not candidates:
        # This token has talked to every patient in their pool already, start over from the pool.
        candidates = pool

    # Never hand back the patient they already have, otherwise resampling can look like it's stuck.
    if exclude_patient_id and len(candidates) > 1:
        candidates = [patient_id for patient_id in candidates if patient_id != exclude_patient_id] or candidates

    weights = [1 / (conversation_counts[patient_id] + 1) for patient_id in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]
