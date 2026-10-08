"""Balances how often each clinical patient is assigned across workshop participants.

Derives history straight from the clinical chatbot conversation log (which already
records token_id + patient_id per message) instead of keeping a separate assignment
log, so counts always reflect real conversations and survive restarts for free.
"""
import json
import os
import random
from collections import defaultdict

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

ROLE_PATIENT_POOLS = {
    "ClinicalChatbotPsychologyStudent": PSYCHOLOGY_PATIENT_POOL,
    "ClinicalChatbotPsychologyInTraining": PSYCHOLOGY_PATIENT_POOL,
    "ClinicalChatbotMedicalStudent": MEDICAL_PATIENT_POOL,
    "ClinicalChatbotMedicalInTraining": MEDICAL_PATIENT_POOL,
    "ClinicalChatbotOther": MEDICAL_PATIENT_POOL,
}

def _patient_pool_for_role(participant_area):
    pool = ROLE_PATIENT_POOLS.get(participant_area)
    if not pool:
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
