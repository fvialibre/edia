import json
import os
import random
import threading
from collections import defaultdict
import yaml

from data.clinical_patients.clinical_prompts import clinical_prompts

CLINICAL_ENCOUNTERS_LOG_PATH = "./logs/clinical_encounters.jsonl"
ACTIVE_ASSIGNMENTS_LOG_PATH = "./logs/active_assignments.json"

MEDICAL_PATIENT_POOL = (
    "PEV-01",
    "PEV-02",
    "PEV-03",
    "PEV-04",
    "PEV-05",
    "PEV-06",
    "PEV-07",
    "PEV-08",
    "PEV-09",
    "PEV-10",
)

PSYCHOLOGY_PATIENT_POOL = (
    "PEV-11",
    "PEV-12",
    "PEV-13",
)

PSYCHOLOGY_AREAS = {
    "ClinicalChatbotPsychologyStudent",
    "ClinicalChatbotPsychologyInTraining",
}

try:
    with open("language/i18n.yaml", "r", encoding="utf-8") as f:
        for lang_dict in (yaml.safe_load(f) or {}).values():
            if isinstance(lang_dict, dict):
                PSYCHOLOGY_AREAS.update(lang_dict[k] for k in list(PSYCHOLOGY_AREAS) if k in lang_dict)
except Exception:
    pass


def is_psychology_participant(area):
    if not area:
        return False
    clean = str(area).strip().lower()
    if clean in {a.lower() for a in PSYCHOLOGY_AREAS}:
        return True
    return "psic" in clean or "psych" in clean


_lock = threading.Lock()
_active_by_token = {}


def _save_active():
    try:
        with open(ACTIVE_ASSIGNMENTS_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(_active_by_token, f)
    except Exception:
        pass


def _read_completed_history():
    counts = {pid: 0 for pid in clinical_prompts}
    by_token = defaultdict(set)
    if os.path.exists(CLINICAL_ENCOUNTERS_LOG_PATH):
        with open(CLINICAL_ENCOUNTERS_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    tok = str(entry.get("token_id", "")).strip().lower()
                    pid = entry.get("patient_id")
                    if pid in counts:
                        counts[pid] += 1
                        if tok:
                            by_token[tok].add(pid)
                except json.JSONDecodeError:
                    continue
    return counts, by_token


def assign_patient(token_id, participant_area=None, exclude_patient_id=None) -> str:
    token = str(token_id).strip().lower() if token_id else ""
    raw_pool = PSYCHOLOGY_PATIENT_POOL if is_psychology_participant(participant_area) else MEDICAL_PATIENT_POOL
    pool = [pid for pid in raw_pool if pid in clinical_prompts]

    with _lock:
        completed_counts, completed_by_token = _read_completed_history()

        # Clean up any active tokens that have now completed that case
        for t, pid in list(_active_by_token.items()):
            if pid in completed_by_token.get(t, set()):
                _active_by_token.pop(t, None)

        # 1. Available candidates: pool cases not yet completed by this user
        candidates = [pid for pid in pool if pid not in completed_by_token.get(token, set())] or list(pool)

        # 2. Exclude current or just-assigned case
        to_exclude = {exclude_patient_id, _active_by_token.get(token)} - {None}
        if len(candidates) > 1:
            candidates = [pid for pid in candidates if pid not in to_exclude] or candidates

        # 3. Workload = completed on disk + currently active in another session
        active_counts = defaultdict(int)
        for t, pid in _active_by_token.items():
            if t != token:
                active_counts[pid] += 1

        def load(pid):
            return completed_counts[pid] + active_counts[pid]

        # 4. Pick candidate with minimum workload (ties broken uniformly at random)
        min_load = min(load(pid) for pid in candidates)
        chosen = random.choice([pid for pid in candidates if load(pid) == min_load])

        # 5. Record assignment
        if token:
            _active_by_token[token] = chosen
            _save_active()

        return chosen
