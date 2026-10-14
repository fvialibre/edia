"""Unit and stress-simulation tests for patient assignment module."""
import threading
import tempfile
import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.module_patientAssignment as pa
from modules.module_patientAssignment import (
    assign_patient,
    MEDICAL_PATIENT_POOL,
    PSYCHOLOGY_PATIENT_POOL,
)


@contextmanager
def isolated_test_env():
    """Redirects both encounters log and active assignments to isolated temporary files."""
    encounters_tmp = tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8")
    encounters_tmp.close()
    active_tmp = tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8")
    active_tmp.close()

    orig_encounters = pa.CLINICAL_ENCOUNTERS_LOG_PATH
    orig_active = pa.ACTIVE_ASSIGNMENTS_LOG_PATH

    pa.CLINICAL_ENCOUNTERS_LOG_PATH = encounters_tmp.name
    pa.ACTIVE_ASSIGNMENTS_LOG_PATH = active_tmp.name

    with pa._lock:
        pa._active_by_token.clear()

    try:
        yield encounters_tmp.name, active_tmp.name
    finally:
        pa.CLINICAL_ENCOUNTERS_LOG_PATH = orig_encounters
        pa.ACTIVE_ASSIGNMENTS_LOG_PATH = orig_active
        with pa._lock:
            pa._active_by_token.clear()
        if os.path.exists(encounters_tmp.name):
            os.remove(encounters_tmp.name)
        if os.path.exists(active_tmp.name):
            os.remove(active_tmp.name)


def _log_completion(token, patient_id):
    """Simulates the clinical interface appending a completed encounter to disk."""
    with open(pa.CLINICAL_ENCOUNTERS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"token_id": token, "patient_id": patient_id}) + "\n")


def test_pool_separation():
    """Verifies Medical and Psychology pools are disjoint and assigned strictly by role."""
    with isolated_test_env():
        assert set(MEDICAL_PATIENT_POOL).isdisjoint(set(PSYCHOLOGY_PATIENT_POOL))
        assert len(MEDICAL_PATIENT_POOL) == 10
        assert len(PSYCHOLOGY_PATIENT_POOL) == 3

        # Medical participants only receive medical pool cases
        for i in range(20):
            med_patient = assign_patient(f"med_user_{i}", participant_area="ClinicalChatbotMedicalStudent")
            assert med_patient in MEDICAL_PATIENT_POOL
            assert med_patient not in PSYCHOLOGY_PATIENT_POOL

        # Psychology participants only receive psychology pool cases
        for i in range(20):
            psych_patient = assign_patient(f"psych_user_{i}", participant_area="ClinicalChatbotPsychologyStudent")
            assert psych_patient in PSYCHOLOGY_PATIENT_POOL
            assert psych_patient not in MEDICAL_PATIENT_POOL


def test_strict_non_repetition_of_completed_patients():
    """A participant must never receive a patient they have completed."""
    with isolated_test_env():
        token = "participant_alpha"
        assigned_patients = []

        # Participant completes 10 medical patients one by one
        for _ in range(10):
            patient = assign_patient(token, participant_area="ClinicalChatbotMedicalStudent")
            assert patient not in assigned_patients
            assigned_patients.append(patient)
            _log_completion(token, patient)

        assert set(assigned_patients) == set(MEDICAL_PATIENT_POOL)


def test_no_immediate_reassignment_on_resample():
    """A participant resampling without completion must never get the patient they just had."""
    with isolated_test_env():
        token = "participant_resample"
        p1 = assign_patient(token, participant_area="ClinicalChatbotMedicalStudent")
        assert p1 in MEDICAL_PATIENT_POOL

        # Resample without completing: exclude_patient_id is p1
        p2 = assign_patient(token, participant_area="ClinicalChatbotMedicalStudent", exclude_patient_id=p1)
        assert p2 != p1
        assert p2 in MEDICAL_PATIENT_POOL


def test_100_user_concurrency_initial_spike():
    """Simulates 100 participants clicking 'Call next patient' simultaneously."""
    with isolated_test_env():
        num_medical = 70
        num_psych = 30
        results = {}
        errors = []

        def call_med(idx):
            try:
                token = f"med_student_{idx:03d}"
                pid = assign_patient(token, participant_area="ClinicalChatbotMedicalStudent")
                results[token] = pid
            except Exception as e:
                errors.append(e)

        def call_psych(idx):
            try:
                token = f"psych_student_{idx:03d}"
                pid = assign_patient(token, participant_area="ClinicalChatbotPsychologyStudent")
                results[token] = pid
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(num_medical):
            threads.append(threading.Thread(target=call_med, args=(i,)))
        for i in range(num_psych):
            threads.append(threading.Thread(target=call_psych, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Encountered errors: {errors}"
        assert len(results) == 100

        med_counts = defaultdict(int)
        psych_counts = defaultdict(int)

        for tok, pid in results.items():
            if "med" in tok:
                assert pid in MEDICAL_PATIENT_POOL
                med_counts[pid] += 1
            else:
                assert pid in PSYCHOLOGY_PATIENT_POOL
                psych_counts[pid] += 1

        # 70 medical students across 10 cases => exactly 7 each
        assert set(med_counts.keys()) == set(MEDICAL_PATIENT_POOL)
        for pid, count in med_counts.items():
            assert count == 7, f"Expected 7 for {pid}, got {count}"

        # 30 psychology students across 3 cases => exactly 10 each
        assert set(psych_counts.keys()) == set(PSYCHOLOGY_PATIENT_POOL)
        for pid, count in psych_counts.items():
            assert count == 10, f"Expected 10 for {pid}, got {count}"


def test_multi_round_simulation_balancing():
    """Simulates 50 participants completing 3 full consultations each under concurrency."""
    with isolated_test_env():
        num_participants = 50
        rounds = 3
        user_histories = defaultdict(list)
        errors = []

        def run_participant(idx):
            try:
                token = f"sim_user_{idx:02d}"
                role = "ClinicalChatbotMedicalStudent" if idx % 2 == 0 else "ClinicalChatbotPsychologyStudent"
                pool = MEDICAL_PATIENT_POOL if idx % 2 == 0 else PSYCHOLOGY_PATIENT_POOL

                for _ in range(rounds):
                    pid = assign_patient(token, participant_area=role)
                    assert pid in pool
                    assert pid not in user_histories[token]
                    user_histories[token].append(pid)
                    _log_completion(token, pid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_participant, args=(i,)) for i in range(num_participants)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Encountered errors: {errors}"

        counts, _ = pa._read_completed_history()

        # Medical: 25 participants * 3 rounds = 75 total across 10 cases => 7 or 8 per case
        med_counts = [counts[pid] for pid in MEDICAL_PATIENT_POOL]
        assert max(med_counts) - min(med_counts) <= 1, f"Medical counts not balanced: {med_counts}"

        # Psychology: 25 participants * 3 rounds = 75 total across 3 cases => exactly 25 per case
        psych_counts = [counts[pid] for pid in PSYCHOLOGY_PATIENT_POOL]
        assert max(psych_counts) - min(psych_counts) <= 1, f"Psychology counts not balanced: {psych_counts}"


def test_crash_recovery_from_log_file():
    """Simulates restarting the application and verifying state restores from JSONL log."""
    with isolated_test_env() as (enc_path, _):
        with open(enc_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"token_id": "user1", "patient_id": "PEV-01"}) + "\n")
            f.write(json.dumps({"token_id": "user1", "patient_id": "PEV-02"}) + "\n")
            f.write(json.dumps({"token_id": "user2", "patient_id": "PEV-11"}) + "\n")
            f.write("\n")  # Blank line
            f.write("{invalid_json_line\n")  # Corrupted line

        counts, by_token = pa._read_completed_history()
        assert counts["PEV-01"] == 1
        assert counts["PEV-02"] == 1
        assert counts["PEV-11"] == 1
        assert "PEV-01" in by_token["user1"]
        assert "PEV-02" in by_token["user1"]
        assert "PEV-11" in by_token["user2"]

        # User1 should NOT get PEV-01 or PEV-02
        next_p = assign_patient("user1", participant_area="ClinicalChatbotMedicalStudent")
        assert next_p not in {"PEV-01", "PEV-02"}


def test_clear_active_assignments():
    """Verifies that clear_active_assignments resets in-memory dict and writes valid empty JSON to disk."""
    with isolated_test_env() as (_, active_path):
        assign_patient("user_active", participant_area="ClinicalChatbotMedicalStudent")
        with open(active_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert "user_active" in data

        pa.clear_active_assignments()
        with open(active_path, "r", encoding="utf-8") as f:
            cleared_data = json.load(f)
        assert cleared_data == {}
        assert len(pa._active_by_token) == 0


if __name__ == "__main__":
    print("Running test_pool_separation...")
    test_pool_separation()
    print("Running test_strict_non_repetition_of_completed_patients...")
    test_strict_non_repetition_of_completed_patients()
    print("Running test_no_immediate_reassignment_on_resample...")
    test_no_immediate_reassignment_on_resample()
    print("Running test_100_user_concurrency_initial_spike...")
    test_100_user_concurrency_initial_spike()
    print("Running test_multi_round_simulation_balancing...")
    test_multi_round_simulation_balancing()
    print("Running test_crash_recovery_from_log_file...")
    test_crash_recovery_from_log_file()
    print("Running test_clear_active_assignments...")
    test_clear_active_assignments()
    print("\nALL 7 TESTS PASSED SUCCESSFULLY!")
