import re
from dataclasses import dataclass
from typing import Optional

from .general_rules import general_rules
from . import (
    PEV_01,
    PEV_02,
    PEV_03,
    PEV_04,
    PEV_05,
    PEV_06,
    PEV_07,
    PEV_08,
    PEV_09,
    PEV_10,
    PEV_11,
    PEV_12,
    PEV_13,
)

PATIENT_NAME_PATTERN = re.compile(r"Nombre y Apellido:\s*(.+)")


@dataclass(frozen=True)
class ClinicalPatient:
    id: str
    prompt: str
    physical_exam: str = ""
    extra_studies: str = ""
    name: str = ""

    def __post_init__(self):
        if not self.name:
            match = PATIENT_NAME_PATTERN.search(self.prompt)
            name = match.group(1).strip().rstrip(".") if match else self.id
            object.__setattr__(self, "name", name)


PATIENT_MODULES = (
    PEV_01,
    PEV_02,
    PEV_03,
    PEV_04,
    PEV_05,
    PEV_06,
    PEV_07,
    PEV_08,
    PEV_09,
    PEV_10,
    PEV_11,
    PEV_12,
    PEV_13,
)


def _load_patient(module) -> ClinicalPatient:
    patient_id = getattr(
        module,
        "PATIENT_ID",
        getattr(module, "patient_id", module.__name__.split(".")[-1].replace("_", "-")),
    )
    raw_prompt = getattr(module, "patient_prompt", "")
    full_prompt = (general_rules + raw_prompt) if raw_prompt else ""
    return ClinicalPatient(
        id=patient_id,
        prompt=full_prompt,
        physical_exam=getattr(module, "physical_exam", ""),
        extra_studies=getattr(module, "extra_studies", ""),
    )


# Single source of truth for clinical patients
clinical_patients: dict[str, ClinicalPatient] = {
    patient.id: patient for patient in (_load_patient(mod) for mod in PATIENT_MODULES)
}


def get_patient(patient_id: str) -> Optional[ClinicalPatient]:
    """Retrieves a ClinicalPatient by ID, or None if not found."""
    return clinical_patients.get(patient_id)


# Backwards-compatible mappings for existing callers
clinical_prompts: dict[str, str] = {pid: p.prompt for pid, p in clinical_patients.items()}
physical_exams: dict[str, str] = {pid: p.physical_exam for pid, p in clinical_patients.items()}
clinical_extra_studies: dict[str, str] = {pid: p.extra_studies for pid, p in clinical_patients.items()}


