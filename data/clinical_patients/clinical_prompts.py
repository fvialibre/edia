from .general_rules import general_rules
from .jose import patient_prompt as jose_prompt
from .victoria import patient_prompt as victoria_prompt
from .javier import patient_prompt as javier_prompt
from .PEV_BETA_DM_01 import patient_prompt as PEV_BETA_DM_01_prompt, extra_studies as PEV_BETA_DM_01_extra_studies
from .PEV_BETA_GERD_02 import patient_prompt as PEV_BETA_GERD_02_prompt, extra_studies as PEV_BETA_GERD_02_extra_studies
from .PEV_BETA_IC_03 import patient_prompt as PEV_BETA_IC_03_prompt, extra_studies as PEV_BETA_IC_03_extra_studies
from .PEV_BETA_RENAL_04 import patient_prompt as PEV_BETA_RENAL_04_prompt, extra_studies as PEV_BETA_RENAL_04_extra_studies
from .PEV_BETA_REUMA_05 import patient_prompt as PEV_BETA_REUMA_05_prompt, extra_studies as PEV_BETA_REUMA_05_extra_studies
from .PEV_BETA_MET_06 import patient_prompt as PEV_BETA_MET_06_prompt, extra_studies as PEV_BETA_MET_06_extra_studies
from .PEV_BETA_HEM_07 import patient_prompt as PEV_BETA_HEM_07_prompt, extra_studies as PEV_BETA_HEM_07_extra_studies
from .PEV_BETA_HEM_08 import patient_prompt as PEV_BETA_HEM_08_prompt, extra_studies as PEV_BETA_HEM_08_extra_studies
from .PEV_BETA_CV_09 import patient_prompt as PEV_BETA_CV_09_prompt, extra_studies as PEV_BETA_CV_09_extra_studies
from .PEV_BETA_RESP_10 import patient_prompt as PEV_BETA_RESP_10_prompt, extra_studies as PEV_BETA_RESP_10_extra_studies

clinical_prompts = {
    # "Paciente A": general_rules + jose_prompt,
    # "Paciente B": general_rules + javier_prompt,
    # "Paciente C": general_rules + victoria_prompt,
    "PEV-BETA-DM-01": general_rules + PEV_BETA_DM_01_prompt,
    "PEV-BETA-GERD-02": general_rules + PEV_BETA_GERD_02_prompt,
    "PEV-BETA-IC-03": general_rules + PEV_BETA_IC_03_prompt,
    "PEV-BETA-RENAL-04": general_rules + PEV_BETA_RENAL_04_prompt,
    "PEV-BETA-REUMA-05": general_rules + PEV_BETA_REUMA_05_prompt,
    "PEV-BETA-MET-06": general_rules + PEV_BETA_MET_06_prompt,
    "PEV-BETA-HEM-07": general_rules + PEV_BETA_HEM_07_prompt,
    "PEV-BETA-HEM-08": general_rules + PEV_BETA_HEM_08_prompt,
    "PEV-BETA-CV-09": general_rules + PEV_BETA_CV_09_prompt,
    "PEV-BETA-RESP-10": general_rules + PEV_BETA_RESP_10_prompt,
}

clinical_extra_studies = {
    "PEV-BETA-DM-01": PEV_BETA_DM_01_extra_studies,
    "PEV-BETA-GERD-02": PEV_BETA_GERD_02_extra_studies,
    "PEV-BETA-IC-03": PEV_BETA_IC_03_extra_studies,
    "PEV-BETA-RENAL-04": PEV_BETA_RENAL_04_extra_studies,
    "PEV-BETA-REUMA-05": PEV_BETA_REUMA_05_extra_studies,
    "PEV-BETA-MET-06": PEV_BETA_MET_06_extra_studies,
    "PEV-BETA-HEM-07": PEV_BETA_HEM_07_extra_studies,
    "PEV-BETA-HEM-08": PEV_BETA_HEM_08_extra_studies,
    "PEV-BETA-CV-09": PEV_BETA_CV_09_extra_studies,
    "PEV-BETA-RESP-10": PEV_BETA_RESP_10_extra_studies,
}

