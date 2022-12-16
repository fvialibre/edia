from modules.module_customPllLabel import CustomPllLabel
from modules.module_pllScore import PllScore
from typing import Dict, List

class CrowsPairs:
    def __init__(
        self, 
        language_model # LanguageModel class instance
    ) -> None:

        self.Label = CustomPllLabel()
        self.pllScore = PllScore(
            language_model=language_model
        )

    def errorChecking(
        self, 
        sent_list: List[str],
    ) -> str:

        out_msj = ""

        mandatory_sents = [0,1]
        for sent_id, sent in enumerate(sent_list):
            c_sent = sent.strip()
            if c_sent:
                if not self.pllScore.sentIsCorrect(c_sent):
                    out_msj = f"Error: The sentence Nº {sent_id+1} does not have the correct format!."
                    break
            else:
                if sent_id in mandatory_sents:
                    out_msj = f"Error: The sentence Nº{sent_id+1} can not be empty!"
                    break
        
        return out_msj

    def rank(
        self, 
        sent_list: List[str],
    ) -> Dict[str, float]:

        err = self.errorChecking(sent_list)
        if err:
            raise Exception(err)
        
        all_plls_scores = {}
        for sent in sent_list:
            if sent:
                all_plls_scores[sent] = self.pllScore.compute(sent)

        return all_plls_scores