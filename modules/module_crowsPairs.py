from modules.module_customPllLabel import CustomPllLabel
from modules.module_pllScore import PllScore
from typing import Dict, List, Callable

class CrowsPairs:
    def __init__(
        self, 
        language_model,                         # LanguageModel class instance
        errorManager,                           # ErrorManager class instance
        rank_func: Callable[[str], float]=None
    ) -> None:

        self.Label = CustomPllLabel()
        self.pllScore = PllScore(
            language_model=language_model
        )
        self.errorManager = errorManager
        self.rank_func = rank_func

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
                    out_msj = ['CROWS-PAIRS_BAD_FORMATTED_SENTENCE', sent_id+1]
                    break
            else:
                if sent_id in mandatory_sents:
                    out_msj = ['CROWS-PAIRS_MANDATORY_SENTENCE_MISSING', sent_id+1]
                    break
        
        return self.errorManager.process(out_msj)

    def rank(
        self, 
        sent_list: List[str],
    ) -> Dict[str, float]:

        err = self.errorChecking(sent_list)
        if err:
            raise ValueError(err)
        
        all_scores = {}
        if self.rank_func is None:
            for sent in sent_list:
                if sent:
                    all_scores[sent] = self.pllScore.compute(sent)
        else:
            for sent in sent_list:
                if sent:
                    all_scores[sent] = self.rank_func(sent)

        return all_scores
        
        