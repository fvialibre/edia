from modules.module_customPllLabel import CustomPllLabel
from modules.module_pllScore import PllScore


class CrowsPairs:
    def __init__(self, language_model):
        self.Label = CustomPllLabel()
        self.pllScore = PllScore(language_model=language_model)

    def errorChecking(self, sent0, sent1, sent2, sent3, sent4, sent5):
        out_msj = ""
        all_sents = [sent0, sent1, sent2, sent3, sent4, sent5]

        mandatory_sents = [0,1]
        for sent_id, sent in enumerate(all_sents):
            c_sent = sent.strip()
            if c_sent:
                if not self.pllScore.sentIsCorrect(c_sent):
                    out_msj = f"Error: La frase Nº {sent_id+1} no posee el formato correcto!."
                    break
            else:
                if sent_id in mandatory_sents:
                    out_msj = f"Error: La farse Nº{sent_id+1} no puede estar vacia!"
                    break
        
        return out_msj

    def rank(self, sent0, sent1, sent2, sent3, sent4, sent5):

        err = self.errorChecking(sent0, sent1, sent2, sent3, sent4, sent5)
        if err:
            raise Exception(err)
        
        all_sents = [sent0, sent1, sent2, sent3, sent4, sent5]
        all_plls_scores = {}
        for sent in all_sents:
            if sent:
                all_plls_scores[sent] = self.pllScore.compute(sent)

        return all_plls_scores