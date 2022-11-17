from difflib import Differ
import torch, re

class PllScore:
    def __init__(self, language_model):
        self.tokenizer = language_model.initTokenizer()
        self.model = language_model.initModel()
        _ = self.model.eval()

        self.logSoftmax = torch.nn.LogSoftmax(dim=-1)

    def sentIsCorrect(self, sent):
        is_correct = True

        # Count number of interest word
        ciw = sent.count("'")
        if (ciw > 0  and ciw % 2 != 0) or (ciw == 0):
            is_correct = False
        
        if is_correct:
            # Check empty interest words
            interest_w = [w.replace("'","").strip() for w in re.findall("\'.*?\'", sent)]
            for word in interest_w:
                if not word:
                    is_correct = False
                    break

        return is_correct

    def compute(self, sent):
        assert(self.sentIsCorrect(sent)), f"Error: La frase < {sent} > no posee el formato correcto!"

        outside_words = re.sub("\'.*?\'", "", sent.replace("'", " ' "))
        outside_words = [w for w in outside_words.split() if w != ""]
        all_words = [w.strip() for w in sent.replace("'"," ").split() if w != ""]
        
        tks_id_outside_words = self.tokenizer.encode(
            " ".join(outside_words), 
            add_special_tokens=False, 
            truncation=True
        )
        tks_id_all_words = self.tokenizer.encode(
            " ".join(all_words), 
            add_special_tokens=False,
            truncation=True
        )

        diff = [(tk[0], tk[2:]) for tk in Differ().compare(tks_id_outside_words, tks_id_all_words)]

        cls_tk_id = self.tokenizer.cls_token_id
        sep_tk_id = self.tokenizer.sep_token_id
        mask_tk_id = self.tokenizer.mask_token_id

        all_sent_masked = []
        all_tks_id_masked = []
        all_tks_position_masked = []

        for i in range(0, len(diff)):
            current_sent_masked = [cls_tk_id]
            add_sent = True
            for j, (mark, tk_id) in enumerate(diff):
                if j == i:
                    if mark == '+':
                        add_sent = False
                        break
                    else:
                        current_sent_masked.append(mask_tk_id)
                        all_tks_id_masked.append(int(tk_id))
                        all_tks_position_masked.append(i+1)
                else:
                    current_sent_masked.append(int(tk_id))

            if add_sent:
                current_sent_masked.append(sep_tk_id)
                all_sent_masked.append(current_sent_masked)
        
        inputs_ids = torch.tensor(all_sent_masked)
        attention_mask = torch.ones_like(inputs_ids)

        with torch.no_grad():
            out = self.model(inputs_ids, attention_mask)
            logits = out.logits
            outputs = self.logSoftmax(logits)

        pll_score = 0
        for out, tk_pos, tk_id in zip(outputs, all_tks_position_masked, all_tks_id_masked):
            probabilities = out[tk_pos]
            tk_prob = probabilities[tk_id]
            pll_score += tk_prob.item()

        return pll_score