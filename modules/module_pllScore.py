from difflib import Differ
import torch, re


class PllScore:
    def __init__(
        self, 
        language_model  # LanguageModel class instance
    ) -> None:

        self.tokenizer = language_model.initTokenizer()
        self.model = language_model.initModel()
        _ = self.model.eval()

        self.logSoftmax = torch.nn.LogSoftmax(dim=-1)

    def sentIsCorrect(
        self, 
        sent: str
    ) -> bool:

        # Mod
        is_correct = True

        # Check mark existence
        open_mark = sent.count("<")
        close_mark = sent.count(">")
        total_mark = open_mark + close_mark
        if (total_mark == 0) or (open_mark != close_mark):
            is_correct = False

        # Check existence of twin marks (ie: '<<' or '>>')
        if is_correct:
            left_twin = sent.count("<<")
            rigth_twin = sent.count(">>")
            if left_twin + rigth_twin > 0:
                is_correct = False

        if is_correct:
            # Check balanced symbols '<' and '>'
            stack = []
            for c in sent:
                if c == '<':
                    stack.append('<')
                elif c == '>':
                    if len(stack) == 0:
                        is_correct = False
                        break

                    if stack.pop() != "<":
                        is_correct = False
                        break

            if len(stack) > 0:
                is_correct = False

        if is_correct:
            for w in re.findall("\<.*?\>", sent):
                # Check empty interest words
                word = w.replace("<","").replace(">","").strip() 
                if not word:
                    is_correct = False
                    break
                
                # Check if there are any marks inside others (ie: <this is a <sentence>>)
                word = w.strip()[1:-1]  #Delete the first and last mark
                if '<' in word or '>' in word:
                    is_correct = False
                    break
        
        if is_correct:
            # Check that there is at least one uninteresting word. The next examples should not be allowed 
            # (ie: <this is a sent>, <this> <is a sent>)
            outside_words = re.sub("\<.*?\>", "", sent.replace("<", " < ").replace(">", " > "))
            outside_words = [w for w in outside_words.split() if w != ""]
            if not outside_words:
                is_correct = False


        return is_correct

    def compute(
        self, 
        sent: str
    ) -> float:

        assert(self.sentIsCorrect(sent)), f"Error: The sentence '{sent}' does not have the correct format!"

        outside_words = re.sub("\<.*?\>", "", sent.replace("<", " < ").replace(">", " > "))
        outside_words = [w for w in outside_words.split() if w != ""]
        all_words = [w.strip() for w in sent.replace("<"," ").replace(">"," ").split() if w != ""]

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