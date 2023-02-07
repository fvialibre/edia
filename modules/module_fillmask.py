import torch
from typing import Tuple, List
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM


class FillMask:
    def __init__(
        self,
        model_name: str
    ) -> None:
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.softmax = torch.nn.Softmax(dim=-1)

    def automatic_compute(
        self,
        sent: str
    ) -> List[Tuple[str,float]]:

        sent_masked = sent.replace("*", self.tokenizer.mask_token)
        
        outputs = pipeline(
            task="fill-mask", 
            model=self.model_name
        )
        
        predictions = outputs(sent_masked)
        predictions = [(dic['token_str'], dic['score']) for dic in predictions]

        predictions = sorted(predictions, key=lambda tup: tup[1], reverse=True)
        return predictions

    def manual_compute(
        self,
        sent: str,
        n: int = 5
    ) -> List[Tuple[str,float]]:

        sent_masked = sent.replace("*", self.tokenizer.mask_token)
        
        inputs = self.tokenizer.encode_plus( 
            sent_masked,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True, 
            truncation=True
        )
        
        token_position_mask = torch.where(
            inputs['input_ids'][0] == self.tokenizer.mask_token_id
        )[0].item()

        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits
            outputs = self.softmax(logits)
            outputs = torch.squeeze(outputs, dim=0)
        
        token_probabilities = outputs[token_position_mask]
        token_ids = torch.argsort(token_probabilities, descending=True)
        
        n_predictions = []
        for id_ in token_ids:
            token_str = self.tokenizer.decode([id_])
            token_prob = token_probabilities[id_].item()

            if len(n_predictions) < n:
                n_predictions.append((token_str, token_prob))

            elif len(n_predictions) >= n:
                break
        
        n_predictions = sorted(n_predictions, key=lambda tup: tup[1], reverse=True)
        return n_predictions

if __name__ == "__main__":
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    fillmask = FillMask(model_name)
    sent = "El * es un animal muy inteligente"
    print(fillmask.automatic_compute(sent))
    print(fillmask.manual_compute(sent))