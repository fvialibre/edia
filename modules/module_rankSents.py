from modules.module_customPllLabel import CustomPllLabel
from modules.module_pllScore import PllScore
from typing import List, Dict
import torch
import numpy as np


class RankSents:
    def __init__(
        self, 
        language_model, # LanguageModel class instance
        generative_language_model, # LanguageModel class instance
        lang: str,
        errorManager    # ErrorManager class instance
    ) -> None:
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
        # Fill-Mask LM
        self.tokenizer = language_model.initTokenizer()
        self.model = language_model.initModel()
        _ = self.model.eval()

        self.Label = CustomPllLabel()
        self.pllScore = PllScore(
            language_model=language_model
        )

        # Generative LM
        self.generative_tokenizer = generative_language_model.initTokenizer()
        self.generative_model = generative_language_model.initModel()
        _ = self.generative_model.eval()

        self.generative_Label = CustomPllLabel()
        self.generative_pllScore = PllScore(
            language_model=generative_language_model
        )

        if lang == "es":
            self.articles = [
                'un','una','unos','unas','el','los','la','las','lo'
            ]
            self.prepositions = [
                'a','ante','bajo','cabe','con','contra','de','desde','en','entre','hacia','hasta','para','por','según','sin','so','sobre','tras','durante','mediante','vía','versus'
            ]
            self.conjunctions = [
                'y','o','ni','que','pero','si'
            ]

        elif lang == "en":
            self.articles = [
                'a','an', 'the'
            ]
            self.prepositions = [
                'above', 'across', 'against', 'along', 'among', 'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'by', 'down', 'from', 'in', 'into', 'near', 'of', 'off', 'on', 'to', 'toward', 'under', 'upon', 'with', 'within'
            ]
            self.conjunctions = [
                'and', 'or', 'but', 'that', 'if', 'whether'
            ]

        self.errorManager = errorManager

    def errorChecking(
        self, 
        sent: str
    ) -> str:

        out_msj = ""
        if not sent:
            out_msj = ['RANKSENTS_NO_SENTENCE_PROVIDED']
        elif sent.count("*") > 1:
            out_msj = ['RANKSENTS_TOO_MANY_MASKS_IN_SENTENCE']
        elif sent.count("*") == 0:
            out_msj = ['RANKSENTS_NO_MASK_IN_SENTENCE']
        else:
            sent_len = len(self.tokenizer.encode(sent.replace("*", self.tokenizer.mask_token)))
            max_len = self.tokenizer.max_len_single_sentence
            if sent_len > max_len:
                out_msj = ['RANKSENTS_TOKENIZER_MAX_TOKENS_REACHED', max_len]
        
        return self.errorManager.process(out_msj)
    
    def errorInterestWords(
        self, 
        interest_word_list: List[str]
    ) -> str:
        
        out_msj = ""
        if len(interest_word_list) <= 1:
            out_msj = ['INTEREST_WORDS_NOT_ENOUGH_WORDS']
        
        return self.errorManager.process(out_msj)
    
    def errorTypeOfBiasExplored(
        self, 
        type_of_bias_explored: List[str]
    ) -> str:
        
        print (f"type_of_bias_explored: {type_of_bias_explored}")
        out_msj = ""
        if len(type_of_bias_explored) == 0:
            print("TYPE_OF_BIAS_EXPLORED_EMPTY")
            out_msj = ['TYPE_OF_BIAS_EXPLORED_EMPTY']
        
        return self.errorManager.process(out_msj)

    # def getTopPredictionsBETO(
    #     self, 
    #     sent: str,
    #     n: int=5,
    #     banned_word_list: List[str]=[], 
    #     exclude_articles: bool=False,
    #     exclude_prepositions: bool=False,
    #     exclude_conjunctions: bool=False,
    # ) -> List[str]:
                                
    #     sent_masked = sent.replace("*", self.tokenizer.mask_token)
    #     inputs = self.tokenizer.encode_plus( 
    #         sent_masked,
    #         add_special_tokens=True,
    #         return_tensors='pt',
    #         return_attention_mask=True, 
    #         truncation=True
    #     )

    #     tk_position_mask = torch.where(inputs['input_ids'][0] == self.tokenizer.mask_token_id)[0].item()

    #     with torch.no_grad():
    #         out = self.model(**inputs)
    #         logits = out.logits
    #         outputs = self.softmax(logits)
    #         outputs = torch.squeeze(outputs, dim=0)
        
    #     probabilities = outputs[tk_position_mask]
    #     first_tk_id = torch.argsort(probabilities, descending=True)
        
    #     top_tks_pred = []
    #     for tk_id in first_tk_id:
    #         tk_string = self.tokenizer.decode([tk_id])
            
    #         tk_is_banned = tk_string in banned_word_list
    #         tk_is_punctuation = not tk_string.isalnum()
    #         tk_is_substring = tk_string.startswith("##")
    #         tk_is_special = (tk_string in self.tokenizer.all_special_tokens)

    #         if exclude_articles:
    #             tk_is_article = tk_string in self.articles
    #         else:
    #             tk_is_article = False
            
    #         if exclude_prepositions:
    #             tk_is_prepositions = tk_string in self.prepositions
    #         else:
    #             tk_is_prepositions = False
            
    #         if exclude_conjunctions:
    #             tk_is_conjunctions = tk_string in self.conjunctions
    #         else:
    #             tk_is_conjunctions = False
            
    #         predictions_is_dessire = not any([  
    #                                 tk_is_banned,
    #                                 tk_is_punctuation,
    #                                 tk_is_substring, 
    #                                 tk_is_special, 
    #                                 tk_is_article, 
    #                                 tk_is_prepositions,
    #                                 tk_is_conjunctions
    #         ])

    #         if predictions_is_dessire and len(top_tks_pred) < n:
    #             top_tks_pred.append(tk_string)

    #         elif len(top_tks_pred) >= n:
    #             break

    #     return top_tks_pred

    def get_generative_next_word_distribution(self, input_ids):
        outputs = self.generative_model.generate(
            inputs=input_ids,
            max_new_tokens=1,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=self.generative_tokenizer.eos_token_id,
            pad_token_id=self.generative_tokenizer.eos_token_id,
        )

        return(outputs['scores'][0], outputs)

    def get_generative_metric(self, s):
        s_tokenized = self.generative_tokenizer(s, return_tensors="pt")
        next_tokens_scores = []
        for i in range(1, len(s_tokenized['input_ids'][0])):
            first_i_tokens = self.generative_tokenizer(
                s,
                return_tensors='pt',
                truncation=True,
                max_length=i
            ).to(self.generative_model.device)
            distr, _ = self.get_generative_next_word_distribution(first_i_tokens['input_ids'])
            softmaxed_distr = torch.nn.functional.softmax(distr, dim=-1)

            next_token = s_tokenized['input_ids'][0][i].item()
            
            # add 1 to avoid multiplying by 0
            next_tokens_scores.append(1 + softmaxed_distr[0][next_token].item())
            final_metric_score = np.prod(next_tokens_scores)
        return final_metric_score

    def generative_lm_rank(
        self, 
        sent: str,
        interest_word_list: List[str]=[],
        n: int=5,
    ) -> List[str]:
        
        # replace * in sent for every word in interest_word_list
        sent_list = []
        for word in interest_word_list:
            sent_list.append((sent.replace("*", word), sent.replace("*", "<"+word+">")))

        all_scores = {}
        for sent, sent_bracket in sent_list:
            all_scores[sent_bracket] = self.get_generative_metric(sent)
        return all_scores

    def rank(self, 
        sent: str, 
        interest_word_list: List[str]=[], 
        banned_word_list: List[str]=[], 
        exclude_articles: bool=False, 
        exclude_prepositions: bool=False, 
        exclude_conjunctions: bool=False,
        n_predictions: int=5,
        model_name: str="BETO"
    ) -> Dict[str, float]:
        
        err = self.errorChecking(sent)
        if err:
            raise ValueError(err)

        if model_name == "BETO":
            sent_list = []
            sent_list2print = []
            for word in interest_word_list:
                sent_list.append(sent.replace("*", "<"+word+">"))
                sent_list2print.append(sent.replace("*", "<"+word+">"))
                
            all_plls_scores = {}
            for sent, sent2print in zip(sent_list, sent_list2print):
                all_plls_scores[sent2print] = self.pllScore.compute(sent)

            return all_plls_scores
        elif model_name == "generative_lm":
            return self.generative_lm_rank(
                sent,
                interest_word_list,
                n_predictions,
            )
