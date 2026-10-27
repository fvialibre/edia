from modules.module_customPllLabel import CustomPllLabel
from modules.module_pllScore import PllScore
from typing import List, Dict
import torch


class RankSents:
    def __init__(
        self, 
        spanish_language_model, # LanguageModel class instance
        english_language_model, # LanguageModel class instance
        lang: str,
        errorManager    # ErrorManager class instance
    ) -> None:
        
        self.Label = CustomPllLabel()
        self.softmax = torch.nn.Softmax(dim=-1)

        spanish_dict_key = "spanish"
        english_dict_key = "english"

        self.tokenizer = {
            spanish_dict_key: spanish_language_model.initTokenizer(),
            english_dict_key: english_language_model.initTokenizer()
        }

        self.model = {
            spanish_dict_key: spanish_language_model.initModel(),
            english_dict_key: english_language_model.initModel()
        }

        _ = self.model[spanish_dict_key].eval()
        _ = self.model[english_dict_key].eval()

        self.pllScore = {
            spanish_dict_key: PllScore(
                language_model=spanish_language_model
            ),
            english_dict_key: PllScore(
                language_model=english_language_model
            ),
        }

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
        sent: str,
        lang_dict_key
    ) -> str:

        out_msj = ""
        if not sent:
            out_msj = ['RANKSENTS_NO_SENTENCE_PROVIDED']
        elif sent.count("*") > 1:
            out_msj = ['RANKSENTS_TOO_MANY_MASKS_IN_SENTENCE']
        elif sent.count("*") == 0:
            out_msj = ['RANKSENTS_NO_MASK_IN_SENTENCE']
        else:
            sent_len = len(self.tokenizer[lang_dict_key].encode(sent.replace("*", self.tokenizer[lang_dict_key].mask_token)))
            max_len = self.tokenizer[lang_dict_key].max_len_single_sentence
            if sent_len > max_len:
                out_msj = ['RANKSENTS_TOKENIZER_MAX_TOKENS_REACHED', max_len]
        
        return self.errorManager.process(out_msj)
    
    def errorInterestWords(
        self, 
        interest_word_list: List[str]
    ) -> str:
        
        out_msj = ""
        # if len(interest_word_list) == 0:
        #     out_msj = ['INTEREST_WORDS_NOT_ENOUGH_WORDS']
        
        return self.errorManager.process(out_msj)
    
    def errorTypeOfBiasExplored(
        self, 
        type_of_bias_explored: List[str]
    ) -> str:
        
        out_msj = ""
        if type_of_bias_explored is None or len(type_of_bias_explored) == 0:
            print("TYPE_OF_BIAS_EXPLORED_EMPTY")
            out_msj = ['TYPE_OF_BIAS_EXPLORED_EMPTY']
        
        return self.errorManager.process(out_msj)

    def getTopPredictions(
        self, 
        sent: str,
        n: int=5,
        banned_word_list: List[str]=[], 
        exclude_articles: bool=False,
        exclude_prepositions: bool=False,
        exclude_conjunctions: bool=False,
        model_name: str="",
    ) -> List[str]:
                                        
        sent_masked = sent.replace("*", self.tokenizer[model_name].mask_token)
        inputs = self.tokenizer[model_name].encode_plus( 
            sent_masked,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True, 
            truncation=True
        )

        tk_position_mask = torch.where(inputs['input_ids'][0] == self.tokenizer[model_name].mask_token_id)[0].item()

        with torch.no_grad():
            out = self.model[model_name](**inputs)
            logits = out.logits
            outputs = self.softmax(logits)
            outputs = torch.squeeze(outputs, dim=0)
        
        probabilities = outputs[tk_position_mask]
        first_tk_id = torch.argsort(probabilities, descending=True)
        
        top_tks_pred = []
        for tk_id in first_tk_id:
            tk_string = self.tokenizer[model_name].decode([tk_id])
            
            tk_is_banned = tk_string in banned_word_list
            tk_is_punctuation = not tk_string.isalnum()
            tk_is_substring = tk_string.startswith("##")
            tk_is_special = (tk_string in self.tokenizer[model_name].all_special_tokens)

            if exclude_articles:
                tk_is_article = tk_string in self.articles
            else:
                tk_is_article = False
            
            if exclude_prepositions:
                tk_is_prepositions = tk_string in self.prepositions
            else:
                tk_is_prepositions = False
            
            if exclude_conjunctions:
                tk_is_conjunctions = tk_string in self.conjunctions
            else:
                tk_is_conjunctions = False
            
            predictions_is_dessire = not any([  
                                    tk_is_banned,
                                    tk_is_punctuation,
                                    tk_is_substring, 
                                    tk_is_special, 
                                    tk_is_article, 
                                    tk_is_prepositions,
                                    tk_is_conjunctions
            ])

            if predictions_is_dessire and len(top_tks_pred) < n:
                top_tks_pred.append(tk_string)

            elif len(top_tks_pred) >= n:
                break

        return top_tks_pred

    def rank(self, 
        sent: str, 
        interest_word_list: List[str]=[], 
        banned_word_list: List[str]=[], 
        exclude_articles: bool=False, 
        exclude_prepositions: bool=False, 
        exclude_conjunctions: bool=False,
        n_predictions: int=5,
        model_name: str=""
    ) -> Dict[str, float]:
        
        err = self.errorChecking(sent, model_name)
        if err:
            raise ValueError(err)

        if not interest_word_list:
            interest_word_list = self.getTopPredictions(
                sent,
                n_predictions,
                banned_word_list,
                exclude_articles,
                exclude_prepositions,
                exclude_conjunctions,
                model_name,
            )

        sent_list = []
        sent_list2print = []
        for word in interest_word_list:
            sent_list.append(sent.replace("*", "<"+word+">"))
            sent_list2print.append(sent.replace("*", "<"+word+">"))
            
        all_plls_scores = {}
        for sent, sent2print in zip(sent_list, sent_list2print):
            all_plls_scores[sent2print] = self.pllScore[model_name].compute(sent)

        return all_plls_scores