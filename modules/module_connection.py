import numpy as np
import pandas as pd
import gradio as gr
from abc import ABC, abstractmethod
from modules.module_WordExplorer import WordExplorer
from modules.module_BiasExplorer import WEBiasExplorer2Spaces, WEBiasExplorer4Spaces
from modules.module_word2Context import Word2Context
from modules.module_rankSents import RankSents
from modules.module_crowsPairs import CrowsPairs
from typing import List, Tuple

class Connector(ABC):
    def parse_word(
        self, 
        word: str
    ) -> str:
        
        return word.lower().strip()

    def parse_words(
        self, 
        array_in_string: str
    ) -> List[str]:

        words = array_in_string.strip()
        if not words:
            return []

        words = [
            self.parse_word(word) 
            for word in words.split(',') if word.strip() != ''
        ]
        return words

    def process_error(
        self, 
        err: str
    ) -> str:

        if err:
            err = "<center><h3>" + err + "</h3></center>"
        return err    

class WordExplorerConnector(Connector):
    def __init__(
        self, 
        **kwargs
    ) -> None:

        if 'embedding' in kwargs:
            embedding = kwargs.get('embedding')
        else:
            raise KeyError
        
        self.word_explorer = WordExplorer(
            embedding=embedding
        )

    def plot_proyection_2d( 
        self,
        wordlist_0: str,
        wordlist_1: str,
        wordlist_2: str,
        wordlist_3: str,
        wordlist_4: str,
        color_wordlist_0: str,
        color_wordlist_1: str,
        color_wordlist_2: str,
        color_wordlist_3: str,
        color_wordlist_4: str,
        n_alpha: float,
        fontsize: int,
        n_neighbors: int
    ) -> Tuple:

        err = ""
        neighbors_method = 'sklearn'
        wordlist_0 = self.parse_words(wordlist_0)
        wordlist_1 = self.parse_words(wordlist_1)
        wordlist_2 = self.parse_words(wordlist_2)
        wordlist_3 = self.parse_words(wordlist_3)
        wordlist_4 = self.parse_words(wordlist_4)

        if not (wordlist_0 or wordlist_1 or wordlist_2 or wordlist_1 or wordlist_4):
            err = self.process_error("Enter at least one word to continue")
            return None, err
        
        err = self.word_explorer.check_oov(
            [wordlist_0, wordlist_1, wordlist_2, wordlist_3, wordlist_4]
        )

        if err:
            return None, self.process_error(err)

        fig = self.word_explorer.plot_projections_2d(
            wordlist_0,
            wordlist_1,
            wordlist_2,
            wordlist_3,
            wordlist_4,
            color_wordlist_0=color_wordlist_0,
            color_wordlist_1=color_wordlist_1,
            color_wordlist_2=color_wordlist_2,
            color_wordlist_3=color_wordlist_3,
            color_wordlist_4=color_wordlist_4,
            n_alpha=n_alpha,
            fontsize=fontsize,
            n_neighbors=n_neighbors,
            nn_method = neighbors_method
        )

        return fig, self.process_error(err)

class BiasWordExplorerConnector(Connector):

    def __init__(
        self, 
        **kwargs
    ) -> None:

        if 'embedding' in kwargs:
            embedding = kwargs.get('embedding')
        else:
            raise KeyError

        self.bias_word_explorer_2_spaces = WEBiasExplorer2Spaces(
            embedding=embedding
        )
        self.bias_word_explorer_4_spaces = WEBiasExplorer4Spaces(
            embedding=embedding
        )

    def calculate_bias_2d(
        self,
        wordlist_1: str,
        wordlist_2: str,
        to_diagnose_list: str
    ) -> Tuple:

        err = ""
        wordlist_1 = self.parse_words(wordlist_1)
        wordlist_2 = self.parse_words(wordlist_2)
        to_diagnose_list = self.parse_words(to_diagnose_list)

        word_lists = [wordlist_1, wordlist_2, to_diagnose_list]
        for _list in word_lists:
            if not _list:
                err = "At least one word should be in the to diagnose list, bias 1 list and bias 2 list"
        if err:
            return None, self.process_error(err)

        err = self.bias_word_explorer_2_spaces.check_oov(word_lists)
        if err:
            return None, self.process_error(err)

        fig = self.bias_word_explorer_2_spaces.calculate_bias(
            to_diagnose_list, 
            wordlist_1, 
            wordlist_2
        )

        return fig, self.process_error(err)

    def calculate_bias_4d(
        self,
        wordlist_1: str,
        wordlist_2: str,
        wordlist_3: str,
        wordlist_4: str,
        to_diagnose_list: str
    ) -> Tuple:

        err = ""
        wordlist_1 = self.parse_words(wordlist_1)
        wordlist_2 = self.parse_words(wordlist_2)
        wordlist_3 = self.parse_words(wordlist_3)
        wordlist_4 = self.parse_words(wordlist_4)
        to_diagnose_list = self.parse_words(to_diagnose_list)

        wordlists = [wordlist_1, wordlist_2, wordlist_3, wordlist_4, to_diagnose_list]
        for _list in wordlists:
            if not _list:
                err = "To plot with 4 spaces, you must enter at least one word in all lists"
        if err:
            return None, self.process_error(err)

        err = self.bias_word_explorer_4_spaces.check_oov(wordlists)
        if err:
            return None, self.process_error(err)

        fig = self.bias_word_explorer_4_spaces.calculate_bias(
            to_diagnose_list, 
            wordlist_1, 
            wordlist_2, 
            wordlist_3, 
            wordlist_4
        )
        
        return fig, self.process_error(err)

class Word2ContextExplorerConnector(Connector):
    def __init__(
        self, 
        **kwargs
    ) -> None:

        vocabulary = kwargs.get('vocabulary', None)
        context = kwargs.get('context', None)

        if vocabulary is None and context is None:
            raise KeyError

        self.word2context_explorer = Word2Context(
            context,    # Context dataset HF name | path
            vocabulary  # Vocabulary class instance
        )

    def get_word_info(
        self, 
        word: str
    ) -> Tuple:
    
        err = ""
        contexts = pd.DataFrame([], columns=[''])
        subsets_info = ""
        distribution_plot = None
        word_cloud_plot = None
        subsets_choice = gr.CheckboxGroup.update(choices=[])

        err = self.word2context_explorer.errorChecking(word)
        if err:
            return self.process_error(err), contexts, subsets_info, distribution_plot, word_cloud_plot, subsets_choice

        word = self.parse_word(word)

        subsets_info, subsets_origin_info = self.word2context_explorer.getSubsetsInfo(word)

        clean_keys = [key.split(" ")[0].strip() for key in subsets_origin_info]
        subsets_choice = gr.CheckboxGroup.update(choices=clean_keys)

        distribution_plot = self.word2context_explorer.genDistributionPlot(word)
        word_cloud_plot = self.word2context_explorer.genWordCloudPlot(word)

        return self.process_error(err), contexts, subsets_info, distribution_plot, word_cloud_plot, subsets_choice

    def get_word_context(
        self,
        word: str,
        n_context: int,
        subset_choice: List[str]
    ) -> Tuple:

        word = self.parse_word(word)
        err = ""
        contexts = pd.DataFrame([], columns=[''])

        err = self.word2context_explorer.errorChecking(word)
        if err:
            return self.process_error(err), contexts

        if len(subset_choice) > 0:
            ds = self.word2context_explorer.findSplits(word, subset_choice)
        else:
            err = self.process_error("Error: Palabra no ingresada y/o conjunto/s de interés no seleccionado/s!")
            return err, contexts

        list_of_contexts = self.word2context_explorer.getContexts(word, n_context, ds)

        contexts = pd.DataFrame(list_of_contexts, columns=['#','contexto','conjunto'])
        contexts["buscar"] = contexts.contexto.apply(lambda text: self.word2context_explorer.genWebLink(text))

        return self.process_error(err), contexts

class PhraseBiasExplorerConnector(Connector):
    def __init__(
        self, 
        **kwargs
    ) -> None:

        if 'language_model' in kwargs and 'lang' in kwargs:
            language_model = kwargs.get('language_model')
            lang =  kwargs.get('lang')
        else:
            raise KeyError

        self.phrase_bias_explorer = RankSents(
            language_model=language_model,
            lang=lang
        )

    def rank_sentence_options(
        self,
        sent: str,
        word_list: str,
        banned_word_list: str,
        useArticles: bool,
        usePrepositions: bool,
        useConjunctions: bool
    ) -> Tuple:

        sent = " ".join(sent.strip().replace("*"," * ").split())

        err = self.phrase_bias_explorer.errorChecking(sent)
        if err:
            return self.process_error(err), "", ""

        word_list = self.parse_words(word_list)
        banned_word_list = self.parse_words(banned_word_list)

        all_plls_scores = self.phrase_bias_explorer.rank(
            sent, 
            word_list, 
            banned_word_list, 
            useArticles, 
            usePrepositions, 
            useConjunctions
        )
        
        all_plls_scores = self.phrase_bias_explorer.Label.compute(all_plls_scores)
        return self.process_error(err), all_plls_scores, ""


class CrowsPairsExplorerConnector(Connector):
    def __init__(
        self, 
        **kwargs
    ) -> None:

        if 'language_model' in kwargs:
            language_model = kwargs.get('language_model')
        else:
            raise KeyError
        
        self.crows_pairs_explorer = CrowsPairs(
            language_model=language_model
        )

    def compare_sentences(
        self,
        sent0: str,
        sent1: str,
        sent2: str,
        sent3: str,
        sent4: str,
        sent5: str
    ) -> Tuple:

        sent_list = [sent0, sent1, sent2, sent3, sent4, sent5]
        err = self.crows_pairs_explorer.errorChecking(
            sent_list
        )

        if err:
            return self.process_error(err), "", ""

        all_plls_scores = self.crows_pairs_explorer.rank(
            sent_list
        )
        
        all_plls_scores = self.crows_pairs_explorer.Label.compute(all_plls_scores)
        return self.process_error(err), all_plls_scores, ""