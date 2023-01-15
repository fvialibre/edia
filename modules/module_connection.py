import csv, os
import pandas as pd
import gradio as gr
from abc import ABC
from modules.utils import DateLogs
from typing import List, Tuple, Any
from modules.module_WordExplorer import WordExplorer
from modules.module_BiasExplorer import WEBiasExplorer2Spaces, WEBiasExplorer4Spaces
from modules.module_word2Context import Word2Context
from modules.module_rankSents import RankSents
from modules.module_crowsPairs import CrowsPairs
from modules.module_ErrorManager import ErrorManager


class Connector(ABC):

    def __init__(
        self,
        lang: str
    ) -> None:

        self.datalog = DateLogs()
        self.log_folder = 'logs'

        if not hasattr(Connector, 'errorManager'):
            Connector.errorManager = ErrorManager(
                path=f"modules/error_messages/{lang}.json"
            )

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
    
    def logs_save(
        self,
        file_name: str,
        headers: List[str]=None,
        *data: List[Any]
    ) -> None:

        if file_name is None:
            return None

        if not os.path.exists(self.log_folder):
            print(f"Creating logs folder '{self.log_folder}' ...")
            os.mkdir(self.log_folder)

        file_path = os.path.join(self.log_folder, file_name+'.csv')
        f_out = None

        if not os.path.exists(file_path):
            print(f"Creating new '{file_name}' logs file...")

            with open(file_path, mode='w', encoding='UTF8') as f_out:
                # Create the csv writer
                writer = csv.writer(f_out)
                
                # Write the header
                if headers is None:
                    headers = [
                        "input_" + str(ith)  
                        for ith,_ in enumerate(data)
                    ]
                headers = headers + ["datatime"]

                writer.writerow(headers)
                    
        with open(file_path, mode='a', encoding='UTF8') as f_out:
            # Create the csv writer
            writer = csv.writer(f_out)

            # Write a row to the csv file
            data = list(data) + [ self.datalog.full() ]
            writer.writerow(data)

            print(f"Logs: '{file_path}' successfully saved!")

class WordExplorerConnector(Connector):
    def __init__(
        self, 
        **kwargs
    ) -> None:

        Connector.__init__(self, kwargs.get('lang', 'en'))
        embedding = kwargs.get('embedding', None)
        self.logs_file_name = kwargs.get('logs_file_name', None)
        self.headers = [
            "word_list_to_diagnose",
            "word_list_1",
            "word_list_2",
            "word_list_3",
            "word_list_4"
        ]
        
        if embedding is None:
            raise KeyError
        
        self.word_explorer = WordExplorer(
            embedding=embedding,
            errorManager=self.errorManager
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
            err = self.errorManager.process(['CONECTION_NO_WORD_ENTERED'])
            return None, err
        
        err = self.word_explorer.check_oov(
            [wordlist_0, wordlist_1, wordlist_2, wordlist_3, wordlist_4]
        )

        if err:
            return None, err

        # Save inputs in logs file
        self.logs_save(
            self.logs_file_name,
            self.headers,
            wordlist_0,
            wordlist_1,
            wordlist_2,
            wordlist_3,
            wordlist_4,
        )

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

        return fig, err

class BiasWordExplorerConnector(Connector):

    def __init__(
        self, 
        **kwargs
    ) -> None:

        Connector.__init__(self, kwargs.get('lang', 'en'))
        embedding = kwargs.get('embedding', None)
        self.logs_file_name = kwargs.get('logs_file_name', None)
        self.headers = [
            "word_list_to_diagnose",
            "word_list_1",
            "word_list_2",
            "word_list_3",
            "word_list_4",
            "plot_space"
        ]

        if embedding is None:
            raise KeyError

        self.bias_word_explorer_2_spaces = WEBiasExplorer2Spaces(
            embedding=embedding,
            errorManager=self.errorManager
        )
        self.bias_word_explorer_4_spaces = WEBiasExplorer4Spaces(
            embedding=embedding,
            errorManager=self.errorManager
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
                err = self.errorManager.process(['BIASEXPLORER_NOT_ENOUGH_WORD_2_KERNELS'])
        if err:
            return None, err

        err = self.bias_word_explorer_2_spaces.check_oov(word_lists)
        if err:
            return None, err

        # Save inputs in logs file
        self.logs_save(
            self.logs_file_name,
            self.headers,
            to_diagnose_list,
            wordlist_1,
            wordlist_2,
            "",
            "",
            "2d"
        )

        fig = self.bias_word_explorer_2_spaces.calculate_bias(
            to_diagnose_list, 
            wordlist_1, 
            wordlist_2
        )

        return fig, err

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
                err = self.errorManager.process(['BIASEXPLORER_NOT_ENOUGH_WORD_4_KERNELS'])
        if err:
            return None, err

        err = self.bias_word_explorer_4_spaces.check_oov(wordlists)
        if err:
            return None, err

        # Save inputs in logs file
        self.logs_save(
            self.logs_file_name,
            self.headers,
            to_diagnose_list, 
            wordlist_1, 
            wordlist_2, 
            wordlist_3, 
            wordlist_4,
            "4d"
        )

        fig = self.bias_word_explorer_4_spaces.calculate_bias(
            to_diagnose_list, 
            wordlist_1, 
            wordlist_2, 
            wordlist_3, 
            wordlist_4
        )
        
        return fig, err

class Word2ContextExplorerConnector(Connector):
    def __init__(
        self, 
        **kwargs
    ) -> None:

        Connector.__init__(self, kwargs.get('lang', 'en'))
        vocabulary = kwargs.get('vocabulary', None)
        context = kwargs.get('context', None)
        self.logs_file_name = kwargs.get('logs_file_name', None)
        self.headers = [
            "word",
            "subsets_choice"
        ]

        if vocabulary is None or context is None:
            raise KeyError

        self.word2context_explorer = Word2Context(
            context,
            vocabulary,
            errorManager=self.errorManager
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
            return err, contexts, subsets_info, distribution_plot, word_cloud_plot, subsets_choice

        word = self.parse_word(word)

        subsets_info, subsets_origin_info = self.word2context_explorer.getSubsetsInfo(word)

        clean_keys = [key.split(" ")[0].strip() for key in subsets_origin_info]
        subsets_choice = gr.CheckboxGroup.update(choices=clean_keys)

        distribution_plot = self.word2context_explorer.genDistributionPlot(word)
        word_cloud_plot = self.word2context_explorer.genWordCloudPlot(word)

        return err, contexts, subsets_info, distribution_plot, word_cloud_plot, subsets_choice

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
            return err, contexts

        if len(subset_choice) > 0:
            ds = self.word2context_explorer.findSplits(word, subset_choice)
        else:
            err = self.errorManager.process(['WORD2CONTEXT_WORDS_OR_SET_MISSING'])
            return err, contexts

        # Save inputs in logs file
        self.logs_save(
            self.logs_file_name, 
            self.headers,
            word,
            subset_choice
        )

        list_of_contexts = self.word2context_explorer.getContexts(word, n_context, ds)

        contexts = pd.DataFrame(list_of_contexts, columns=['#','contexto','conjunto'])
        contexts["buscar"] = contexts.contexto.apply(lambda text: self.word2context_explorer.genWebLink(text))

        return err, contexts

class PhraseBiasExplorerConnector(Connector):
    def __init__(
        self, 
        **kwargs
    ) -> None:

        Connector.__init__(self, kwargs.get('lang', 'en'))
        language_model = kwargs.get('language_model', None)
        lang =  kwargs.get('lang', None)
        self.logs_file_name = kwargs.get('logs_file_name', None)
        self.headers = [
            "sent",
            "word_list"
        ]

        if language_model is None or lang is None:
            raise KeyError

        self.phrase_bias_explorer = RankSents(
            language_model=language_model,
            lang=lang,
            errorManager=self.errorManager
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
            return err, "", ""

        word_list = self.parse_words(word_list)
        banned_word_list = self.parse_words(banned_word_list)

        # Save inputs in logs file
        self.logs_save(
            self.logs_file_name, 
            self.headers,
            sent,
            word_list
        )

        all_plls_scores = self.phrase_bias_explorer.rank(
            sent, 
            word_list, 
            banned_word_list, 
            useArticles, 
            usePrepositions, 
            useConjunctions
        )
        
        all_plls_scores = self.phrase_bias_explorer.Label.compute(all_plls_scores)
        return err, all_plls_scores, ""

class CrowsPairsExplorerConnector(Connector):
    def __init__(
        self, 
        **kwargs
    ) -> None:

        Connector.__init__(self, kwargs.get('lang', 'en'))
        language_model = kwargs.get('language_model', None)
        self.logs_file_name = kwargs.get('logs_file_name', None)
        self.headers = [
            "sent_1",
            "sent_2",
            "sent_3",
            "sent_4",
            "sent_5",
            "sent_6",
        ]

        if language_model is None:
            raise KeyError
        
        self.crows_pairs_explorer = CrowsPairs(
            language_model=language_model,
            errorManager=self.errorManager
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
            return err, "", ""

        # Save inputs in logs file
        self.logs_save(
            self.logs_file_name, 
            self.headers,
            sent_list
        )

        all_plls_scores = self.crows_pairs_explorer.rank(
            sent_list
        )
        
        all_plls_scores = self.crows_pairs_explorer.Label.compute(all_plls_scores)
        return err, all_plls_scores, ""