import numpy as np
import pandas as pd
import gradio as gr
from abc import ABC, abstractmethod
from modules.module_WordExplorer import WordExplorer
from modules.module_BiasExplorer import WordBiasExplorer
from modules.module_word2Context import Word2Context

class Connector(ABC):
    def parse_word(self, word : str):
        return word.lower().strip()

    def parse_words(self, array_in_string : str):
        words = array_in_string.strip()
        if not words:
            return []
        words = [self.parse_word(word) for word in words.split(',') if word != '']
        return words

    def buff_figure(self, fig):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        return im
    

class WordExplorerConnector(Connector):

    def __init__(self, **kwargs):
        if 'embedding' in kwargs:
            embedding = kwargs.get('embedding')
        else:
            raise KeyError
        self.word_explorer = WordExplorer(embedding)

    def plot_proyection_2d( self,
                            wordlist_0,
                            wordlist_1,
                            wordlist_2,
                            wordlist_3,
                            wordlist_4,
                            color_wordlist_0,
                            color_wordlist_1,
                            color_wordlist_2,
                            color_wordlist_3,
                            color_wordlist_4,
                            n_alpha,
                            fontsize,
                            n_neighbors
                            ):
        err = ""
        wordlist_0 = self.parse_words(wordlist_0)
        wordlist_1 = self.parse_words(wordlist_1)
        wordlist_2 = self.parse_words(wordlist_2)
        wordlist_3 = self.parse_words(wordlist_3)
        wordlist_4 = self.parse_words(wordlist_4)

        if not (wordlist_0 or wordlist_1 or wordlist_2 or wordlist_1 or wordlist_4):
            err = "<center><h3>" + "Ingresa al menos 1 palabras para continuar" + "<center><h3>"
            return None, err
        
        err = self.word_explorer.check_oov([wordlist_0, wordlist_1, wordlist_2, wordlist_3, wordlist_4])
        if err:
            return None, err

        fig = self.word_explorer.plot_projections_2d(wordlist_0,
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
                                                     n_neighbors=n_neighbors
                                                     )
        return self.buff_figure(fig), err

class BiasWordExplorerConnector(Connector):

    def __init__(self, **kwargs):
        if 'embedding' in kwargs:
            embedding = kwargs.get('embedding')
        else:
            raise KeyError
        self.bias_word_explorer = WordBiasExplorer(embedding)

    def calculate_bias_2d(self,
                         wordlist_1,
                         wordlist_2,
                         to_diagnose_list
                         ):
        err = ""
        wordlist_1 = self.parse_words(wordlist_1)
        wordlist_2 = self.parse_words(wordlist_2)
        to_diagnose_list = self.parse_words(to_diagnose_list)

        wordlists = [wordlist_1, wordlist_2, to_diagnose_list]
        for list in wordlists:
            if not list:
                err = "<center><h3>" + \
                      'Debe ingresar al menos 1 palabra en las lista de palabras a diagnosticar, sesgo 1 y sesgo 2' + \
                      "<center><h3>"
        if err:
            return None, err

        err = self.bias_word_explorer.check_oov(wordlists)
        if err:
            return None, err

        fig = self.bias_word_explorer.plot_biased_words(to_diagnose_list, wordlist_2, wordlist_1)

        return self.buff_figure(fig), err

    def calculate_bias_4d(self,
                         wordlist_1,
                         wordlist_2,
                         wordlist_3,
                         wordlist_4,
                         to_diagnose_list
                         ):
        err = ""
        wordlist_1 = self.parse_words(wordlist_1)
        wordlist_2 = self.parse_words(wordlist_2)
        wordlist_3 = self.parse_words(wordlist_3)
        wordlist_4 = self.parse_words(wordlist_4)
        to_diagnose_list = self.parse_words(to_diagnose_list)

        wordlists = [wordlist_1, wordlist_2, wordlist_3, wordlist_4, to_diagnose_list]
        for list in wordlists:
            if not list:
                err = "<center><h3>" + \
                      '¡Para graficar con 4 espacios, debe ingresar al menos 1 palabra en todas las listas!' + \
                      "<center><h3>"
        if err:
            return None, err

        err = self.bias_word_explorer.check_oov(wordlists)
        if err:
            return None, err

        fig = self.bias_word_explorer.plot_biased_words(to_diagnose_list, wordlist_1, wordlist_2, wordlist_3, wordlist_4)
        return self.buff_figure(fig), err

class Word2ContextExplorerConnector(Connector):
    def __init__(self, **kwargs):
        vocabulary = kwargs.get('vocabulary', None)
        context = kwargs.get('context', None)

        if vocabulary is None and context is None:
            raise KeyError
        self.word2context_explorer = Word2Context(context, vocabulary)

    def get_word_info(self, word):
        err = ""
        contexts = pd.DataFrame([],columns=[''])
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

    def get_word_context(self, word, n_context, subset_choice):
        word = self.parse_word(word)
        n_context = int(n_context)
        err = ""
        contexts = pd.DataFrame([], columns=[''])

        if len(subset_choice) > 0:
            ds = self.word2context_explorer.findSplits(word, subset_choice)
        else:
            err = "<center><h3>" + \
                  "Error: Palabra no ingresada y/o conjunto/s de interés no seleccionado/s!" + \
                  "</h3></center>"
            return err, contexts

        list_of_contexts = self.word2context_explorer.getContexts(word, n_context, ds)

        contexts = pd.DataFrame(list_of_contexts, columns=['#','contexto','conjunto'])
        contexts["buscar"] = contexts.contexto.apply(lambda text: self.word2context_explorer.genWebLink(text))

        return err, contexts