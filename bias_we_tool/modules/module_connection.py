import numpy as np
from modules.module_WordExplorer import WordExplorer
from modules.module_BiasExplorer import  WEBiasExplorer2d, WEBiasExplorer4d, WordBiasExplorer

class Connector:
    def __init__(self, vocabulary, to = 'explore'):
        if to == 'explore':
            self.word_explorer = WordExplorer(vocabulary)
            self.to = 'explore'
        else:
            self.we_bias_2d = WEBiasExplorer2d(vocabulary)
            self.we_bias_4d = WEBiasExplorer4d(vocabulary)
            self.bias_word_explorer = WordBiasExplorer(vocabulary)
            self.to = 'bias'

    def parse_words(self, array_in_string : str):
        words = array_in_string.strip()
        if not words:
            return []
        words = [word.lower().strip() for word in words.split(',') if word != '']
        return words

    def buff_figure(self, fig):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        return im

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
        assert self.to == 'explore', 'Inconsistency between Connector object created and its use'

        list_0 = self.parse_words(wordlist_0)
        list_1 = self.parse_words(wordlist_1)
        list_2 = self.parse_words(wordlist_2)
        list_3 = self.parse_words(wordlist_3)
        list_4 = self.parse_words(wordlist_4)
        fig = self.word_explorer.plot_projections_2d(list_0,
                                                     list_1,
                                                     list_2,
                                                     list_3,
                                                     list_4,
                                                     color_wordlist_0=color_wordlist_0,
                                                     color_wordlist_1=color_wordlist_1,
                                                     color_wordlist_2=color_wordlist_2,
                                                     color_wordlist_3=color_wordlist_3,
                                                     color_wordlist_4=color_wordlist_4,
                                                     n_alpha=n_alpha,
                                                     fontsize=fontsize,
                                                     n_neighbors=n_neighbors
                                                     )
        return self.buff_figure(fig), ''
    
    def calculate_bias_2d(self,
                         wordlist_1,
                         wordlist_2,
                         diagnose_list
                         ):
        assert self.to == 'bias', 'Inconsistency between Connector object created and its use'

        list_1 = self.parse_words(wordlist_1)
        list_2 = self.parse_words(wordlist_2)
        to_diagnose_list = self.parse_words(diagnose_list)

        fig = self.bias_word_explorer.plot_biased_words(to_diagnose_list, list_2, list_1)

        return self.buff_figure(fig), ''

    def calculate_bias_4d(self,
                         wordlist_1,
                         wordlist_2,
                         wordlist_3,
                         wordlist_4,
                         diagnose_list
                         ):
        assert self.to == 'bias', 'Inconsistency between Connector object created and its use'
        
        list_1 = self.parse_words(wordlist_1)
        list_2 = self.parse_words(wordlist_2)
        list_3 = self.parse_words(wordlist_3)
        list_4 = self.parse_words(wordlist_4)
        to_diagnose_list = self.parse_words(diagnose_list)

        fig = self.bias_word_explorer.plot_biased_words(to_diagnose_list, list_1, list_2, list_3, list_4)
        return self.buff_figure(fig), ''