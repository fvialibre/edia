from modules.module_WordExplorer import WordExplorer
from modules.module_BiasExplorer import  WEBiasExplorer2d, WEBiasExplorer4d

class Connector:
    def __init__(self, vocabulary, to = 'explore'):
        if to == 'explore':
            self.word_explorer = WordExplorer(vocabulary)
            self.to = 'explore'
        else:
            self.we_bias_2d = WEBiasExplorer2d(vocabulary)
            self.we_bias_4d = WEBiasExplorer4d(vocabulary)
            self.to = 'bias'

    def parse_words(self, array_in_string : str):
        words = array_in_string.strip()
        if words:
            words = [word.strip() for word in words.split(',') if word != '']
        return words

    def plot_proyection_2d( self,
                            wordlist,
                            wordlist_1,
                            wordlist_2,
                            wordlist_3,
                            wordlist_4,
                            color_wordlist,
                            color_wordlist_1,
                            color_wordlist_2,
                            color_wordlist_3,
                            color_wordlist_4,
                            n_alpha,
                            fontsize,
                            n_neighbors
                            ):
        assert self.to == 'explore', 'Inconsistency between Connector object created and its use'

        to_diagnose_list = self.parse_words(wordlist)
        list_1 = self.parse_words(wordlist_1)
        list_2 = self.parse_words(wordlist_2)
        list_3 = self.parse_words(wordlist_3)
        list_4 = self.parse_words(wordlist_4)
        return self.word_explorer.plot_projections_2d(to_diagnose_list,
                                                    list_1,
                                                    list_2,
                                                    list_3,
                                                    list_4,
                                                    color_wordlist,
                                                    color_wordlist_1,
                                                    color_wordlist_2,
                                                    color_wordlist_3,
                                                    color_wordlist_4,
                                                    n_alpha,
                                                    fontsize,
                                                    n_neighbors
                                                    )
    
    def calculate_bias_2d(self,
                         wordlist_1,
                         wordlist_2,
                         diagnose_list
                         ):
        assert self.to == 'bias', 'Inconsistency between Connector object created and its use'

        list_1 = self.parse_words(wordlist_1)
        list_2 = self.parse_words(wordlist_2)
        to_diagnose_list = self.parse_words(diagnose_list)
        return self.we_bias_2d.calculate_bias(list_1, list_2, to_diagnose_list)

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
        return self.we_bias_4d.calculate_bias(list_1, list_2, list_3, list_4, to_diagnose_list)