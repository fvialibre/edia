from memory_profiler import profile
import pandas as pd
from typing import List, Dict, Tuple

class Vocabulary:
    def __init__(
        self, 
        subset_name: str
    ) -> None:

        # Dataset info
        self.subset_name = subset_name
        self.ds_path = f"data/{subset_name}_vocab_v6.zip"
        
        # Pandas dataset
        self.df_vocab = None

        # Minimal list with (percentile,freq) tuples to be able to plot the word distribution graph
        self.histogram = None

        # Load vocabulary dataset
        self.__load()

    def __contains__(
        self, 
        word: str
    ) -> bool:

        return word in self.df_vocab['word'].to_list()

    def __load(
        self
    ) -> None:

        print(f"Preparing {self.subset_name} vocabulary...")

        # --- Download vocab dataset ---
        self.df_vocab = pd.read_json(self.ds_path)

        # --- Create min histogram to plot the word distribution graph ---
        x_values = self.df_vocab['percentile'].to_list()
        y_values = self.df_vocab['freq'].to_list()

        # Delete duplicated tups
        uniques_tups_list = set(list(zip(x_values, y_values)))
        # Leave only tuples with different first element
        uniques_tups_list = dict(uniques_tups_list)

        self.histogram = sorted(
            uniques_tups_list.items(),
            key=lambda tup: tup[0], 
            reverse=True
        )
        
    def __getValue(
        self, 
        word: str, 
        feature: str
    ):
        word_id, value = None, None

        if word in self:
            word_id = self.df_vocab['word'].to_list().index(word)
        
        if word_id != None:
            value = self.df_vocab[feature].to_list()[word_id]

        return value

    def getFreq(
        self, 
        word
    ) -> int:

        return self.__getValue(word, 'freq')

    def getPercentile(
        self, 
        word:str
    ) -> float:

        return self.__getValue(word, 'percentile')

    def getSplits(
        self, 
        word: str
    ) -> List[str]:

        return self.__getValue(word, 'splits')
    
    def getSubsets(
        self, 
        word: str
    ) -> Dict[str, int]:

        return self.__getValue(word, 'in_subset')

    def distribution(
        self
    ) -> Tuple:

        x_values, y_values = zip(*self.histogram)
        return x_values, y_values
     
    def getWordNeighbors(
        self, 
        word: str, 
        n_neighbors: int=20
    )-> Tuple:

        word_id = self.df_vocab['word'].to_list().index(word)
        words = self.df_vocab['word'].to_list()
        freqs = self.df_vocab['freq'].to_list()
        l_sorted = list(zip(words, freqs))

        g = l_sorted[max(0, word_id-n_neighbors):word_id]    # less than
        e = l_sorted[word_id]                               # equal than
        l = l_sorted[word_id+1:word_id+n_neighbors]         # greter than

        dic = dict(g+[e]+l)
        l = [x[0] for x in l]
        g = [x[0] for x in g]

        return dic, l, g