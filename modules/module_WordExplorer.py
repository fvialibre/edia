import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import norm

import matplotlib as mpl
mpl.use('Agg')
from typing import List, Dict, Tuple


class WordToPlot:
    def __init__(
        self,
        word: str,
        color: str,
        bias_space: int,
        alpha: float
    ) -> None:

        self.word = word
        self.color = color
        self.bias_space = bias_space
        self.alpha = alpha


class WordExplorer:
    def __init__(
        self,
        embedding,      # Embedding Class instance
        errorManager    # ErrorManager class instance
    ) -> None:

        self.embedding = embedding
        self.errorManager = errorManager

    def __errorChecking(
        self,
        word: str
    ) -> str:

        out_msj = ""

        if not word:
            out_msj = ['EMBEDDING_NO_WORD_PROVIDED']
        else:
            if word not in self.embedding:
                out_msj = ['EMBEDDING_WORD_OOV', word]

        return self.errorManager.process(out_msj)

    def check_oov(
        self, 
        wordlists: List[List[str]]
    ) -> str:

        for wordlist in wordlists:
            for word in wordlist:
                msg = self.__errorChecking(word)
                if msg:
                    return msg
        return None

    def get_neighbors(
        self, 
        word: str, 
        n_neighbors: int, 
        nn_method: str
    ) -> List[str]:

        err = self.check_oov([[word]])
        if err:
            raise ValueError(err)
        
        return self.embedding.getNearestNeighbors(word, n_neighbors, nn_method)

    def get_df(
        self, 
        words_embedded: np.ndarray, 
        processed_word_list: List[str]
    ) -> pd.DataFrame:

        df = pd.DataFrame(words_embedded)

        df['word'] = [wtp.word for wtp in processed_word_list]
        df['color'] = [wtp.color for wtp in processed_word_list]
        df['alpha'] = [wtp.alpha for wtp in processed_word_list]
        df['word_bias_space'] = [wtp.bias_space for wtp in processed_word_list]
        return df

    def get_plot(
        self,
        data: pd.DataFrame,
        processed_word_list: List[str],
        words_embedded: np.ndarray,
        color_dict: Dict,
        n_neighbors: int,
        n_alpha: float,
        fontsize: int=18,
        figsize: Tuple[int, int]=(20, 15)
    ):

        fig, ax = plt.subplots(figsize=figsize)

        sns.scatterplot(
            data=data[data['alpha'] == 1],
            x=0,
            y=1,
            style='word_bias_space',
            hue='word_bias_space',
            ax=ax,
            palette=color_dict
        )

        if n_neighbors > 0:
            sns.scatterplot(
                data=data[data['alpha'] != 1],
                x=0,
                y=1,
                style='color',
                hue='word_bias_space',
                ax=ax,
                alpha=n_alpha,
                legend=False,
                palette=color_dict
            )
            
        for i, wtp in enumerate(processed_word_list):
            x, y = words_embedded[i, :]
            ax.annotate(
                wtp.word, 
                xy=(x, y), 
                xytext=(5, 2), 
                color=wtp.color,
                textcoords='offset points',
                ha='right', 
                va='bottom', 
                size=fontsize, 
                alpha=wtp.alpha
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        fig.tight_layout()

        return fig

    def plot_projections_2d(
        self,
        wordlist_0: List[str],
        wordlist_1: List[str]=[],
        wordlist_2: List[str]=[],
        wordlist_3: List[str]=[],
        wordlist_4: List[str]=[],
        **kwargs
    ):

        # convertirlas a vector
        choices = [0, 1, 2, 3, 4]
        wordlist_choice = [
            wordlist_0,
            wordlist_1,
            wordlist_2,
            wordlist_3,
            wordlist_4
        ]

        err = self.check_oov(wordlist_choice)
        if err:
            raise ValueError(err)

        color_dict = {
            0: kwargs.get('color_wordlist_0', '#000000'),
            1: kwargs.get('color_wordlist_1', '#1f78b4'),
            2: kwargs.get('color_wordlist_2', '#33a02c'),
            3: kwargs.get('color_wordlist_3', '#e31a1c'),
            4: kwargs.get('color_wordlist_4', '#6a3d9a')
        }

        n_neighbors = kwargs.get('n_neighbors', 0)
        n_alpha = kwargs.get('n_alpha', 0.3)

        processed_word_list = []
        for word_list_to_process, color in zip(wordlist_choice, choices):
            for word in word_list_to_process:
                processed_word_list.append(
                    WordToPlot(word, color_dict[color], color, 1)
                )

                if n_neighbors > 0:
                    neighbors = self.get_neighbors(
                        word,
                        n_neighbors=n_neighbors,
                        nn_method=kwargs.get('nn_method', 'sklearn')
                    )

                    for n in neighbors:
                        if n not in [wtp.word for wtp in processed_word_list]:
                            processed_word_list.append(
                                WordToPlot(n, color_dict[color], color, n_alpha)
                            )

        if not processed_word_list:
            raise ValueError(self.errorManager.process(['WORDEXPLORER_ONLY_EMPTY_LISTS']))

        words_embedded = np.array(
            [self.embedding.getPCA(wtp.word) for wtp in processed_word_list]
        )

        data = self.get_df(
            words_embedded, 
            processed_word_list
        )

        fig = self.get_plot(
            data, 
            processed_word_list, 
            words_embedded,
            color_dict, 
            n_neighbors, 
            n_alpha,
            kwargs.get('fontsize', 18),
            kwargs.get('figsize', (20, 15))
        )

        plt.show()
        return fig

    def doesnt_match(
        self, 
        wordlist: List[str]
    ) -> str:

        err = self.check_oov([wordlist])
        if err:
            raise ValueError(err)

        words_emb = np.array([self.embedding.getEmbedding(word)
                             for word in wordlist])
        mean_vec = np.mean(words_emb, axis=0)

        doesnt_match = ""
        farthest_emb = 1.0
        for word in wordlist:
            word_emb = self.embedding.getEmbedding(word)
            cos_sim = np.dot(mean_vec, word_emb) / (norm(mean_vec)*norm(word_emb))
            if cos_sim <= farthest_emb:
                farthest_emb = cos_sim
                doesnt_match = word

        return doesnt_match
