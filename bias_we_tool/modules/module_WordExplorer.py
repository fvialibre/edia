import numpy as np
from numpy.linalg import norm
import seaborn as sns
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class WordToPlot:
    def __init__(self, word, color, bias_space, alpha):
        self.word = word
        self.color = color
        self.bias_space = bias_space
        self.alpha = alpha

class WordExplorer:
    def __init__(self, vocabulary) -> None:
        self.vocabulary = vocabulary

    def __errorChecking(self, word):
        out_msj = ""

        if not word:
            out_msj = "Error: Primero debe ingresar una palabra!"
        else:
            if word not in self.vocabulary:
                out_msj = f"Error: La palabra '<b>{word}</b>' no se encuentra en el vocabulario!"

        if out_msj:
            out_msj = "<center><h3>"+out_msj+"</h3></center>"

        return out_msj


    def parse_words(self, string):
        words = string.strip()
        if words:
            words = [word.strip() for word in words.split(',') if word != ""]
        return words

    def check_oov(self, wordlists):
        for wordlist in wordlists:
            for word in wordlist:
                msg = self.__errorChecking(word)
                if msg:
                    return msg
        return None

    def get_neighbors(self, word, n_neighbors=5):
        return self.vocabulary.ann.get(word, n_neighbors=n_neighbors)

    def get_df(self, words_embedded, processed_word_list):
        df = pd.DataFrame(words_embedded)

        df['word'] = [wtp.word for wtp in processed_word_list]
        df['color'] = [wtp.color for wtp in processed_word_list]
        df['alpha'] = [wtp.alpha for wtp in processed_word_list]
        df['word_bias_space'] = [wtp.bias_space for wtp in processed_word_list]
        return df

    def get_plot(self,
                 data, 
                 processed_word_list, 
                 words_embedded,
                 color_dict,
                 n_neighbors, 
                 n_alpha, 
                 fontsize=18, 
                 figsize=(15, 15)
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
            ax.annotate(wtp.word, xy=(x, y), xytext=(5, 2), color=wtp.color,
                        textcoords='offset points',
                        ha='right', va='bottom', size=fontsize, alpha=wtp.alpha)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        fig.tight_layout()
        fig.canvas.draw()

        return fig

    def plot_projections_2d(self,
                            wordlist_0,
                            wordlist_1 = [],
                            wordlist_2 = [],
                            wordlist_3 = [],
                            wordlist_4 = [],
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

        if self.check_oov(wordlist_choice):
            raise Exception('Word not in vocabulary')

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
                processed_word_list.append(WordToPlot(word, color_dict[color], color, 1))

                if n_neighbors > 0:
                    neighbors = self.get_neighbors(word, n_neighbors=n_neighbors+1)
                    for n in neighbors:
                        if n not in [wtp.word for wtp in processed_word_list]:
                            processed_word_list.append(WordToPlot(n, color_dict[color], color, n_alpha))

        if not processed_word_list:
            raise Exception('Only empty lists were passed')
        
        words_embedded = np.array([self.vocabulary.getPCA(wtp.word) for wtp in processed_word_list])

        data = self.get_df(words_embedded, processed_word_list)

        fig = self.get_plot(data, processed_word_list, words_embedded, 
                            color_dict, n_neighbors, n_alpha, 
                            kwargs.get('fontsize', 18), 
                            kwargs.get('figsize', (15, 15))
                            )
        plt.show()
        return fig

    def doesnt_match(self, wordlist):
        err = self.check_oov([wordlist])
        if err:
            raise Exception(err)
        
        words_emb = np.array([self.vocabulary.getEmbedding(word) for word in wordlist])
        mean_vec = np.mean(words_emb, axis=0)

        doesnt_match = ""
        farthest_emb = 1.0
        for word in wordlist:
            word_emb = self.vocabulary.getEmbedding(word)
            cos_sim = np.dot(mean_vec, word_emb) / (norm(mean_vec)*norm(word_emb))
            if cos_sim <= farthest_emb:
                farthest_emb = cos_sim
                doesnt_match = word

        return doesnt_match
