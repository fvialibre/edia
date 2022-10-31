import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

    def process_word_to_plot(self, 
                            word,
                            word_list, 
                            color, 
                            word_bias_space, 
                            words_colors, 
                            color_dict, 
                            alpha, 
                            alpha_value = 1,
                            n_value=0.2,
                            n_neighbors = 0
                            ):
        word_bias_space[word] = color
        words_colors[word] = color_dict[color]
        alpha[word] = alpha_value

        if n_neighbors > 0:
            neighbors = self.get_neighbors(word, n_neighbors=n_neighbors)
            for n in neighbors:
                if n not in alpha:
                    self.process_word_to_plot(n, word_list, color, word_bias_space, words_colors, color_dict, alpha, alpha_value=n_value)
            word_list += neighbors

    def get_df(self, words_embedded, word_list, words_colors, alpha, word_bias_space):
        df = pd.DataFrame(words_embedded)
        df['word'] = word_list
        df['color'] = [words_colors[word] for word in word_list]
        df['alpha'] = [alpha[word] for word in word_list]
        df['word_bias_space'] = [word_bias_space[word] for word in word_list]
        return df

    def get_plot(self,
                 data, 
                 word_list, 
                 words_embedded, 
                 words_colors, 
                 color_dict, 
                 alpha, 
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
        for i, label in enumerate(word_list):
            x, y = words_embedded[i, :]
            ax.annotate(label, xy=(x, y), xytext=(5, 2), color=words_colors[label],
                        textcoords='offset points',
                        ha='right', va='bottom', size=fontsize, alpha=alpha[label])

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

        err = self.check_oov(wordlist_choice)
        if err:
            return None, err

        color_dict = {
            0: kwargs.get('color_wordlist_0', '#000000'),
            1: kwargs.get('color_wordlist_1', '#1f78b4'),
            2: kwargs.get('color_wordlist_2', '#33a02c'),
            3: kwargs.get('color_wordlist_3', '#e31a1c'),
            4: kwargs.get('color_wordlist_4', '#6a3d9a')
        }
        words_colors = {}
        word_bias_space = {}
        alpha = {}

        n_neighbors = kwargs.get('n_neighbors', 0)
        n_alpha = kwargs.get('n_alpha', 0.3)

        word_list = []
        for word_list_to_process, color in zip(wordlist_choice, choices):
            for word in word_list_to_process:
                self.process_word_to_plot(word, word_list, color, word_bias_space, words_colors, color_dict, alpha, 1, n_alpha, n_neighbors)
                
            word_list += word_list_to_process

        if not word_list:
            return None, "<center><h3>" + "Ingresa al menos 2 palabras para continuar" + "<center><h3>"
        
        words_embedded = np.array([self.vocabulary.getPCA(word) for word in word_list])

        data = self.get_df(words_embedded, word_list, words_colors, alpha, word_bias_space)
        fig = self.get_plot(data, word_list, words_embedded, words_colors, 
                            color_dict, alpha, n_neighbors, n_alpha, 
                            kwargs.get('fontsize', 18), 
                            kwargs.get('figsize', (15, 15))
                            )
        plt.show()
        return fig
