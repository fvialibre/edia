import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

class WordExplorer:
    def __init__(self,vocabulary) -> None:
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
            parsed_words = self.parse_words(wordlist)
            for word in parsed_words:
                msg = self.__errorChecking(word)
                if msg:
                    return msg
        return None

    def plot_projections_2d(self,
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
                            plot_neighbors,
                            n_alpha,
                            fontsize,
                            n_neighbors,
                            figsize=(15, 15),
                            ):
        # convertirlas a vector
        choices = [0, 1, 2, 3, 4]
        word_list = []
        wordlist_choice = [wordlist, wordlist_1,
                           wordlist_2, wordlist_3, wordlist_4]
        err = self.check_oov(wordlist_choice)
        if err:
            return None, err
        words_colors = {}
        color_dict = {
            0: color_wordlist,
            1: color_wordlist_1,
            2: color_wordlist_2,
            3: color_wordlist_3,
            4: color_wordlist_4
        }
        word_bias_space = {}
        alpha = {}

        for raw_word_list, color in zip(wordlist_choice, choices):
            parsed_words = self.parse_words(raw_word_list)
            if parsed_words:
                for word in parsed_words:
                    word_bias_space[word] = color
                    words_colors[word] = color_dict[color]
                    alpha[word] = 1
                    if plot_neighbors:
                        neighbors = self.vocabulary.getNearestNeighbors(word, n_neighbors=n_neighbors)
                        for n in neighbors:
                            if n not in alpha:
                                word_bias_space[n] = color
                                words_colors[n] = color_dict[color]
                                alpha[n] = n_alpha
                        word_list += neighbors
            word_list += parsed_words
        if not word_list:
            return None, "<center><h3>" + "Ingresa al menos 2 palabras para continuar" + "<center><h3>"
        words_embedded = np.array([self.vocabulary.getPCA(word)
                                   for word in word_list])
        data = pd.DataFrame(words_embedded)
        data['word'] = word_list
        data['color'] = [words_colors[word] for word in word_list]
        data['alpha'] = [alpha[word] for word in word_list]
        data['word_bias_space'] = [word_bias_space[word] for word in word_list]
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
        if plot_neighbors:
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

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        return im, ''
