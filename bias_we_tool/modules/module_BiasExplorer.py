import copy

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def take_two_sides_extreme_sorted(df, n_extreme,
                                  part_column=None,
                                  head_value='',
                                  tail_value=''):
    head_df = df.head(n_extreme)[:]
    tail_df = df.tail(n_extreme)[:]

    if part_column is not None:
        head_df[part_column] = head_value
        tail_df[part_column] = tail_value

    return (pd.concat([head_df, tail_df])
            .drop_duplicates()
            .reset_index(drop=True))

def normalize(v):
    """Normalize a 1-D vector."""
    if v.ndim != 1:
        raise ValueError('v should be 1-D, {}-D was given'.format(
            v.ndim))
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def project_params(u, v):
    """Projecting and rejecting the vector v onto direction u with scalar."""
    normalize_u = normalize(u)
    projection = (v @ normalize_u)
    projected_vector = projection * normalize_u
    rejected_vector = v - projected_vector
    return projection, projected_vector, rejected_vector


def cosine_similarity(v, u):
    """Calculate the cosine similarity between two vectors."""
    v_norm = np.linalg.norm(v)
    u_norm = np.linalg.norm(u)
    similarity = v @ u / (v_norm * u_norm)
    return similarity


DIRECTION_METHODS = ['single', 'sum', 'pca']
DEBIAS_METHODS = ['neutralize', 'hard', 'soft']
FIRST_PC_THRESHOLD = 0.5
MAX_NON_SPECIFIC_EXAMPLES = 1000

__all__ = ['GenderBiasWE', 'BiasWordEmbedding']


class WordBiasExplorer():
    def __init__(self, vocabulary):
        # pylint: disable=undefined-variable

        self.vocabulary = vocabulary
        self.direction = None
        self.positive_end = None
        self.negative_end = None

    def __copy__(self):
        bias_word_embedding = self.__class__(self.vocabulary)
        bias_word_embedding.direction = copy.deepcopy(self.direction)
        bias_word_embedding.positive_end = copy.deepcopy(self.positive_end)
        bias_word_embedding.negative_end = copy.deepcopy(self.negative_end)
        return bias_word_embedding

    def __deepcopy__(self, memo):
        bias_word_embedding = copy.copy(self)
        bias_word_embedding.model = copy.deepcopy(bias_word_embedding.model)
        return bias_word_embedding

    def __getitem__(self, key):
        return self.vocabulary.getEmbedding(key)

    def __contains__(self, item):
        return item in self.vocabulary

    def _is_direction_identified(self):
        if self.direction is None:
            raise RuntimeError('The direction was not identified'
                               ' for this {} instance'
                               .format(self.__class__.__name__))

    def _identify_subspace_by_pca(self, definitional_pairs, n_components):
        matrix = []

        for word1, word2 in definitional_pairs:
            vector1 = normalize(self[word1])
            vector2 = normalize(self[word2])

            center = (vector1 + vector2) / 2

            matrix.append(vector1 - center)
            matrix.append(vector2 - center)

        pca = PCA(n_components=n_components)
        pca.fit(matrix)
        return pca


    def _identify_direction(self, positive_end, negative_end,
                            definitional, method='pca'):
        if method not in DIRECTION_METHODS:
            raise ValueError('method should be one of {}, {} was given'.format(
                DIRECTION_METHODS, method))

        if positive_end == negative_end:
            raise ValueError('positive_end and negative_end'
                             'should be different, and not the same "{}"'
                             .format(positive_end))
        direction = None

        if method == 'single':
            direction = normalize(normalize(self[definitional[0]])
                                  - normalize(self[definitional[1]]))

        elif method == 'sum':
            group1_sum_vector = np.sum([self[word]
                                        for word in definitional[0]], axis=0)
            group2_sum_vector = np.sum([self[word]
                                        for word in definitional[1]], axis=0)

            diff_vector = (normalize(group1_sum_vector)
                           - normalize(group2_sum_vector))

            direction = normalize(diff_vector)

        elif method == 'pca':
            pca = self._identify_subspace_by_pca(definitional, 10)
            if pca.explained_variance_ratio_[0] < FIRST_PC_THRESHOLD:
                raise RuntimeError('The Explained variance'
                                   'of the first principal component should be'
                                   'at least {}, but it is {}'
                                   .format(FIRST_PC_THRESHOLD,
                                           pca.explained_variance_ratio_[0]))
            direction = pca.components_[0]

            # if direction is opposite (e.g. we cannot control
            # what the PCA will return)
            ends_diff_projection = cosine_similarity((self[positive_end]
                                                      - self[negative_end]),
                                                     direction)
            if ends_diff_projection < 0:
                direction = -direction  # pylint: disable=invalid-unary-operand-type

        self.direction = direction
        self.positive_end = positive_end
        self.negative_end = negative_end

    def project_on_direction(self, word):
        """Project the normalized vector of the word on the direction.
        :param str word: The word tor project
        :return float: The projection scalar
        """

        self._is_direction_identified()

        vector = self[word]
        projection_score = self.vocabulary.cosineSimilarities(self.direction,
                                                          [vector])[0]
        return projection_score



    def _calc_projection_scores(self, words):
        self._is_direction_identified()

        df = pd.DataFrame({'word': words})

        # TODO: maybe using cosine_similarities on all the vectors?
        # it might be faster
        df['projection'] = df['word'].apply(self.project_on_direction)
        df = df.sort_values('projection', ascending=False)

        return df

    def calc_projection_data(self, words):
        """
        Calculate projection, projected and rejected vectors of a words list.
        :param list words: List of words
        :return: :class:`pandas.DataFrame` of the projection,
                 projected and rejected vectors of the words list
        """
        projection_data = []
        for word in words:
            vector = self[word]
            normalized_vector = normalize(vector)

            (projection,
             projected_vector,
             rejected_vector) = project_params(normalized_vector,
                                               self.direction)

            projection_data.append({'word': word,
                                    'vector': vector,
                                    'projection': projection,
                                    'projected_vector': projected_vector,
                                    'rejected_vector': rejected_vector})

        return pd.DataFrame(projection_data)

    def plot_dist_projections_on_direction(self, word_groups, ax=None):
        """Plot the projection scalars distribution on the direction.
        :param dict word_groups word: The groups to projects
        :return float: The ax object of the plot
        """

        if ax is None:
            _, ax = plt.subplots(1)

        names = sorted(word_groups.keys())

        for name in names:
            words = word_groups[name]
            label = '{} (#{})'.format(name, len(words))
            vectors = [self[word] for word in words]
            projections = self.vocabulary.cosineSimilarities(self.direction,
                                                         vectors)
            sns.distplot(projections, hist=False, label=label, ax=ax)

        plt.axvline(0, color='k', linestyle='--')

        plt.title('← {} {} {} →'.format(self.negative_end,
                                        ' ' * 20,
                                        self.positive_end))
        plt.xlabel('Direction Projection')
        plt.ylabel('Density')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return ax

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

    def check_oov(self, wordlists):
        for wordlist in wordlists:
            for word in wordlist:
                msg = self.__errorChecking(word)
                if msg:
                    return msg
        return None
    
    def plot_biased_words(self,
                       words_to_diagnose,
                       wordlist_right,
                       wordlist_left,
                       wordlist_top=[],
                       wordlist_bottom=[]
                       ):
        bias_2D = wordlist_top == [] and wordlist_bottom == []

        err = self.check_oov([words_to_diagnose + wordlist_right + wordlist_left + wordlist_top + wordlist_bottom])
        if bias_2D and (not wordlist_right or not wordlist_left):
            err = ""
        elif not bias_2D and (not wordlist_right or not wordlist_left or not wordlist_top or not wordlist_bottom):
            err = ""

        if err:
            return None, err

        return self.get_bias_plot(bias_2D,
                                  words_to_diagnose,
                                  definitional_1=(wordlist_right, wordlist_left),
                                  definitional_2=(wordlist_top, wordlist_bottom)
                                  )

    def get_bias_plot(self,
                      plot_2D,
                      words_to_diagnose,
                      definitional_1,
                      definitional_2=([], []),
                      method='sum',
                      n_extreme=10,
                      figsize=(10, 10)
                      ):
        fig, ax = plt.subplots(1, figsize=figsize)
        self.method = method
        self.plot_projection_scores(plot_2D, words_to_diagnose, definitional_1, definitional_2, n_extreme, ax)
        
        if plot_2D:
            fig.tight_layout()
        fig.canvas.draw()

        return fig

    def plot_projection_scores(self,
                                  plot_2D,
                                  words,
                                  definitional_1,
                                  definitional_2=([], []),
                                  n_extreme=10,
                                  ax=None,
                                  axis_projection_step=0.1):
        name_left  = ', '.join(definitional_1[1])
        name_right = ', '.join(definitional_1[0])

        self._identify_direction(name_left, name_right, definitional=definitional_1, method='sum')
        self._is_direction_identified()

        projections_df = self._calc_projection_scores(words)
        projections_df['projection_x'] = projections_df['projection'].round(2)

        if not plot_2D:
            name_top    = ', '.join(definitional_2[1])
            name_bottom = ', '.join(definitional_2[0])
            self._identify_direction(name_top, name_bottom, definitional=definitional_2, method='sum')
            self._is_direction_identified()

            projections_df['projection_y'] = self._calc_projection_scores(words)['projection'].round(2)

        if n_extreme is not None:
            projections_df = take_two_sides_extreme_sorted(projections_df, n_extreme=n_extreme)
        
        if ax is None:
            _, ax = plt.subplots(1)
        
        cmap = plt.get_cmap('RdBu')
        projections_df['color'] = ((projections_df['projection'] + 0.5).apply(cmap))
        most_extream_projection = np.round(
            projections_df['projection']
            .abs()
            .max(),
            decimals=1)
        
        if plot_2D:
            sns.barplot(x='projection', y='word', data=projections_df,
                    palette=projections_df['color'])
        else:
            sns.scatterplot(x='projection_x', y='projection_y', data=projections_df,
                        palette=projections_df['color'])
        
        plt.xticks(np.arange(-most_extream_projection,
                             most_extream_projection + axis_projection_step,
                             axis_projection_step))

        x_label = '← {} {} {} →'.format(name_left,
                                        ' ' * 20,
                                        name_right)
        if not plot_2D:
            y_label = '← {} {} {} →'.format(name_top,
                                        ' ' * 20,
                                        name_bottom)
            for _, row in (projections_df.iterrows()):
                ax.annotate(row['word'], (row['projection_x'], row['projection_y']))
        
        plt.xlabel(x_label)
        plt.ylabel('Words')

        if not plot_2D:
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.set_label_coords(.5, 0)

            plt.ylabel(y_label)
            ax.yaxis.set_label_position('left')
            ax.yaxis.set_label_coords(0, .5)

            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')

            ax.set_xticks([])
            ax.set_yticks([])

        return ax
        
# Would be erased if decided to keep all info in BiasWordExplorer
class WEBiasExplorer2d(WordBiasExplorer):
    def __init__(self, word_embedding) -> None:
        super().__init__(word_embedding)

    def calculate_bias( self,
                        palabras_extremo_1,
                        palabras_extremo_2,
                        palabras_para_situar
                        ):
        wordlists = [palabras_extremo_1, palabras_extremo_2, palabras_para_situar] 
        
        err = self.check_oov(wordlists)
        for wordlist in wordlists:
            if not wordlist:
                err = "<center><h3>" + 'Debe ingresar al menos 1 palabra en las lista de palabras a diagnosticar, sesgo 1 y sesgo 2' + "<center><h3>"
        if err:
            return None, err

        im = self.get_bias_plot(
            palabras_para_situar,
            definitional=(
                palabras_extremo_1, palabras_extremo_2),
            method='sum',
            n_extreme=10
        )
        return im, ''

    def get_bias_plot(self,
                      palabras_para_situar,
                      definitional,
                      method='sum',
                      n_extreme=10,
                      figsize=(10, 10)
                      ):

        fig, ax = plt.subplots(1, figsize=figsize)
        self.method = method
        self.plot_projection_scores(
            definitional,
            palabras_para_situar, n_extreme, ax=ax,)

        fig.tight_layout()
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        return im

    def plot_projection_scores(self, definitional,
                               words, n_extreme=10,
                               ax=None, axis_projection_step=None):
        """Plot the projection scalar of words on the direction.
        :param list words: The words tor project
        :param int or None n_extreme: The number of extreme words to show
        :return: The ax object of the plot
        """
        nombre_del_extremo_1 = ', '.join(definitional[0])
        nombre_del_extremo_2 = ', '.join(definitional[1])

        self._identify_direction(nombre_del_extremo_1, nombre_del_extremo_2,
                                 definitional=definitional,
                                 method='sum')

        self._is_direction_identified()

        projections_df = self._calc_projection_scores(words)
        projections_df['projection'] = projections_df['projection'].round(2)

        if n_extreme is not None:
            projections_df = take_two_sides_extreme_sorted(projections_df,
                                                           n_extreme=n_extreme)

        if ax is None:
            _, ax = plt.subplots(1)

        if axis_projection_step is None:
            axis_projection_step = 0.1

        cmap = plt.get_cmap('RdBu')
        projections_df['color'] = ((projections_df['projection'] + 0.5)
                                   .apply(cmap))

        most_extream_projection = np.round(
            projections_df['projection']
            .abs()
            .max(),
            decimals=1)

        sns.barplot(x='projection', y='word', data=projections_df,
                    palette=projections_df['color'])

        plt.xticks(np.arange(-most_extream_projection,
                             most_extream_projection + axis_projection_step,
                             axis_projection_step))
        xlabel = ('← {} {} {} →'.format(self.negative_end,
                                        ' ' * 20,
                                        self.positive_end))

        plt.xlabel(xlabel)
        plt.ylabel('Words')

        return ax


class WEBiasExplorer4d(WordBiasExplorer):
    def __init__(self, word_embedding) -> None:
        super().__init__(word_embedding)

    def calculate_bias( self,
                        palabras_extremo_1,
                        palabras_extremo_2,
                        palabras_extremo_3,
                        palabras_extremo_4,
                        palabras_para_situar
                        ):
        wordlists = [
            palabras_extremo_1,
            palabras_extremo_2,
            palabras_extremo_3,
            palabras_extremo_4,
            palabras_para_situar
        ]
        for wordlist in wordlists:
            if not wordlist:
                err = "<center><h3>" + \
                    '¡Para graficar con 4 espacios, debe ingresar al menos 1 palabra en todas las listas!' + "<center><h3>"

        err = self.check_oov(wordlist)

        if err:
            return None, err

        im = self.get_bias_plot(
            palabras_para_situar,
            definitional_1=(
                palabras_extremo_1, palabras_extremo_2),
            definitional_2=(
                palabras_extremo_3, palabras_extremo_4),
            method='sum',
            n_extreme=10
        )
        return im, ''

    def get_bias_plot(self,
                      palabras_para_situar,
                      definitional_1,
                      definitional_2,
                      method='sum',
                      n_extreme=10,
                      figsize=(10, 10)
                      ):

        fig, ax = plt.subplots(1, figsize=figsize)
        self.method = method
        self.plot_projection_scores(
            definitional_1,
            definitional_2,
            palabras_para_situar, n_extreme, ax=ax,)
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        return im

    def plot_projection_scores(self, definitional_1, definitional_2,
                               words, n_extreme=10,
                               ax=None, axis_projection_step=None):
        """Plot the projection scalar of words on the direction.
        :param list words: The words tor project
        :param int or None n_extreme: The number of extreme words to show
        :return: The ax object of the plot
        """

        nombre_del_extremo_1 = ', '.join(definitional_1[1])
        nombre_del_extremo_2 = ', '.join(definitional_1[0])

        self._identify_direction(nombre_del_extremo_1, nombre_del_extremo_2,
                                 definitional=definitional_1,
                                 method='sum')

        self._is_direction_identified()

        projections_df = self._calc_projection_scores(words)
        projections_df['projection_x'] = projections_df['projection'].round(2)

        nombre_del_extremo_3 = ', '.join(definitional_2[1])
        nombre_del_extremo_4 = ', '.join(definitional_2[0])
        self._identify_direction(nombre_del_extremo_3, nombre_del_extremo_4,
                                 definitional=definitional_2,
                                 method='sum')

        self._is_direction_identified()

        projections_df['projection_y'] = self._calc_projection_scores(words)[
            'projection'].round(2)

        if n_extreme is not None:
            projections_df = take_two_sides_extreme_sorted(projections_df,
                                                           n_extreme=n_extreme)

        if ax is None:
            _, ax = plt.subplots(1)

        if axis_projection_step is None:
            axis_projection_step = 0.1

        cmap = plt.get_cmap('RdBu')
        projections_df['color'] = ((projections_df['projection'] + 0.5)
                                   .apply(cmap))
        most_extream_projection = np.round(
            projections_df['projection']
            .abs()
            .max(),
            decimals=1)
        sns.scatterplot(x='projection_x', y='projection_y', data=projections_df,
                        palette=projections_df['color'])

        plt.xticks(np.arange(-most_extream_projection,
                             most_extream_projection + axis_projection_step,
                             axis_projection_step))
        for _, row in (projections_df.iterrows()):
            ax.annotate(
                row['word'], (row['projection_x'], row['projection_y']))
        x_label = '← {} {} {} →'.format(nombre_del_extremo_1,
                                        ' ' * 20,
                                        nombre_del_extremo_2)

        y_label = '← {} {} {} →'.format(nombre_del_extremo_3,
                                        ' ' * 20,
                                        nombre_del_extremo_4)

        plt.xlabel(x_label)
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_label_coords(.5, 0)

        plt.ylabel(y_label)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_label_coords(0, .5)

        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        ax.set_xticks([])
        ax.set_yticks([])
        #plt.yticks([], [])
        # ax.spines['left'].set_position('zero')
        # ax.spines['bottom'].set_position('zero')

        return ax
