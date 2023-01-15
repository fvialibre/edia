import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Any
from modules.utils import normalize, cosine_similarity, project_params, take_two_sides_extreme_sorted

__all__ = ['WordBiasExplorer', 'WEBiasExplorer2Spaces', 'WEBiasExplorer4Spaces']

class WordBiasExplorer:
    def __init__(
        self, 
        embedding,      # Embedding class instance
        errorManager    # ErrorManager class instance
    ) -> None:

        self.embedding = embedding
        self.direction = None
        self.positive_end = None
        self.negative_end = None
        self.DIRECTION_METHODS = ['single', 'sum', 'pca']
        self.errorManager = errorManager

    def __copy__(
        self
    ) -> 'WordBiasExplorer':

        bias_word_embedding = self.__class__(self.embedding)
        bias_word_embedding.direction = copy.deepcopy(self.direction)
        bias_word_embedding.positive_end = copy.deepcopy(self.positive_end)
        bias_word_embedding.negative_end = copy.deepcopy(self.negative_end)
        return bias_word_embedding

    def __deepcopy__(
        self, 
        memo: Optional[Dict[int, Any]]
    )-> 'WordBiasExplorer':

        bias_word_embedding = copy.copy(self)
        bias_word_embedding.model = copy.deepcopy(bias_word_embedding.model)
        return bias_word_embedding

    def __getitem__(
        self, 
        key: str
    ) -> np.ndarray:

        return self.embedding.getEmbedding(key)

    def __contains__(
        self, 
        item: str
    ) -> bool:

        return item in self.embedding

    def _is_direction_identified(
        self
    ):
        if self.direction is None:
            raise RuntimeError('The direction was not identified'
                               ' for this {} instance'
                               .format(self.__class__.__name__))

    def _identify_subspace_by_pca(
        self, 
        definitional_pairs: List[Tuple[str, str]], 
        n_components: int
    ) -> PCA:

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


    def _identify_direction(
        self, 
        positive_end: str, 
        negative_end: str,
        definitional: Tuple[str, str], 
        method: str='pca',
        first_pca_threshold: float=0.5
    ) -> None:

        if method not in self.DIRECTION_METHODS:
            raise ValueError('method should be one of {}, {} was given'.format(
                self.DIRECTION_METHODS, method))

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
            if pca.explained_variance_ratio_[0] < first_pca_threshold:
                raise RuntimeError('The Explained variance'
                                   'of the first principal component should be'
                                   'at least {}, but it is {}'
                                   .format(first_pca_threshold,
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

    def project_on_direction(
        self, 
        word: str
    ) -> float:

        """Project the normalized vector of the word on the direction.
        :param str word: The word tor project
        :return float: The projection scalar
        """

        self._is_direction_identified()

        vector = self[word]
        projection_score = self.embedding.cosineSimilarities(self.direction,
                                                          [vector])[0]
        return projection_score

    def _calc_projection_scores(
        self, 
        words: List[str]
    ) -> pd.DataFrame:

        self._is_direction_identified()

        df = pd.DataFrame({'word': words})

        # TODO: maybe using cosine_similarities on all the vectors?
        # it might be faster
        df['projection'] = df['word'].apply(self.project_on_direction)
        df = df.sort_values('projection', ascending=False)

        return df

    def calc_projection_data(
        self, 
        words: List[str]
    ) -> pd.DataFrame:

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

    def plot_dist_projections_on_direction(
        self, 
        word_groups: Dict[str, List[str]], 
        ax: plt.Axes=None
    ) -> plt.Axes:

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
            projections = self.embedding.cosineSimilarities(self.direction,
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
        wordlists: List[str]
    ) -> str:

        for wordlist in wordlists:
            for word in wordlist:
                msg = self.__errorChecking(word)
                if msg:
                    return msg
        return None
    
class WEBiasExplorer2Spaces(WordBiasExplorer):
    def __init__(
        self, 
        embedding,      # Embedding class instance
        errorManager    # ErrorManager class instance
    ) -> None:

        super().__init__(embedding, errorManager)

    def calculate_bias(
        self,
        wordlist_to_diagnose: List[str],
        wordlist_right: List[str],
        wordlist_left: List[str]
    ) -> plt.Figure:

        wordlists = [wordlist_to_diagnose, wordlist_right, wordlist_left] 
        
        for wordlist in wordlists:
            if not wordlist:
                raise Exception('At least one word should be in the to diagnose list, bias 1 list and bias 2 list')
        
        err = self.check_oov(wordlists)
        if err:
            raise Exception(err)

        return self.get_bias_plot(
                wordlist_to_diagnose,
                definitional=(wordlist_left, wordlist_right),
                method='sum',
                n_extreme=10
            )

    def get_bias_plot(
        self,
        wordlist_to_diagnose: List[str],
        definitional: Tuple[List[str], List[str]],
        method: str='sum',
        n_extreme: int=10,
        figsize: Tuple[int, int]=(10, 10)
    ) -> plt.Figure:

        fig, ax = plt.subplots(1, figsize=figsize)
        self.method = method
        self.plot_projection_scores(
            definitional,
            wordlist_to_diagnose, n_extreme, ax=ax,)

        fig.tight_layout()
        fig.canvas.draw()

        return fig

    def plot_projection_scores(
        self, 
        definitional: Tuple[List[str], List[str]],
        words: List[str], 
        n_extreme: int=10,
        ax: plt.Axes=None, 
        axis_projection_step: float=None
    ) -> plt.Axes:

        """Plot the projection scalar of words on the direction.
        :param list words: The words tor project
        :param int or None n_extreme: The number of extreme words to show
        :return: The ax object of the plot
        """
        name_left = ', '.join(definitional[0])
        name_right = ', '.join(definitional[1])

        self._identify_direction(name_left, name_right,
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


class WEBiasExplorer4Spaces(WordBiasExplorer):
    def __init__(
        self, 
        embedding,      # Embedding Class instance
        errorManager    # ErrorManager class instance
    ) -> None:

        super().__init__(embedding, errorManager)

    def calculate_bias(
        self,
        wordlist_to_diagnose: List[str],
        wordlist_right: List[str],
        wordlist_left: List[str],
        wordlist_top: List[str],
        wordlist_bottom: List[str],
    ) -> plt.Figure:

        wordlists = [
            wordlist_to_diagnose,
            wordlist_left,
            wordlist_right,
            wordlist_top,
            wordlist_bottom
        ]

        for wordlist in wordlists:
            if not wordlist:
                raise Exception('To plot with 4 spaces, you must enter at least one word in all lists')

        err = self.check_oov(wordlists)
        if err:
            raise Exception(err)

        return self.get_bias_plot(
                wordlist_to_diagnose,
                definitional_1=(wordlist_right, wordlist_left),
                definitional_2=(wordlist_top, wordlist_bottom),
                method='sum',
                n_extreme=10
            )

    def get_bias_plot(
        self,
        wordlist_to_diagnose: List[str],
        definitional_1: Tuple[List[str], List[str]],
        definitional_2: Tuple[List[str], List[str]],
        method: str='sum',
        n_extreme: int=10,
        figsize: Tuple[int, int]=(10, 10)
    ) -> plt.Figure:

        fig, ax = plt.subplots(1, figsize=figsize)
        self.method = method
        self.plot_projection_scores(
            definitional_1,
            definitional_2,
            wordlist_to_diagnose, n_extreme, ax=ax,)
        fig.canvas.draw()

        return fig

    def plot_projection_scores(
        self, 
        definitional_1: Tuple[List[str], List[str]], 
        definitional_2: Tuple[List[str], List[str]],
        words: List[str], 
        n_extreme: int=10,
        ax: plt.Axes=None, 
        axis_projection_step: float=None
    ) -> plt.Axes:

        """Plot the projection scalar of words on the direction.
        :param list words: The words tor project
        :param int or None n_extreme: The number of extreme words to show
        :return: The ax object of the plot
        """

        name_left = ', '.join(definitional_1[1])
        name_right = ', '.join(definitional_1[0])

        self._identify_direction(name_left, name_right,
                                 definitional=definitional_1,
                                 method='sum')

        self._is_direction_identified()

        projections_df = self._calc_projection_scores(words)
        projections_df['projection_x'] = projections_df['projection'].round(2)

        name_top = ', '.join(definitional_2[1])
        name_bottom = ', '.join(definitional_2[0])
        self._identify_direction(name_top, name_bottom,
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
            decimals=1
        )
        
        sns.scatterplot(x='projection_x', 
                        y='projection_y', 
                        data=projections_df,
                        # color=list(projections_df['color'].to_list()), # No se distinguen los colores
                        color='blue'
        )

        plt.xticks(np.arange(-most_extream_projection,
                             most_extream_projection + axis_projection_step,
                             axis_projection_step))
        for _, row in (projections_df.iterrows()):
            ax.annotate(
                row['word'], (row['projection_x'], row['projection_y']))
        x_label = '← {} {} {} →'.format(name_left,
                                        ' ' * 20,
                                        name_right)

        y_label = '← {} {} {} →'.format(name_top,
                                        ' ' * 20,
                                        name_bottom)

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

        return ax
