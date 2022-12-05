from wordcloud import WordCloud
import matplotlib.pyplot as plt


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {
            word: color
            for (color, words) in color_to_words.items()
            for word in words
        }

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class SegmentedWordCloud:
    def __init__(self, freq_dic, less_group, greater_group):
        colors = {
            'less': '#529ef3',
            'salient':'#d35400',
            'greater':'#5d6d7e',
        }

        color_to_words = {
            colors['greater']: greater_group,
            colors['less']: less_group,
        }
        

        grouped_color_func = SimpleGroupedColorFunc(
            color_to_words=color_to_words, 
            default_color=colors['salient']
        )

        self.wc = WordCloud(
            background_color="white", 
            width=900, 
            height=300,
            random_state=None).generate_from_frequencies(freq_dic)
        
        self.wc.recolor(color_func=grouped_color_func)

    def plot(self, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.wc, interpolation="bilinear")
        ax.axis("off")
        fig.tight_layout()
        return fig