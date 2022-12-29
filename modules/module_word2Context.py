from datasets import load_dataset, interleave_datasets
from modules.module_segmentedWordCloud import SegmentedWordCloud
from modules.module_customSubsetsLabel import CustomSubsetsLabel
from random import sample as random_sample
from typing import Tuple, List, Dict
import re

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class Word2Context:
    def __init__(
        self, 
        context_ds_name: str, 
        vocabulary  # Vocabulary class instance
    ) -> None:

        self.context_ds_name = context_ds_name
        
        # Vocabulary class 
        self.vocab = vocabulary

        # Custom Label component
        self.Label = CustomSubsetsLabel()

    def errorChecking(
        self, 
        word: str
    ) -> str:

        out_msj = ""

        if not word:
            out_msj = "Error: Primero debe ingresar una palabra!"
        else:
            if word not in self.vocab:
                out_msj = f"Error: La palabra '<b>{word}</b>' no se encuentra en el vocabulario!"
        
        return out_msj

    def genWebLink(
        self,
        text: str
    ) -> str:

        text = text.replace("\"", "'")
        text = text.replace("<u><b>", "")
        text = text.replace("</b></u>", "")
        url = "https://www.google.com.tr/search?q={}".format(text)
        return '<a href="{}" rel="noopener noreferrer" target="_blank"><center>🌐🔍</center></a>'.format(url)

    def genWordCloudPlot(
        self, 
        word: str, 
        figsize: Tuple[int,int]=(9,3)
    ) -> plt.Figure:

        err = self.errorChecking(word)
        if err:
            raise Exception(err)

        freq_dic, l_group, g_group = self.vocab.getWordNeighbors(word, n_neighbors=10)
        wc = SegmentedWordCloud(freq_dic, l_group, g_group)
        return wc.plot(figsize)

    def genDistributionPlot(
        self, 
        word: str, 
        figsize: Tuple[int,int]=(6,1)
    ) -> plt.Figure:

        err = self.errorChecking(word)
        if err:
            raise Exception(err)

        x_values, y_values = self.vocab.distribution()
        w_percentile = self.vocab.getPercentile(word)
        w_freq = self.vocab.getFreq(word)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_values, y_values, color='green')
        ax.fill_between(x_values, y_values, color='lightgreen',)
        
        ax.axvline(x=max(0,w_percentile-.01), 
            color='blue', 
            linewidth=7, 
            alpha=.1,
            linestyle='-'
        )

        ax.axvline(x=min(100,w_percentile+.01), 
            color='black', 
            linewidth=7, 
            alpha=.1, 
            linestyle='-'
        )

        ax.axvline(x=w_percentile, 
            color='#d35400', 
            linewidth=2, 
            linestyle='--',
            label=f'{w_freq}\n(frecuencia total)'
        )

        ax.axis('off')
        plt.legend(loc='upper left', prop={'size': 7})
        return fig
    
    def findSplits(
        self, 
        word: str, 
        subsets_list: List[str]
    ):

        err = self.errorChecking(word)
        if err:
            raise Exception(err)

        w_splits = self.vocab.getSplits(word)

        splits_list = [] 
        for subset in subsets_list:
            current_split_list = []
            for s in w_splits:
                if (subset == s.split("_")[0]):
                    current_split_list.append(s)
            
            if current_split_list:
                splits_list.append(current_split_list)

        splits_list = [random_sample(s_list, 1)[0] for s_list in splits_list]

        ds_list = [ 
            load_dataset(path=self.context_ds_name, name=split, streaming=True, split='all') 
            for split in splits_list
        ]

        datasets = ds_list[0]
        if len(ds_list) > 1:
            datasets = interleave_datasets(ds_list, probabilities=None)

        return datasets

    def findContexts(
        self, 
        sample: str, 
        word: str
    ) -> Dict[str,str]:

        sample = sample['text'].strip()
        context = ""
        m = re.search(r'\b{}\b'.format(word), sample)
        if m:
            init = m.span()[0]
            end = init+len(word)
            context = sample[:init]+"<u><b>"+word+"</b></u>"+sample[end:]
        return {'context':context}

    def getSubsetsInfo(
        self, 
        word: str
    ) -> Tuple:

        err = self.errorChecking(word)
        if err:
            raise Exception(err)

        total_freq = self.vocab.getFreq(word)
        subsets_name_list = list(self.vocab.getSubsets(word).keys())
        subsets_freq_list = list(self.vocab.getSubsets(word).values())

        # Create subset frequency dict to subset_freq component
        subsets_info = {
            s_name + f" ({s_freq})": s_freq/total_freq
            for s_name, s_freq in zip(subsets_name_list, subsets_freq_list) 
        }

        subsets_origin_info = dict(sorted(subsets_info.items(), key=lambda x: x[1], reverse=True))
        subsets_info = self.Label.compute(subsets_origin_info)
        return subsets_info, subsets_origin_info

    def getContexts(
        self, 
        word: str, 
        n_context: int, 
        ds
    ) -> List[Tuple]:

        err = self.errorChecking(word)
        if err:
            raise Exception(err)

        ds_w_contexts = ds.map(lambda sample: self.findContexts(sample, word))
        only_contexts = ds_w_contexts.filter(lambda sample: sample['context'] != "")
        shuffle_contexts = only_contexts.shuffle(buffer_size=10)
        
        list_of_dict = list(shuffle_contexts.take(n_context))
        list_of_contexts = [
            (i, dic['context'], dic['subset']) 
            for i,dic in enumerate(list_of_dict)
        ]

        return list_of_contexts