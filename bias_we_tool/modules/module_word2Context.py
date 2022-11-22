from datasets import load_dataset, interleave_datasets
from modules.module_segmentedWordCloud import SegmentedWordCloud
from modules.module_customSubsetsLabel import CustomSubsetsLabel

from random import sample as random_sample
#import gradio as gr
#import pandas as pd
import re

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class Word2Context:
    def __init__(self, context_ds_name, vocabulary):
        self.context_ds_name = context_ds_name
        
        # Vocabulary class 
        self.vocab = vocabulary

        # Custom Label component
        self.Label = CustomSubsetsLabel()

    def errorChecking(self, word):
        out_msj = ""

        if not word:
            out_msj = "Error: Primero debe ingresar una palabra!"
        else:
            if word not in self.vocab:
                out_msj = f"Error: La palabra '<b>{word}</b>' no se encuentra en el vocabulario!"
        
        return out_msj

    def genWebLink(self,text):
        text = text.replace("\"", "'")
        text = text.replace("<u><b>", "")
        text = text.replace("</b></u>", "")
        url = "https://www.google.com.tr/search?q={}".format(text)
        return '<a href="{}" rel="noopener noreferrer" target="_blank"><center>🌐🔍</center></a>'.format(url)

    def genWordCloudPlot(self, word, figsize=(9,3)):
        freq_dic, l_group, g_group = self.vocab.getWordNeighbors(word, n_neighbors=10)
        wc = SegmentedWordCloud(freq_dic, l_group, g_group)
        return wc.plot(figsize)

    def genDistributionPlot(self, word, figsize=(6,1)):
        x_values, y_values = self.vocab.distribution()
        w_percentile = self.vocab.getPercentile(word)
        w_freq = self.vocab.getFreq(word)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_values, y_values, color='green')
        ax.fill_between(x_values, y_values, color='lightgreen',)
        
        ax.axvline(x=max(0,w_percentile-.01), 
            color='blue', 
            linewidth=7, 
            alpha=.2,
            linestyle='-'
        )
        ax.axvline(x=min(100,w_percentile+.01), 
            color='black', 
            linewidth=7, 
            alpha=.2, 
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
    
    def findSplits(self, word, subsets_list):
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

    def findContexts(self, sample, word):
        sample = sample['text'].strip()
        context = ""
        m = re.search(r'\b{}\b'.format(word), sample)
        if m:
            init = m.span()[0]
            end = init+len(word)
            context = sample[:init]+"<u><b>"+word+"</b></u>"+sample[end:]
        return {'context':context}

    def getSubsetsInfo(self, word):
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

    def getContexts(self, word, n_context, ds):
        ds_w_contexts = ds.map(lambda sample: self.findContexts(sample, word))
        only_contexts = ds_w_contexts.filter(lambda sample: sample['context'] != "")
        shuffle_contexts = only_contexts.shuffle(buffer_size=10)
        
        list_of_dict = list(shuffle_contexts.take(n_context))
        list_of_contexts = [(i,dic['context'],dic['subset']) for i,dic in enumerate(list_of_dict)]

        return list_of_contexts

    # TODO: The next methods can be removed, or keep them as a wrapper method of several ones
    '''
    def getWordInfo(self, word):
        errors = ""
        contexts = pd.DataFrame([],columns=[''])
        subsets_info = ""
        distribution_plot = None
        word_cloud_plot = None
        subsets_choice = gr.CheckboxGroup.update(choices=[])
    
        errors = self.errorChecking(word)
        if errors:
            return errors, contexts, subsets_info, distribution_plot, word_cloud_plot, subsets_choice

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

        # Create sort list to subsets_choice component
        clean_keys = [key.split(" ")[0].strip() for key in subsets_origin_info]
        subsets_choice = gr.CheckboxGroup.update(choices=clean_keys)

        # Get word distribution, and wordcloud graph
        distribution_plot = self.genDistributionPlot(word)
        word_cloud_plot = self.genWordCloudPlot(word)

        return errors, contexts, subsets_info, distribution_plot, word_cloud_plot, subsets_choice
    
    def getWordContext(self, word, n_context, subset_choice):
        n_context = int(n_context)
        errors = ""
        
        if len(subset_choice) > 0:
            ds = self.findSplits(word, subset_choice)

        else:
            errors = "Error: Palabra no ingresada y/o conjunto/s de interés no seleccionado/s!"
            errors = "<center><h3>"+errors+"</h3></center>"
            return errors, pd.DataFrame([], columns=[''])
        
        ds_w_contexts = ds.map(lambda sample: self.findContexts(sample, word))
        only_contexts = ds_w_contexts.filter(lambda sample: sample['context'] != "")
        shuffle_contexts = only_contexts.shuffle(buffer_size=10)
        
        list_of_dict = list(shuffle_contexts.take(n_context))
        list_of_contexts = [(i,dic['context'],dic['subset']) for i,dic in enumerate(list_of_dict)]

        contexts = pd.DataFrame(list_of_contexts, columns=['#','contexto','conjunto'])
        contexts["buscar"] = contexts.contexto.apply(lambda text: self.genWebLink(text))

        return errors, contexts
    '''