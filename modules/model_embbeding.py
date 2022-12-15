from modules.module_ann import Ann
from memory_profiler import profile
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from typing import List
import os
import operator
import pandas as pd

import numpy as np
from numpy import dot
from gensim import matutils


class Embedding:
    @profile
    def __init__(self, 
        path: str, 
        binary: bool, 
        limit: int=None, 
        randomizedPCA: bool=False,
        max_neighbors: int=20
    ) -> None:

        # Embedding vars
        self.path = path
        self.limit = limit
        self.randomizedPCA = randomizedPCA
        self.binary = binary
        self.max_neighbors = max_neighbors
        
        # Full embedding dataset 
        self.ds = None

        # Estimate NearestNeighbors
        self.ann = None     # Aproximate with Annoy method
        self.neigh = None   # Exact with Sklearn method

        # Load embedding and pca dataset
        self.__load()

    def __load(
        self, 
    ) -> None:

        print(f"Preparing {os.path.basename(self.path)} embeddings...")

        # --- Prepare dataset ---
        self.ds = self.__preparate(
            self.path, self.binary, self.limit, self.randomizedPCA
        )

        # --- Estimate Nearest Neighbors
        # Method A: Througth annoy using forest tree
        self.ann = Ann(
            words=self.ds['word'], 
            vectors=self.ds['embedding'], 
            coord=self.ds['pca']
        )
        self.ann.init(
            n_trees=20, metric='dot', n_jobs=-1
        )

        # Method B: Througth Sklearn method
        self.neigh = NearestNeighbors(
            n_neighbors=self.max_neighbors
        )
        self.neigh.fit(
            X=self.ds['embedding'].to_list()
        )
    
    def __preparate(
        self, 
        path: str, 
        binary: bool, 
        limit: int,
        randomizedPCA: bool
    ) -> pd.DataFrame:

        if randomizedPCA:
            pca = PCA(
                n_components=2, 
                copy=False, 
                whiten=False, 
                svd_solver='randomized', 
                iterated_power='auto'
            )

        else:
            pca = PCA(
                n_components=2
            )
        
        print("--------> PATH:", path)
        model = KeyedVectors.load_word2vec_format(
            fname=path, 
            binary=binary, 
            limit=limit,
            unicode_errors='ignore'
        )

        # Cased Vocab
        cased_words = model.index_to_key
        cased_emb = model.get_normed_vectors()
        cased_pca = pca.fit_transform(cased_emb)

        df_cased = pd.DataFrame(
            zip(
                cased_words,
                cased_emb,
                cased_pca
            ),
            columns=['word', 'embedding', 'pca']
        )

        df_cased['word'] = df_cased.word.apply(lambda w: w.lower())
        df_uncased = df_cased.drop_duplicates(subset='word')
        return df_uncased

    def __getValue(
        self, 
        word: str, 
        feature: str
    ):
        word_id, value = None, None

        if word in self:
            word_id = self.ds['word'].to_list().index(word)
        
        if word_id != None:
            value = self.ds[feature].to_list()[word_id]

        return value
    
    def getEmbedding(
        self, 
        word: str
    ):

        return self.__getValue(word, 'embedding')

    def getPCA(
        self, 
        word: str
    ):

        return self.__getValue(word, 'pca')
    
    def getNearestNeighbors(
        self, 
        word: str, 
        n_neighbors: int=10, 
        nn_method: str='sklearn'
    ) -> List[str]:

        assert(n_neighbors <= self.max_neighbors), f"Error: The value of the parameter 'n_neighbors:{n_neighbors}' must less than or equal to {self.max_neighbors}!."

        if nn_method == 'ann':
            words = self.ann.get(word, n_neighbors)
            
        elif nn_method == 'sklearn':
            word_emb = self.getEmbedding(word).reshape(1,-1)
            _, nn_ids = self.neigh.kneighbors(word_emb, n_neighbors+1)
            words = [self.ds['word'].to_list()[idx] for idx in nn_ids[0]][1:]
        else: 
            words = []
        return words

    def __contains__(
        self, 
        word: str
    ) -> bool:

        return word in self.ds['word'].to_list()

    # ToDo: Revisar estos dos métodos usados en la pestaña sesgoEnPalabras
    # ya que ahora los embedding vienen normalizados
    def cosineSimilarities(self, vector_1, vectors_all):
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities 

    def getCosineSimilarities(self, w1, w2):
        return dot(
            matutils.unitvec(self.getEmbedding(w1)), 
            matutils.unitvec(self.getEmbedding(w2))
        )
