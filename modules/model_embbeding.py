from memory_profiler import profile
from gensim import matutils
import numpy as np
from numpy import dot
import pandas as pd
import ast


class Embedding:
    @profile
    def __init__(self, subset_name):
        # Dataset info
        self.ds_subset = subset_name
        self.ds_path = f"data/{subset_name}_embedding_v6.zip"
        
        # Pandas dataset
        self.ds = None

        # All Words embedding List[List[float]]
        self.embedding = None

        # Load embedding and pca dataset
        self.__load()

    def __contains__(self, word):
        return word in self.ds['word'].to_list()

    def __load(self):
        print(f"Preparing {self.ds_subset} embedding...")

        # --- Download dataset ---
        self.ds = pd.read_json(self.ds_path)

        # --- Get embedding from string
        self.embedding = self.ds['embedding'].to_list()

    def __getValue(self, word, feature):
        word_id, value = None, None

        if word in self:
            word_id = self.ds['word'].to_list().index(word)
        
        if word_id != None:
            value = self.ds[feature].to_list()[word_id]

        return value
    
    def getEmbedding(self, word):
        return self.__getValue(word, 'embedding')


    def getPCA(self, word):
        return self.__getValue(word, 'pca')
    
    def cosineSimilarities(self, vector_1, vectors_all):
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities 
    
    def getNearestNeighbors(self, word, n_neighbors=10):
        word_vector = self.getEmbedding(word)
        dists = dot(self.embedding, word_vector)
        best = matutils.argsort(dists, topn=n_neighbors, reverse=True)
        neighbors = [self.ds['word'].to_list()[n] for n in best[1:]]
        return neighbors

    def getCosineSimilarities(self, w1, w2):
        return dot(
            matutils.unitvec(self.getEmbedding(w1)), 
            matutils.unitvec(self.getEmbedding(w2))
        )