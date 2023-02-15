from modules.module_ann import Ann
from memory_profiler import profile
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from typing import List, Any
import os
import pandas as pd

import numpy as np
from numpy import dot
from gensim import matutils


class Embedding:
    def __init__(self, 
        path: str, 
        limit: int=None,
        randomizedPCA: bool=False,
        max_neighbors: int=20,
        nn_method: str='sklearn'
    ) -> None:

        # Embedding vars
        self.path = path
        self.limit = limit
        self.randomizedPCA = randomizedPCA
        self.max_neighbors = max_neighbors

        self.availables_nn_methods = ['sklearn', 'ann']
        self.nn_method = nn_method
        
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

        #assert(self.nn_method in self.availables_nn_methods), f"Error: The value of the parameter 'nn method' can only be {self.availables_nn_methods}!"
        if self.nn_method not in self.availables_nn_methods:
            raise ValueError(f"'nn method' parameter possible values are {self.availables_nn_methods}")
        
        print(f"Preparing {os.path.basename(self.path)} embeddings...")

        # --- Prepare dataset ---
        self.ds = self.__preparate(
            self.path, self.limit, self.randomizedPCA
        )

        # --- Estimate Nearest Neighbors
        if self.nn_method == 'sklearn':
            # Method A: Througth Sklearn method
            self.__init_sklearn_method(
                max_neighbors=self.max_neighbors,
                vectors=self.ds['embedding'].to_list()
            )
        
        elif self.nn_method == 'ann':
            # Method B: Througth annoy using forest tree
            self.__init_ann_method(
                words=self.ds['word'].to_list(), 
                vectors=self.ds['embedding'].to_list(), 
                coord=self.ds['pca'].to_list()
            )
    
    def __preparate(
        self, 
        path: str,
        limit: int,
        randomizedPCA: bool
    ) -> pd.DataFrame:

        if not os.path.isfile(path):
            raise FileNotFoundError(path)

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

        try:
            model = KeyedVectors.load_word2vec_format(
                    fname=path, 
                    binary=path.endswith('.bin'), 
                    limit=limit,
                    unicode_errors='ignore'
                )
        except:
            raise TypeError(f"Can't load {path} file. If it's .bin extended, only Gensim c binary format is valid")

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

    def __init_ann_method(
        self, 
        words: List[str],
        vectors: List[float],
        coord: List[float], 
        n_trees: int=20,
        metric: str='dot'
    ) -> None:

        print("Initializing Annoy method to search for nearby neighbors...")
        self.ann = Ann(
            words=words,
            vectors=vectors,
            coord=coord,
        )

        self.ann.init(
            n_trees=n_trees, 
            metric=metric, 
            n_jobs=-1
        )
    
    def __init_sklearn_method(
        self,
        max_neighbors: int,
        vectors: List[float]
    ) -> None:
        
        print("Initializing sklearn method to search for nearby neighbors...")
        self.neigh = NearestNeighbors(
            n_neighbors=max_neighbors
        )
        self.neigh.fit(
            X=vectors
        )

    def __getValue(
        self, 
        word: str, 
        feature: str
    ) -> Any:

        word_id, value = None, None

        if word in self:
            word_id = self.ds['word'].to_list().index(word)
        
        if word_id != None:
            value = self.ds[feature].to_list()[word_id]
        else:
            print(f"The word '{word}' does not exist")

        return value
    
    def getEmbedding(
        self, 
        word: str
    ) -> np.ndarray:

        return self.__getValue(word, 'embedding')

    def getPCA(
        self, 
        word: str
    ) -> np.ndarray:

        return self.__getValue(word, 'pca')
    
    def getNearestNeighbors(
        self, 
        word: str, 
        n_neighbors: int=10, 
        nn_method: str='sklearn'
    ) -> List[str]:

        #assert(n_neighbors <= self.max_neighbors), f"Error: The value of the parameter 'n_neighbors:{n_neighbors}' must less than or equal to {self.max_neighbors}!."

        #assert(nn_method in self.availables_nn_methods), f"Error: The value of the parameter 'nn method' can only be {self.availables_nn_methods}!"
        
        if n_neighbors > self.max_neighbors:
            raise ValueError(f"'n_neighbors: {n_neighbors}' parameter must less than or equal to {self.max_neighbors}")
        elif nn_method not in self.availables_nn_methods:
            raise ValueError(f"'nn method' parameter possible values are {self.availables_nn_methods}")

        neighbors_list = []

        if word not in self:
            print(f"The word '{word}' does not exist")
            return neighbors_list

        if nn_method == 'ann':
            if self.ann is None:
                self.__init_ann_method(
                    words=self.ds['word'].to_list(), 
                    vectors=self.ds['embedding'].to_list(), 
                    coord=self.ds['pca'].to_list()
                )
            neighbors_list = self.ann.get(word, n_neighbors)
            
        elif nn_method == 'sklearn':
            if self.neigh is None:
                self.__init_sklearn_method(
                    max_neighbors=self.max_neighbors,
                    vectors=self.ds['embedding'].to_list()
                )

            word_emb = self.getEmbedding(word).reshape(1,-1)
            _, nn_ids = self.neigh.kneighbors(word_emb, n_neighbors + 1)     
            neighbors_list = [self.ds['word'].to_list()[idx] for idx in nn_ids[0]][1:]

        return neighbors_list

    def cosineSimilarities(
        self, 
        vector_1, 
        vectors_all
    ):
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities 

    def getCosineSimilarities(
        self, 
        w1, 
        w2
    ):
    
        return dot(
            matutils.unitvec(self.getEmbedding(w1)), 
            matutils.unitvec(self.getEmbedding(w2))
        )

    def __contains__(
        self, 
        word: str
    ) -> bool:

        return word in self.ds['word'].to_list()