import time
from tqdm import tqdm
from annoy import AnnoyIndex
from memory_profiler import profile
from typing import List

class TicToc:
    def __init__(
        self
    ) -> None:

        self.i = None

    def start(
        self
    ) -> None:

        self.i = time.time()

    def stop(
        self
    ) -> None:

        f = time.time()
        print(f - self.i, "seg.")


class Ann:
    def __init__(
        self, 
        words: List[str], 
        vectors: List, 
        coord: List, 
    ) -> None:

        self.words = words
        self.vectors = vectors
        self.coord = coord
        self.tree = None

        self.tt = TicToc()
        self.availables_metrics = ['angular','euclidean','manhattan','hamming','dot']

    def init(self, 
        n_trees: int=10, 
        metric: str='angular', 
        n_jobs: int=-1  # n_jobs=-1 Run over all CPU availables
    ) -> None:

        assert(metric in self.availables_metrics), f"Error: The value of the parameter 'metric' can only be {availables_metrics}!"

        print("\tInit tree...")
        self.tt.start()
        self.tree = AnnoyIndex(len(self.vectors[0]), metric=metric)
        for i, v in tqdm(enumerate(self.vectors), total=len(self.vectors)):
            self.tree.add_item(i, v)
        self.tt.stop()

        print("\tBuild tree...")
        self.tt.start()
        self.tree.build(n_trees=n_trees, n_jobs=n_jobs)
        self.tt.stop()

    def __getWordId(
        self, 
        word: str
    ) -> int:

        word_id = None
        try:
            word_id = self.words.index(word)
        except:
            pass
        return word_id

    def get(
        self, 
        word: str, 
        n_neighbors: int=10
    ) -> List[str]:
        
        word_id = self.__getWordId(word)
        neighbors_list = None

        if word_id != None:
            neighbords_id = self.tree.get_nns_by_item(word_id, n_neighbors + 1)
            neighbors_list = [self.words[idx] for idx in neighbords_id][1:]

        else:
            print(f"The word '{word}' does not exist")

        return neighbors_list