from memory_profiler import profile
from annoy import AnnoyIndex
from tqdm import tqdm
import time
import operator

class TicToc:
    def __init__(self):
        self.i = None
    def start(self):
        self.i = time.time()
    def stop(self):
        f = time.time()
        print(f - self.i, "seg.")

class Ann:
    def __init__(self, words, vectors, coord):
        self.words = words.to_list()
        self.vectors = vectors.to_list()
        self.coord = coord.to_list()
        self.tree = None

        self.tt = TicToc()

    @profile
    def init(self, n_trees=10, metric='angular', n_jobs=-1):
        # metrics options = "angular", "euclidean", "manhattan", "hamming", or "dot"
        # n_jobs=-1 Run over all CPU availables

        print("Init tree...")
        self.tt.start()
        self.tree = AnnoyIndex(len(self.vectors[0]), metric=metric)
        for i,v in tqdm(enumerate(self.vectors), total=len(self.vectors)):
            self.tree.add_item(i,v)
        self.tt.stop()

        print("Build tree...")
        self.tt.start()
        self.tree.build(n_trees=n_trees, n_jobs=n_jobs)
        self.tt.stop()

    def __getWordId(self, word):
        word_id = None
        try:
            word_id = self.words.index(word)
        except:
            pass
        return word_id

    def get(self, word, n_neighbors=10):
        word_id = self.__getWordId(word)
        reword_xy_list = None

        if word_id != None:
            neighbord_id = self.tree.get_nns_by_item(word_id, n_neighbors)
            # word_xy_list = list(map(lambda i: (self.words[i],self.coord[i]), neighbord_id))
            # word_xy_list = list(map(lambda i: self.words[i], neighbord_id))
            word_xy_list = operator.itemgetter(*neighbord_id)(self.words)
        else:
            print(f"The word '{word}' does not exist")
        
        return word_xy_list