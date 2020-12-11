
import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle


def load(args):
    """
    parses the dataset
    """
    dataset = parser(args.data, args.logger).parse()

    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    file = os.path.join(Dir, args.data, "splits", str(args.split) + ".pickle")

    if not os.path.isfile(file): print("split + ", str(args.split), "does not exist")
    with open(file, 'rb') as H: splits = pickle.load(H)

    return dataset, splits



class parser(object):
    """
    an object for parsing data
    """
    
    def __init__(self, data, log):
        """
        initialises the data directory 
        arguments:
        data: cora/dblp/acm
        log: logger object
        """
        
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, data)
        self.log = log

    

    def parse(self):
        """
        returns a dataset specific function to parse
        """
        
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()



    def _load_data(self):
        """
        loads the coauthorship hypergraph, features, and labels of cora
        assumes the following files to be present in the dataset directory:
        hypergraph.pickle: coauthorship, cocitation recursive hypergraph
        features.pickle: bag of word features
        labels.pickle: labels of papers
        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels as keys
        """
        
        with open(os.path.join(self.d, 'depth1.pickle'), 'rb') as handle:
            depth1 = pickle.load(handle)
            self.log.info("number of depth 1 hyperedges is " + str(len(depth1)))

        with open(os.path.join(self.d, 'depth0.pickle'), 'rb') as handle:
            depth0 = pickle.load(handle)
            self.log.info("number of depth 0 hyperedges is " + str(len(depth0)))

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle))

        return {'hypergraph': {'D0': depth0, 'D1': depth1, 'n': features.shape[0]}, 'X': features, 'Y': labels, 'd': features.shape[1], 'c': labels.shape[1]}



    def _1hot(self, labels):
        """
        converts each positive integer (representing a unique class) into ints one-hot form
        Arguments:
        labels: a list of positive integers with eah integer representing a unique label
        """
        
        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)