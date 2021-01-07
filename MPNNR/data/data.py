
import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle


def load(args):
    """
    parses the dataset
    """
    dataset = parser(args.data).parse()

    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    file = os.path.join(Dir, args.data, "splits", str(args.split) + ".pickle")

    if not os.path.isfile(file): print("split + ", str(args.split), "does not exist")
    with open(file, 'rb') as H: 
        splits = pickle.load(H)
        if 'n' in splits: splits['test'] = list(np.delete(range(splits['n']), splits['train']))

    log = args.logger
    log.info("Number of depth 0 hyperedges is " + str(len(dataset['hypergraph']['D0'])))
    log.info("Number of depth 1 hyperedges is " + str(len(dataset['hypergraph']['D1'])))

    log.info("Number of vertices is " + str(dataset['hypergraph']['n']))
    log.info("Number of vertex features is " + str(dataset['d']))
    log.info("Number of classes is " + str(dataset['c']))

    rate = float(len(splits['train'])/(len(splits['train'])+len(splits['test'])))
    log.info("Label rate is " + str(round(rate, 5)))

    return dataset, splits



class parser(object):
    """
    an object for parsing data
    """
    
    def __init__(self, data):
        """
        initialises the data directory 
        arguments:
        data: cora/dblp/acm
        """
        
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, data)
        self.data = data

    

    def parse(self):
        """
        returns a dataset specific function to parse
        """
        
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()



    def _load_data(self):
        """
        assumes the following files to be present in the dataset directory:
        depth0.pickle: dictionary of Depth-0 hyperedges
        depth1.pickle: dictionary of Depth-1 hyperedges
        features.pickle: bag of word features
        labels.pickle: labels of instances
        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels as keys
        """
        
        with open(os.path.join(self.d, 'depth1.pickle'), 'rb') as h: depth1 = pickle.load(h)
        with open(os.path.join(self.d, 'depth0.pickle'), 'rb') as h: depth0 = pickle.load(h)
        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as h: labels = self._1hot(pickle.load(h))
        
        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            if self.data == 'cora': features = pickle.load(handle).todense()
            else: 
                D = pickle.load(handle)
                n, d, I = D['n'], D['d'], D['I']
                features = np.zeros((n, d))
                for j, indices in enumerate(I):
                    for i in indices: features[j][i] = 1

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