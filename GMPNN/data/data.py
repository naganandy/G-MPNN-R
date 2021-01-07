import os, inspect, numpy as np, torch
from data import utils

from torch.utils.data import Dataset, DataLoader
from functools import partial



class Loader(object):
    """
    Load train, valid, test splits of the dataset
    Assumes the presence of train, valid, and test files 
    Each file is assumed to contain multi-relational hyperedges 
    """
    def __init__(self, p):
        self.p = p        
        d, _ = os.path.split(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.files = []
        
        for s in ['train', 'valid', 'test', 'aux']: 
            path = os.path.join(d, p.data, s + ".txt")
            if os.path.isfile(path): self.files.append(path)
        
        self.uns = os.path.join(d, p.data, "unseen.txt")
        self.X = os.path.join(d, p.data, "X.txt")

        self.ent = os.path.join(d, p.data, "entities.txt")
        self.rel = os.path.join(d, p.data, "relations.txt")


    def load(self):
        """
        load train, valid, test splits of the multi-relational ordered hyergraph
        """
        log = self.p.logger

        v2i, r2i, m = utils.Map(self.files)
        self.p.n, self.p.N, self.p.m  = len(v2i), len(r2i), m
        self.p.i2v, self.p.i2r = utils.rawmap(v2i, self.ent), utils.rawmap(r2i, self.rel)
        
        log.info("Number of vertices is " + str(len(v2i)-1))
        log.info("Maximum size of hyperedge is " + str(m))
        log.info("Number of relations is " + str(len(r2i)-1))
        
        U = utils.unseen(self.uns, v2i)
        inc, self.data, self.items = utils.read(self.files, v2i, r2i, m, U)

        X, self.p.i = utils.features(self.X, v2i)
        log.info("Number of vertex features is " + str(self.p.i))

        structure = {"I": inc, "U": U, "X": torch.FloatTensor(X)}
        return structure, self._splits()


    def _splits(self):
        """
        split the dataset into train-valid-test splits
        """
        train = self._loader(Train, 'train', '', shuffle=False)
        
        #validraw = self._loader(Test, 'valid', 'raw', shuffle=False) 
        #testraw = self._loader(Test, 'test', 'raw', shuffle=False)
        valid = self._loader(Test, 'valid', 'fil', shuffle=False) 
        test = self._loader(Test, 'test', 'fil', shuffle=False)
          
        
        return {
        "train_valid": {'train': train, 'valid': valid}, 
        "test": test
        }


    def _loader(self, Split, s, t, shuffle=False):
        return DataLoader(
            Split(self.data[s], self.items, t, self.p),
            batch_size = self.p.b if s == 'train' else self.p.B,
            shuffle = shuffle,
            num_workers = max(0, self.p.w),
            collate_fn = partial(Split.collate_fn, m=self.p.m)
        )



class Train(Dataset):
    def __init__(self, data, items, t, p): self.data, self.p = data, p
    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        """
        get positive and corresponding negative instances
        d: datapoint (hyperedge)
        a: arity
        r: negative ratio
        e: positive hyperedge - [relation, e1, ..., em, label, arity] 
        """
        d = self.data[i]
        a = len(d)-1
        r = self.p.nr

        e = np.int32(np.zeros(3+self.p.m))   # [relation, e1, ..., em, label, arity]  
        e[:a+1] = d
        e[-1] = a

        item = np.repeat([e], 1+r*a , axis=0)
        item[0, -2] = 1
        for i in range(a): item[1+r*i: 1+r*(i+1), i+1] = np.random.randint(low=1, high=self.p.n, size=r)
        
        return torch.LongTensor(item)


    @staticmethod
    def collate_fn(batch, m):
        """
        m: maximum arity
        batch: batch of instances
        b: full batch
        
        a: arities of the batch
        r: relation
        e: ordered hyperedge (list of entities)     

        o: masks of ones
        z: masks of zeroes
        l: labels
        """
        b = torch.cat(batch)
        a = b[:,m+2]
        
        o, z = np.zeros((len(b),m)), np.ones((len(b), m))
        for i in range(len(batch)): 
            o[i][0:a[i]] = 1
            z[i][0:a[i]] = 0
        
        r = torch.LongTensor(b[:,0])
        e = []
        for i in range(m): e.append(torch.LongTensor(b[:,i+1]))
        
        l = torch.LongTensor(b[:,m+1])
        o = torch.FloatTensor(o)
        z = torch.FloatTensor(z)
        
        return {'r':r, 'e':e, 'l':l, 'o':o, 'z':z}



class Test(Dataset):
    def __init__(self, data, items, t, p): self.data, self.items, self.t, self.p = data, items, t, p
    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        """
        get positive and corresponding negative instances

        d: datapoint (hyperedge)
        a: arity
        r: negative ratio
        e: positive hyperedge - [relation, e1, ..., em, label, arity]  
        """     
        d = self.data[i]
        a = len(d)-1
        
        e = np.int32(np.zeros(3+self.p.m))    # [relation, e1, ..., em, label, arity] 
        e[:a+1] = d
        e[-1] = a

        Item = []
        for i in range(a):
            item = np.repeat([e], self.p.n-1 , axis=0)
            item[:, i+1] = np.arange(self.p.n-1) + 1    
            
            raw = tuple(map(tuple, item))
            if self.t == 'raw': item = np.vstack([e], list(raw))
            elif self.t == 'fil': item = np.vstack([e, list(set(raw)-self.items)])
            
            item[0, -2] = 1
            Item.append(item)

        return torch.cat([torch.LongTensor(I) for I in Item]), torch.LongTensor([len(I) for I in Item])


    @staticmethod
    def collate_fn(batch, m):
        """
        m: maximum arity
        batch: batch of instances
        b: full batch
        L: lengths
        
        a: arities of the batch
        r: relation
        e: ordered hyperedge (list of entities)     

        o: masks of ones
        z: masks of zeroes
        l: labels
        """   
        b, L = list(map(torch.cat, list(map(list, zip(*batch)))))    # store the lengths of individual sets of hyperedges for each arity
        a = b[:,m+2]
        
        o, z = np.zeros((len(b),m)), np.ones((len(b), m))
        for i in range(len(b)):
            o[i][0:a[i]] = 1
            z[i][0:a[i]] = 0

        r = torch.LongTensor(b[:,0])
        e = []
        for i in range(m): e.append(torch.LongTensor(b[:,i+1]))

        l = torch.LongTensor(b[:,m+1])
        o = torch.FloatTensor(o)
        z = torch.FloatTensor(z)

        return {'r':r, 'e':e, 'l':l, 'o':o, 'z':z, 'L':L}