import os, numpy as np, torch
from ordered_set import OrderedSet
from collections import defaultdict as ddict



def Map(files):
    """
    Read a list of files containing multi-relational hyperedges and return index maps

    Arguments
    files: list of files (train, test, valid, aux splits)

    Returns
    v2i: vertex-to-index map
    r2i: relation-to-index map
    m: maximum size of hyperedge
    """
    V, R = OrderedSet(), OrderedSet()
    V.add(''), R.add('')
    v2i, r2i = {}, {}

    m = 0 
    for F in files:
        with open(F, "r") as f:
            for line in f.readlines():
                row = line.strip().split('\t')
                R.add(row[0])
                
                e = row[1:]
                V.update(e)
                if len(e) > m: m = len(e)

    v2i, r2i = {v: i for i, v in enumerate(V)}, {r: i for i, r in enumerate(R)}
    return v2i, r2i, m



def rawmap(k2i, file):
    """
    Map index to raw data from file

    Arguments
    k2i: key-to-index map
    file: file containing raw data map

    Returns
    raw: index-to-raw map if file exists else identity map
    """
    raw = {0: ''}
    if os.path.isfile(file):
        with open(file, "r") as f:
            for line in f.readlines():
                line = line.split("\t")
                k, rw = line[0].strip(), line[1].strip()
                raw[k2i[k]] = rw
    else:
        for k in k2i: raw[k2i[k]] = k2i[k]

    return raw



def unseen(file, v2i):
    """
    Get unseen vertices at test time

    Arguments
    file: file containing a list of vertices
    v2i: vertex-to-index map

    Returns
    U: set of unseen vertices
    """
    U = set()
    if os.path.isfile(file):
        with open(file, "r") as f:
            for line in f.readlines(): U.add(v2i[line.strip()])  

    return U



def read(files, v2i, r2i, m, U):
    """
    Read data from files 

    Arguments
    files: list of files (train, test, valid, aux splits)   
    v2i: vertex-to-index map
    r2i: relation-to-index map
    m: maximum size of hyperedge
    U: set of unseen entities

    Returns
    inc: the incidence structure of the hypergraph (generalisation of graph neighbourhood)
    data: dictionary with splits as the key and list of hyperedges  (corresponding to the split) as values
    items: hyperedges 
    """
    inc, data, items = ddict(list), ddict(list), set()

    for F in files:
        with open(F, "r") as f:
            s = os.path.split(F)[1].split(".")[0]
            for line in f.readlines():
                row = line.strip().split('\t')
                r = r2i[row[0]]

                e, vs = [r], list(map(lambda v: v2i[v], row[1:]))
                e.extend(vs)
                data[s].append(np.int32(e))

                a, item = len(vs), np.int32(np.zeros(m + 3))
                item[:a+1], item[-2], item[-1] = e, 0, a
                items.add(tuple(item))

                if s == "train":
                    for v in vs: inc[v].append(e) 
                elif s == "aux":
                    for v in vs: 
                        if v in U: inc[v].append(e) 

    return inc, dict(data), items



def features(file, v2i):
    """ 
    Read external vertex features (if they exist)

    Arguments
    file: path to external vertex features
    v2i: vertex-to-index map

    Returns
    X: list of external vertex features
    d: number of external features
    """
    X, d = [], 0
    if os.path.isfile(file):
        with open(file, "r") as f:
            lines = f.readlines()
            
            d = 0
            for line in lines: 
                line = line.split("\t")
                for i in line[1:]:
                    if i != '':
                        i = int(i.strip())
                        if i > d: d = i
            d = d + 1
            X = torch.zeros(len(v2i), d)

            for line in lines:
                line = line.split("\t")
                v = line[0].strip()
                for i in line[1:]:
                    if i != '':
                        i = int(i.strip())
                        X[v2i[v]][i] = 1

    return X, d