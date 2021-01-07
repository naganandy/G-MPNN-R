'''
data: MFB-IND / WP-IND / JF-IND
s: initial seed value
'''
data = "JF-IND"
s = 130



'''
l: minimum degree
h: maximum degree
u: number of unseen vertices
'''
l = 5
h = 8
u = 100



import argparse
def parse():
    """
    add and parse arguments / hyperparameters
    """
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser(description="Inductive Vertex Embedding on Multi-Relational Ordered Hypergraphs")

    p.add_argument('--data', type=str, default=data, help='data name (FB-AUTO)')
    p.add_argument('--s', type=int, default=s, help='initial seed value')

    p.add_argument('--l', type=int, default=l, help='minimum degree')
    p.add_argument('--h', type=int, default=h, help='maximum degree')
    p.add_argument('--u', type=int, default=u, help='number of unseen vertices')

    p.add_argument('-f') # for jupyter default
    return p.parse_args()



import os, random, numpy as np
def setup():
    args = parse()
    args.dir = os.path.join(os.getcwd(), args.data)

    random.seed(args.s)
    np.random.seed(args.s)
    os.environ['PYTHONHASHSEED'] = str(args.s) 

    return args