import torch, numpy as np, random
from torch.nn import functional as F
from tqdm import tqdm



def padDecompose(Z, m, P):
    S = []
    for i in range(len(P)):
        if(i == len(P)-1):  S.append(pad(Z[P[i]:], m))
        else: S.append(pad(Z[P[i]:P[i+1]], m))
    return torch.stack(S)


def pad(a, m): return F.pad(a, (0,m-len(a)), 'constant', -np.inf)



def metrics(Z, results):
    ranks = [getRank(x) for x in Z]
    rank, h1, h3, h10 = list(map(list, zip(*ranks)))

    results['h@1'] += np.float32(h1).sum()
    results['h@3'] += np.float32(h3).sum()
    results['h@10'] += np.float32(h10).sum()

    results['mr'] += np.float32(rank).sum()
    results['mr'] += np.float32(rank).sum()
    results['mrr'] += (1.0 / np.float32(rank)).sum()

    return results, ranks



def getRank(pred):
    rank    = (pred >= pred[0]).sum().item()    # assuming the first one is the true hyperedge
    hits1   = 1.0 if rank == 1 else 0.0
    hits3   = 1.0 if rank <= 3 else 0.0
    hits10  = 1.0 if rank <= 10 else 0.0

    return [float(rank), hits1, hits3, hits10]



def aggregate(S, V, R, P, p):
    I, U, X = S['I'], S['U'], S['X']
    N = torch.zeros(p.n, p.d).to(p.device)

    p.logger.info("Aggregating neighbouring features...")
    if p.agg == 'mean': p.s = p.s/10
    for k in tqdm(I):
        for e in I[k]:
            
            f = []
            for i, v in enumerate(e[1:]): 
                if v not in U and v != k: f.append((i, v))
            
            if len(f) > 0: i, v = random.choice(f)
            else:
                v = k
                for j, w in enumerate(e[1:]):
                    if w == k: i = j
            
            value = R[e[0]]*P[i]*V[v]*p.s
            if p.agg == 'max': N[k] = torch.max(torch.stack((N[k], value)), 0)[0]
            elif p.agg == 'sum' or p.agg == 'mean': N[k] += value 
        if p.agg == 'mean': N[k] = N[k] / len(I[k])
      
    X = X.to(p.device)
    return torch.cat([X, N], dim=1)



def raw(b, p):
    """
    get raw hyperedges in a batch

    b: batch
    p: argparse parameters

    returns:
    list of true hyperedges in the batch
    """
    rs = torch.split(b['r'], b['L'].tolist())
    es = []
    for j in range(p.m): es.append(torch.split(b['e'][j], b['L'].tolist()))

    E = []
    for i in range(len(rs)):
        e = [p.i2r[rs[i][0].item()]]
        for j in range(p.m): e.append(p.i2v[es[j][i][0].item()])
        E.append(tuple(e))

    return E