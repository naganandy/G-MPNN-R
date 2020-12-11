import torch, numpy as np
from torch.nn import functional as F



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

	return results



def getRank(pred):
    rank    = (pred >= pred[0]).sum().item()    # assuming the first one is the true hyperedge
    hits1   = 1.0 if rank == 1 else 0.0
    hits3   = 1.0 if rank <= 3 else 0.0
    hits10  = 1.0 if rank <= 10 else 0.0

    return [float(rank), hits1, hits3, hits10]