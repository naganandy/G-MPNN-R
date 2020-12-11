import torch, numpy as np, scipy.sparse as sp
from torch.nn import functional as F
from tqdm import tqdm



def adjacency(H):
    """
    construct adjacency for recursive hypergraph
    arguments:
    H: recursive hypergraph
    """
    A = np.eye(H['n'])
    E = H['D0']
    
    for k in tqdm(E):
        e = list(E[k])
        for u in e:
            A[k][u], A[u][k] = 1, 1
            for v in e:
                if u != v: A[u][v], A[v][u] = 1, 1

    E = H['D1']
    for k in tqdm(E):
        e = list(E[k])
        for u in e:
            for v in e:
                if u != v: A[u][v], A[v][u] = 1, 1

    
    return ssm2tst(symnormalise(sp.csr_matrix(A)))



def symnormalise(M):
    """
    symmetrically normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 



def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)
    arguments:
    M: scipy sparse matrix
    returns:
    a torch sparse tensor of M
    """
    
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)



def normalise(M):
    """
    row-normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)