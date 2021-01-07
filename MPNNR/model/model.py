import numpy as np, os
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from tqdm import tqdm

import torch, torch.nn.functional as F
from torch.nn.modules.module import Module

from model import utils



class MPNNR():
    def __init__(self, data, p): 
        
        data['A'] = utils.adjacency(data['hypergraph'])
        self.model = MPNN(data, p).to(p.device)
        self.Y = torch.LongTensor(np.where(data['Y'])[1]).to(p.device)
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr=p.lr)
        self.loss = torch.nn.CrossEntropyLoss()
        self.p = p
        

    def summary(self, log): log.info("Model is\n" + str(self.model) + "\n\n")


    def learn(self, train):
        model, opt = self.model, self.opt
        model.train()
        Y = self.Y[train]

        for _ in tqdm(range(self.p.epochs)):
            opt.zero_grad()

            H = model.Forward()
            Z = F.log_softmax(H[train], dim=1)

            loss = F.nll_loss(Z, Y)
            loss.backward()
            opt.step()

        self.model = model
        self.train = train


    def Test(self, test):
        train = self.train
        model = self.model
        Y = self.Y[test]

        H = model.Forward()
        Z = F.log_softmax(H[test], dim=1)
        
        predictions = Z.max(1)[1].type_as(Y)
        correct = predictions.eq(Y).double()
        correct = correct.sum()

        accuracy = correct / len(Y)
        return accuracy



class MPNN(torch.nn.Module):
    def __init__(self, data, p):
        super(MPNN, self).__init__()

        A = data['A'].to(p.device)
        d, h, c = data['d'], p.h, data['c']

        L = [MessagePassing(A, d, h, p), MessagePassing(A, h, c, p, out=True)]
        self.layers = torch.nn.Sequential(*L)
        self.X = torch.FloatTensor(utils.normalise(data['X'])).to(p.device)


    def Forward(self): return self.layers.forward(self.X)



class MessagePassing(Module):

    def __init__(self, A, i, o, p, out=False):
        super(MessagePassing, self).__init__()
        
        self.A, self.d, self.relu = A, p.drop, p.relu
        self.linear = torch.nn.Linear(i, o)
        self.out = out
        

    def forward(self, X):
        A = self.A
        H = self.linear.forward(X)
        
        H = torch.spmm(A, H)
        if not self.out and self.relu: H = F.relu(H)
        if not self.out: H = F.dropout(H, self.d, training=self.training)

        return H