import numpy as np, os, uuid
import torch, torch.nn as nn
from torch.nn.init import xavier_normal_

from tqdm import tqdm
from collections import defaultdict as ddict
from model import utils



class GMPNN():
    def __init__(self, S, p): 
        
        V = torch.ones(p.n, p.d).to(p.device)
        R = torch.ones(p.N, p.d).to(p.device)
        P = torch.ones(p.m, p.d).to(p.device)

        xavier_normal_(V[1:])
        xavier_normal_(R[1:])
        xavier_normal_(P)
        
        N = utils.aggregate(S, V, R, P, p).to(p.device)
        self.model = GeneralisedMessagePassing(N, R, P, p).to(p.device)
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr=p.lr)
        self.loss = nn.CrossEntropyLoss()
        self.p = p
        
        if p.log: 
            self.cpt = os.path.join(p.dir, 'best.pt')
            self.rnk = os.path.join(p.dir, 'ranks.txt')
        else: 
            u = str(uuid.uuid4())
            self.cpt, self.rnk = 'best_' + u + '.pt', 'ranks_' + u + '.txt'
        
        

    def summary(self, log): log.info("Model is\n" + str(self.model) + "\n\n")


    def learn(self, data):
        self.best = 0.0
        self.p.logger.info("Training the model (validating every " + str(self.p.v) + " epochs)...")

        epochs = tqdm(range(self.p.epochs), unit="epoch")
        for e in epochs:
            loss = self._fit(data['train'])
            
            if (e%self.p.v==0):
                 results = self.test(data['valid'], load=False)
                 if results['mrr'] > self.best: 
                    self.best = results['mrr']
                    torch.save(self.model.state_dict(), self.cpt)

            epochs.set_postfix(loss=loss, best=self.best)
            epochs.update()


    def _fit(self, data):
        self.model.train()
        losses = []

        for batch in data:
            batch = self._cast(batch)
            out = self.model.forward(batch)
            loss = self._loss(out, batch['l'])

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            losses.append(loss.item())

        return np.mean(losses)


    def _cast(self, batch):
        d = self.p.device
        
        for k in batch.keys():
            b = batch[k]
            if k == "e": batch[k] = [b[i].to(d) for i in range(len(b))]
            else: batch[k] = b.to(d)

        return batch


    def _loss(self, Z, Y):

        P = torch.nonzero(Y, as_tuple=False).squeeze()        
        Z = utils.padDecompose(Z, self.p.nr*self.p.m, P)
        Y = torch.zeros(len(P)).long().to(self.p.device)
        
        return self.loss(Z, Y)


    def test(self, data, load=True):
        with torch.no_grad():
            if load: self.model.load_state_dict(torch.load(self.cpt))
            self.model.eval()

            results, norm = ddict(int), 0
            with open(self.rnk, "w") as f:

                for batch in data:
                    b = self._cast(batch)
                    out = self.model.forward(b)

                    Z = torch.split(out, batch['L'].tolist())
                    results, ranks = utils.metrics(Z, results)
                    norm += len(Z) 

                    E = utils.raw(batch, self.p)
                    for i, e in enumerate(E):
                        f.write(str(e[0].replace("_", "\t")) + "\n") 
                        for v in e[1:]: f.write(str(v) + "\t")
                        f.write("\n")
                        f.write(str(int(ranks[i][0])))
                        f.write("\n\n")

        if load and not self.p.log: 
            os.remove(self.cpt)
            os.remove(self.rnk)
        return {k: round(v / norm, 5) for k,v in results.items()}



class GeneralisedMessagePassing(nn.Module):
    def __init__(self, V, R, P, p):
        super(GeneralisedMessagePassing, self).__init__()
                
        self.V, self.R, self.P = V, R, P
        self.dropout = nn.Dropout(p.drop)

        self.Vl = nn.Linear(p.d+p.i, p.h)
        self.Rl = nn.Linear(p.d, p.h)
        self.Pl = nn.Linear(p.d, p.h)


    def forward(self, batch):
        r = batch['r']
        x = self.R[r]
        x = self.Rl(x)

        for i in range(len(batch['e'])): x =  x * self.Pl(self.P[i]) * self.Vl(self.V[batch['e'][i]])
        x = self.dropout(x)
        x = torch.sum(x, dim=1)
        
        return x