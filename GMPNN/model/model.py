import torch, numpy as np, os
from torch.nn.init import xavier_normal_

from tqdm import tqdm
from collections import defaultdict as ddict
from model import utils



class GMPNN():
    def __init__(self, I, p): 
        
        self.model = MessagePassing(p).to(p.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=p.lr)
        self.loss = torch.nn.CrossEntropyLoss()
        
        self.cpt = os.path.join(p.dir, 'best.pt')
        self.p = p
        

    def summary(self, log): log.info("Model is\n" + str(self.model) + "\n\n")


    def learn(self, data):
        self.best = 0.0

        epochs = tqdm(range(self.p.epochs), unit="epoch")
        for e in epochs:
            loss = self._fit(data['train'])
            
            if ((e+1)%self.p.v==0):
                 results = self.test(data['valid'], load=False)
                 if results['mrr'] > self.best: 
                    self.best = results['mrr']
                    torch.save(self.model.state_dict(), self.cpt)

            epochs.set_postfix(loss=loss, Best=self.best)
            epochs.set_description('{}/{}'.format(e, self.p.epochs))
            epochs.update()


    def _fit(self, data):
        self.model.train()
        losses = []

        for i, b in enumerate(data):
            batch = self._cast(b)
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
        if load: self.model.load_state_dict(torch.load(self.cpt))
        self.model.eval()

        results, norm = ddict(int), 0
        with torch.no_grad():
            for i, b in enumerate(data):
                batch = self._cast(b)
                out = self.model.forward(batch)

                Z = torch.split(out, batch['L'].tolist())
                results = utils.metrics(Z, results)
                norm += len(Z) 

        return {k: round(v / norm, 5) for k,v in results.items()}



class MessagePassing(torch.nn.Module):
    def __init__(self, p):
        super(MessagePassing, self).__init__()
        
        self.d = p.d
        self.V = torch.nn.Embedding(p.n, self.d, padding_idx=0)
        self.R = torch.nn.Embedding(p.N, self.d, padding_idx=0)
        
        self.init()
        self.dropout = torch.nn.Dropout(p.drop)
        

    def init(self):
        
        self.V.weight.data[0] = torch.ones(self.d)
        self.R.weight.data[0] = torch.ones(self.d)
        
        xavier_normal_(self.V.weight.data[1:])
        xavier_normal_(self.R.weight.data[1:])


    def forward(self, batch):
        r = batch['r']
        x  = self.R(r)
        
        for i in range(len(batch['e'])): x =  x * self.V(batch['e'][i])
        x = self.dropout(x)
        x = torch.sum(x, dim=1)
        
        return x