import torch
from torch import nn

class ComplEx(nn.Module):
    """CQD ComplEx score/factors with explicit plain-state interface."""
    def __init__(self, entities, relations, rank=400, init_size=1e-3):
        super().__init__()
        self.rank=rank
        self.embeddings=nn.ModuleList([nn.Embedding(n,2*rank,sparse=True) for n in (entities,relations)])
        for table in self.embeddings:table.weight.data*=init_size

    def forward(self, triples):
        lhs=self.embeddings[0](triples[:,0]);rel=self.embeddings[1](triples[:,1]);rhs=self.embeddings[0](triples[:,2]);rank=self.rank
        lr,li=lhs[:,:rank],lhs[:,rank:];rr,ri=rel[:,:rank],rel[:,rank:];tr,ti=rhs[:,:rank],rhs[:,rank:];entities=self.embeddings[0].weight
        scores=(lr*rr-li*ri)@entities[:,:rank].T+(lr*ri+li*rr)@entities[:,rank:].T
        return scores,(torch.sqrt(lr**2+li**2),torch.sqrt(rr**2+ri**2),torch.sqrt(tr**2+ti**2))

    def score(self, triples):
        lhs=self.embeddings[0](triples[:,0]);rel=self.embeddings[1](triples[:,1]);rhs=self.embeddings[0](triples[:,2]);rank=self.rank
        lr,li=lhs[:,:rank],lhs[:,rank:];rr,ri=rel[:,:rank],rel[:,rank:];tr,ti=rhs[:,:rank],rhs[:,rank:]
        return ((lr*rr-li*ri)*tr+(lr*ri+li*rr)*ti).sum(1,keepdim=True)

def n3(factors,weight=.05):
    """Match official N3 on complex magnitudes, summed then divided by batch."""
    norm=0
    for factor in factors:norm=norm+weight*torch.sum(torch.abs(factor)**3)
    return norm/factors[0].shape[0]
