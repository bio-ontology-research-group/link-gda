"""Source-pinned native CLMPT layers with exact-arity query buckets."""
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F


def official_classes():
    directory=Path(__file__).resolve().parent.parent/'references'
    namespace={'torch':torch,'nn':nn,'F':F,'defaultdict':defaultdict,'NeuralBinaryPredicate':object}
    exec(compile((directory/'official-transformer.py').read_text(),'official-transformer','exec'),namespace)
    tree=ast.parse((directory/'official-complex.py').read_text())
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ComplEx')
    node.bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'official-complex','exec'),namespace)
    tree=ast.parse((directory/'official-clmpt.py').read_text())
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CLMPTLayer')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'official-clmpt','exec'),namespace)
    return namespace['ComplEx'],namespace['CLMPTLayer']


class QueryModel(nn.Module):
    def __init__(self, pretrained, allowed_hp_ids, device='cuda', temperature=.05):
        super().__init__()
        Complex,Layer=official_classes()
        self.num_entities=int(pretrained['embeddings.0.weight'].shape[0])
        self.num_relations=int(pretrained['embeddings.1.weight'].shape[0])
        assert pretrained['embeddings.0.weight'].shape[1]==800 and pretrained['embeddings.1.weight'].shape[1]==800
        assert pretrained['embeddings.0.weight'].dtype==torch.float32 and pretrained['embeddings.1.weight'].dtype==torch.float32
        assert torch.get_default_dtype()==torch.float32
        self.temperature=float(temperature)
        assert self.temperature in (.025,.05,.1)
        self.nbp=Complex(self.num_entities,self.num_relations,400,device=device)
        self.layer=Layer(2048,self.nbp,layers=2,agg_func='mean',pre_norm=True)
        self.nbp.load_state_dict({'_entity_embedding.weight':pretrained['embeddings.0.weight'],'_relation_embedding.weight':pretrained['embeddings.1.weight']},strict=True)
        self.to(device)
        self.device=device
        self.allowed_hp_ids=set(map(int,allowed_hp_ids))

    @staticmethod
    def signature(query):
        return tuple((h,t) for h,r,t in query['atoms']),tuple(sorted(query['constants']))

    def encode_bucket(self,queries):
        assert queries and all(self.signature(q)==self.signature(queries[0]) for q in queries)
        first=queries[0]
        assert not set(first['constants'])&{'f','e1','e2','e3'}
        for query in queries:
            assert all(0<=int(i)<self.num_entities for i in query['constants'].values())
            if query['type']=='profile':
                assert len(set(query['hp_ids']))==len(query['hp_ids']) and query['hp_ids'] and set(query['hp_ids'])<=self.allowed_hp_ids
                assert sorted(query['constants'].values())==sorted(query['hp_ids'])
                assert all(atom[2]=='e1' for atom in query['atoms'][:-1]) and query['atoms'][-1][0:3:2]==['e1','f']
        terms={t for h,r,t in first['atoms']}|{h for h,r,t in first['atoms']}
        assert 'f' in terms
        local={}
        for term in sorted(terms):
            if term in first['constants']:
                local[term]=self.nbp.get_entity_emb([q['constants'][term] for q in queries])
            elif term=='f':local[term]=self.layer.free_embedding
            else:
                assert term.startswith('e') and term[1:].isdigit()
                local[term]=self.layer.existential_embedding
        atoms={};relations={};inverse={}
        for i,(head,role,tail) in enumerate(first['atoms']):
            name='r'+str(i)
            atoms[name]=SimpleNamespace(head=SimpleNamespace(name=head),tail=SimpleNamespace(name=tail),relation=name,negated=False)
            roles=[q['atoms'][i][1] for q in queries]
            relations[name]=self.nbp.get_relation_emb(roles)
            inverse[name]=self.nbp.get_relation_emb(roles,inv=True)
        for _ in range(len(terms-set(first['constants']))):
            local=self.layer(local,atoms,relations,inverse)
        return local['f']

    def encode(self,queries):
        groups=defaultdict(list)
        for i,q in enumerate(queries):groups[self.signature(q)].append(i)
        results={}
        for indices in groups.values():
            values=self.encode_bucket([queries[i] for i in indices])
            for j,index in enumerate(indices):results[index]=values[j]
        return torch.stack([results[i] for i in range(len(queries))])

    def loss(self,queries,positive,negative):
        representation=self.encode(queries)
        target=self.nbp._entity_embedding(positive)
        negatives=self.nbp._entity_embedding(negative)
        positive_score=F.cosine_similarity(representation,target,dim=-1)/self.temperature
        negative_score=F.cosine_similarity(representation[:,None,:],negatives,dim=-1)/self.temperature
        logits=torch.cat([positive_score[:,None],negative_score],dim=1)
        return F.cross_entropy(logits,torch.zeros(len(queries),device=logits.device,dtype=torch.long))

    def scores(self,queries,candidates):
        representation=F.normalize(self.encode(queries),dim=-1)
        return representation@F.normalize(self.nbp._entity_embedding(candidates),dim=-1).T
