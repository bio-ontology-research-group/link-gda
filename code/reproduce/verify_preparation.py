"""Compare rebuilt projections, vocabularies and every graph tensor with frozen inputs."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phenoquery.common import sha, read, write


def compare(a,b,path='graph'):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
            raise ValueError('Graph tensor differs: '+path)
    elif hasattr(a,'to_dict'):
        compare(a.to_dict(),b.to_dict(),path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Graph fields differ: '+path)
        for k in a:compare(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError('Graph sequence differs: '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(i))
    elif a!=b:
        raise ValueError('Graph value differs: '+path)


def graph_pair(original,rebuilt):
    a=torch.load(original,map_location='cpu',weights_only=False)
    b=torch.load(rebuilt,map_location='cpu',weights_only=False)
    compare(a,b)
    return {'original_sha256':sha(original),'rebuilt_sha256':sha(rebuilt),'all_tensor_values_equal':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True,help='Extracted input release root')
    p.add_argument('--benchmark',type=Path)
    p.add_argument('--eq',type=Path)
    p.add_argument('--matched',type=Path)
    p.add_argument('--projection',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--folds',type=int,nargs='+',default=list(range(10)))
    a=p.parse_args()
    if not any([a.benchmark,a.eq,a.matched,a.projection]):p.error('Specify at least one rebuilt input')
    result={'source_sha256':sha(__file__),'folds':a.folds,'graphs':{},'projection':{}}
    if a.projection:
        old=a.reference/'eq/projection';new=a.projection/'projection'
        original=read(old/'manifest.json');rebuilt=read(new/'manifest.json')
        if original!=rebuilt:raise ValueError('Projection manifest values differ')
        for name,expected in original['files'].items():
            if sha(old/name)!=expected or sha(new/name)!=expected:
                raise ValueError('Projected artifact differs: '+name)
            result['projection'][name]=expected
    for fold in a.folds:
        if a.benchmark:
            original=a.reference/'benchmark/data'/f'fold{fold}'
            rebuilt=a.benchmark/'data'/f'fold{fold}'
            if read(original/'vocabulary.json')!=read(rebuilt/'vocabulary.json'):
                raise ValueError('Benchmark vocabulary differs')
            result['graphs'][f'benchmark/fold{fold}']=graph_pair(original/'processed/data.pt',rebuilt/'processed/data.pt')
        if a.matched:
            original=a.reference/'matched/graphs'/f'fold{fold}'
            rebuilt=a.matched/'graphs'/f'fold{fold}'
            if read(original/'vocabulary.json')!=read(rebuilt/'vocabulary.json'):
                raise ValueError('Matched vocabulary differs')
            result['graphs'][f'matched/fold{fold}']=graph_pair(original/'graph.pt',rebuilt/'graph.pt')
        if a.eq:
            original=a.reference/'eq/data'/f'fold{fold}'
            rebuilt=a.eq/'data'/f'fold{fold}'
            for filename in ['profiles.json','candidates.txt','training.tsv','validation.tsv','test.tsv']:
                if sha(original/filename)!=sha(rebuilt/filename):raise ValueError('Copied query identities differ')
            for variant in ['bridge','plain','entity','typed','structure']:
                if read(original/variant/'vocabulary.json')!=read(rebuilt/variant/'vocabulary.json'):
                    raise ValueError('EQ vocabulary differs')
                result['graphs'][f'eq/fold{fold}/{variant}']=graph_pair(original/variant/'graph.pt',rebuilt/variant/'graph.pt')
        print('VERIFIED_PREPARATION_FOLD',fold,flush=True)
    write(a.output,result)


if __name__=='__main__':main()
