#!/usr/bin/env python3
"""Rebuild ULTRA caches from the frozen raw benchmark triples, without using old caches."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phenoquery.common import read, write, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--ultra-root',type=Path,required=True)
    p.add_argument('--folds',type=int,nargs='+',default=list(range(10)))
    a=p.parse_args()
    if not a.folds or len(set(a.folds))!=len(a.folds) or min(a.folds)<0 or max(a.folds)>9:
        p.error('Choose distinct fold IDs in 0..9')
    from phenoquery.ultra_io import build_dataset
    for fold in a.folds:
        source=a.base/'data'/f'fold{fold}'
        destination=a.root/'data'/f'fold{fold}'
        provenance=read(source/'preparation.json')
        for filename,key in [('raw/train.txt','graph'),('profiles.json','profiles'),
                             ('candidates.txt','candidates'),('training.tsv','training'),
                             ('validation.tsv','validation'),('test.tsv','test')]:
            if sha(source/filename)!=provenance[key+'_sha256']:
                raise ValueError(f'Input checksum mismatch: fold{fold}/{filename}')
        destination.mkdir(parents=True,exist_ok=False)
        shutil.copytree(source/'raw',destination/'raw')
        for filename in ['profiles.json','candidates.txt','training.tsv','validation.tsv','test.tsv']:
            shutil.copy2(source/filename,destination/filename)
        dataset,entities,relations=build_dataset(str(a.ultra_root),str(a.root/'data'),f'fold{fold}')
        vocabulary={'entities':entities,'relations':relations}
        if vocabulary!=read(source/'vocabulary.json'):
            raise ValueError('Rebuilt vocabulary differs from retained vocabulary')
        # Preserve vocabulary ordering, but record the actual rebuilt cache hash.
        # The prior preparation remains available as explicit source provenance.
        shutil.copy2(source/'vocabulary.json',destination/'vocabulary.json')
        shutil.copy2(source/'preparation.json',destination/'source-preparation.json')
        write(destination/'preparation.json',dict(provenance,
            processed_sha256=sha(destination/'processed/data.pt')))
        write(destination/'cache-rebuild.json',{'fold':fold,'source_sha256':sha(__file__),
            'raw_sha256':sha(destination/'raw/train.txt'),
            'cache_sha256':sha(destination/'processed/data.pt'),
            'original_cache_sha256':provenance['processed_sha256'],
            'nodes':int(dataset[0].num_nodes),'edges':int(dataset[0].num_edges)})
        print('REBUILT_BENCHMARK_CACHE',fold,flush=True)


if __name__=='__main__':main()
