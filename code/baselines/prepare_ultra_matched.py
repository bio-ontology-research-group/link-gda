#!/usr/bin/env python3
"""Rebuild the historical matched EQ-augmented graphs from frozen inputs."""
import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phenoquery.common import sha, read, write, pairs


def prepare(root, base, eq, ultra_root, folds):
    import torch
    from phenoquery.ultra_io import build_dataset
    sys.path.insert(0, str(ultra_root))
    from ultra.tasks import build_relation_graph
    manifest = read(eq/'projection/manifest.json')
    extra = set()
    for name in ['entity.tsv', 'typed.tsv']:
        path = eq/'projection'/name
        if sha(path) != manifest['files'][name]:
            raise ValueError(f'Projection checksum mismatch: {name}')
        with path.open() as handle:
            extra.update(tuple(line.rstrip().split('\t')) for line in handle)
    if not extra: raise ValueError('Empty EQ projection')
    for fold in folds:
        out = root/'graphs'/f'fold{fold}'
        out.mkdir(parents=True, exist_ok=False)
        data = base/'data'/f'fold{fold}'
        provenance = read(data/'preparation.json')
        for filename, key in [('raw/train.txt','graph'), ('profiles.json','profiles'),
                              ('candidates.txt','candidates'), ('training.tsv','training'),
                              ('validation.tsv','validation'), ('test.tsv','test')]:
            if sha(data/filename) != provenance[key+'_sha256']:
                raise ValueError(f'Prepared input checksum mismatch: fold{fold}/{filename}')
        dataset, entities, relations = build_dataset(str(ultra_root), str(base/'data'), f'fold{fold}')
        original = dataset[0]
        held_out = {d for split in ['validation','test'] for _,d in pairs(data/(split+'.tsv'))}
        if held_out & set(entities): raise ValueError('Held-out disease node occurs in graph')
        genes = (data/'candidates.txt').read_text().splitlines()
        if not set(genes) <= set(entities): raise ValueError('Candidate missing from graph')
        n = len(relations)
        entities, relations = dict(entities), dict(relations)
        for head, relation, tail in sorted(extra):
            for node in [head, tail]:
                if node not in entities: entities[node] = len(entities)
            if relation not in relations: relations[relation] = len(relations)
        keep = original.edge_type < n
        old_index, old_type = original.edge_index[:,keep], original.edge_type[keep]
        added = torch.tensor([[entities[h],relations[r],entities[t]] for h,r,t in sorted(extra)], dtype=torch.long)
        if not (added[:,1] >= n).all(): raise ValueError('EQ relation collides with existing relation')
        joined = torch.cat([torch.cat([old_index[:1],old_type[None],old_index[1:]],0).T, added],0)
        index, types = joined[:,[0,2]].T, joined[:,1]
        graph = copy.copy(original)
        graph.edge_index = torch.cat([index,index.flip(0)],1)
        graph.edge_type = torch.cat([types,types+len(relations)])
        graph.num_nodes, graph.num_relations = len(entities), 2*len(relations)
        graph = build_relation_graph(graph)
        torch.save(graph, out/'graph.pt')
        write(out/'vocabulary.json', {'entities':entities, 'relations':relations})
        write(out/'identity.json', {'fold':fold, 'source_preparation':provenance,
            'source_projection_sha256':sha(eq/'projection/manifest.json'),
            'added_projection_edges':len(extra), 'original_forward_edges':len(old_type),
            'new_forward_edges':len(types), 'nodes':len(entities), 'relations':len(relations),
            'files':{name:sha(out/name) for name in ['graph.pt','vocabulary.json']}})
        print('PREPARED', fold, len(entities), len(types), flush=True)
    write(root/'PREPARED.json', {'folds':len(folds), 'fold_ids':folds,
          'source_sha256':sha(__file__), 'projection_sha256':sha(eq/'projection/manifest.json')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='New output directory')
    p.add_argument('--base', type=Path, required=True, help='Frozen benchmark workspace containing data/foldN')
    p.add_argument('--eq', type=Path, required=True, help='Frozen EQ workspace containing projection/')
    p.add_argument('--ultra-root', type=Path, required=True)
    p.add_argument('--folds', type=int, nargs='+', default=list(range(10)))
    a = p.parse_args()
    if len(set(a.folds)) != len(a.folds) or not a.folds or min(a.folds)<0 or max(a.folds)>9:
        p.error('Choose distinct fold IDs in 0..9')
    prepare(a.root, a.base, a.eq, a.ultra_root, a.folds)


if __name__ == '__main__': main()
