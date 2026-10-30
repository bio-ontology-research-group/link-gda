"""Check completed matched checkpoints, inputs, graph mappings and model states."""
import gc
from importlib.metadata import version
from pathlib import Path
import sys
import time

import pandas as pd
import torch
from pykeen.models import TransD
from pykeen.triples import TriplesFactory

BASE = Path('/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029')
RELEASE = BASE/'release-workspace'
WORK = RELEASE/'interim-matched-artifact-check'
SOURCE = RELEASE/'matched-training-source'
sys.path.insert(0, str(SOURCE/'code'))
from phenoquery.common import read, sha, write
from training.run_matched_kge import validate_completion, PRIMARY
from link_gda.checkpoint_identity import graph_identity
from link_gda.head_experiment import validate_splits
sys.path.append(str(RELEASE/'source/reproducibility'))
from build_matched_kge_manifest import verify_selections
sys.path.insert(0, str(SOURCE/'code'))


def context(folder, device='cpu'):
    identity = read(folder/'identity.json')
    settings = identity['settings']
    assert settings['model'] in ('U1', 'T1')
    assert settings['dimension'] == 200 and settings['seed'] == 0
    assert identity['torch'] == str(torch.__version__) and identity['pykeen'] == version('pykeen')
    data = RELEASE/'inputs/benchmark/data'/f"fold{identity['fold']}"
    assert identity['inputs'] == {name:sha(data/name) for name in identity['inputs']}
    genes = (data/'candidates.txt').read_text().splitlines()
    assert len(genes) == len(set(genes)) == 13625
    profiles = read(data/'profiles.json')
    train, val, test = [pd.read_csv(data/(split+'.tsv'), sep='\t')
                        for split in ('training', 'validation', 'test')]
    validate_splits(train, val, test, genes)
    raw = pd.read_csv(data/'raw/train.txt', sep='\t', header=None, dtype=str, names=['h','r','t'])
    kept = raw if identity['graph'] == 'original' else raw[~raw.r.isin(['associated_with','has_symptom'])]
    assert identity['graph'] in ('original', 'nobridge')
    factory = TriplesFactory.from_labeled_triples(kept.values, create_inverse_triples=True)
    assert graph_identity(factory) == identity['graph_sha256']
    assert factory.num_entities == identity['nodes'] and len(kept) == identity['forward_triples']
    assert set(genes) <= set(factory.entity_to_id)
    assert not (set(val.Disease) | set(test.Disease)) & set(factory.entity_to_id)
    if identity['graph'] == 'nobridge':
        assert not set(train.Disease) & set(factory.entity_to_id)
    model = TransD(triples_factory=factory, embedding_dim=200, relation_dim=200,
                   random_seed=0).to(device).eval()
    return identity, factory, genes, profiles, val, model


def main():
    if (WORK/'verification.json').exists():
        raise ValueError('Verification output already exists')
    plan = read(WORK/'plan.json')
    transferred = read(WORK/'TRANSFER_COMPLETE.json')
    assert transferred['status'] == 'transferred'
    assert transferred['plan_sha256'] == sha(WORK/'plan.json')
    assert transferred['runs'] == len(plan['groups']) == 20
    assert transferred['files'] == 140
    assert len({g['relative'] for g in plan['groups']}) == 20
    source_hashes = {str(p.relative_to(SOURCE/'code')):sha(p) for p in sorted((SOURCE/'code').rglob('*.py'))}
    first = plan['groups'][0]
    first_identity = Path(plan['destination'])/first['relative']/'identity.json'
    assert sha(first_identity) == first['files']['identity.json']
    assert source_hashes == read(first_identity)['source']
    protocol_sha = sha(SOURCE/'reproducibility/protocols/matched-kge-20261029.md')
    torch.set_num_threads(4)
    records, total_bytes = [], 0
    started = time.time()
    for group in plan['groups']:
        folder = Path(plan['destination'])/group['relative']
        assert set(p.name for p in folder.iterdir()) == set(group['files'])
        for name, digest in group['files'].items():
            assert sha(folder/name) == digest, str(folder/name)
            total_bytes += (folder/name).stat().st_size
        completed = validate_completion(folder)
        identity, factory, genes, profiles, val, model = context(folder)
        assert identity['source'] == source_hashes
        assert identity['protocol_sha256'] == protocol_sha
        assert completed['primary'] == PRIMARY[identity['settings']['model']]
        choices = verify_selections(folder, completed, identity)
        states = {}
        for name, choice in choices.items():
            state = torch.load(folder/choice['checkpoint'], map_location='cpu', weights_only=True)
            assert state and all(isinstance(value, torch.Tensor) for value in state.values())
            assert all(torch.isfinite(value).all() for value in state.values())
            model.load_state_dict(state, strict=True)
            states[name] = dict(epoch=choice['epoch'], checkpoint_sha256=sha(folder/choice['checkpoint']),
                tensors=len(state), tensor_elements=sum(value.numel() for value in state.values()),
                finite=True, strict_model_load=True)
            del state
        records.append(dict(run=group['relative'], file_sha256=group['files'],
            graph_sha256=identity['graph_sha256'], nodes=factory.num_entities,
            validation_pairs=len(val), validation_diseases=len(set(val.Disease)), checkpoints=states))
        print(group['relative'], 'PASSED', flush=True)
        del model, factory, genes, profiles, val
        gc.collect()
    proof = dict(status='passed',
        scope='Interim checkpoint bytes, finite state tensors, strict model loads, graph mapping and fixed selection checks. No inference, global freeze or production test result claim.',
        training_runs=len(records), files=140, checkpoint_files=sum(len(r['checkpoints']) for r in records),
        artifact_bytes=total_bytes, plan_sha256=sha(WORK/'plan.json'),
        transfer_sha256=sha(WORK/'TRANSFER_COMPLETE.json'), verifier_sha256=sha(__file__),
        selection_verifier_sha256=sha(RELEASE/'source/reproducibility/build_matched_kge_manifest.py'),
        training_source_files=source_hashes, protocol_sha256=protocol_sha,
        torch=str(torch.__version__), pykeen=version('pykeen'),
        seconds=time.time()-started, records=records)
    assert proof['checkpoint_files'] == 80
    write(WORK/'verification.json', proof)
    print({k:v for k,v in proof.items() if k not in ('records','training_source_files')}, flush=True)


if __name__ == '__main__':
    main()
