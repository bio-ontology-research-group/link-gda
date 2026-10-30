"""Original matched validation context, with explicit native data/source roots."""
from importlib.metadata import version
from pathlib import Path
import sys
import pandas as pd
import torch
from pykeen.models import TransD
from pykeen.triples import TriplesFactory
WORK = Path(__file__).resolve().parent
SOURCE = Path('/mnt/data/u/hohndor/matched-kge-20261029/public-full-replay-v1/training-source')
DATA = Path('/mnt/data/u/hohndor/ultra-top-hits-20261023/data')
sys.path.insert(0, str(SOURCE/'code'))
from phenoquery.common import read, sha, write
from link_gda.checkpoint_identity import graph_identity
from link_gda.head_experiment import validate_splits


def context(folder, device='cpu'):
    identity = read(folder/'identity.json')
    settings = identity['settings']
    assert settings['model'] in ('U1', 'T1')
    assert settings['dimension'] == 200 and settings['seed'] == 0
    assert identity['torch'] == str(torch.__version__) and identity['pykeen'] == version('pykeen')
    data = DATA/f"fold{identity['fold']}"
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
