"""Explicit unqualified CLMPT execution variant with consolidated embedding lookups."""
from collections import defaultdict
from types import MethodType, SimpleNamespace
import hashlib
from pathlib import Path

import torch


EXPECTED_MODEL_SHA256 = 'aba29919d70e4d15df894bc0bb487b2978a69141cbd49d429775acb05c4d91d4'


def packed_encode(self, queries):
    """Keep original bucket/message order while gathering embeddings once per table."""
    groups = defaultdict(list)
    for index, query in enumerate(queries):
        groups[self.signature(query)].append(index)
    plans = []
    entity_ids = []
    relation_ids = []
    entity_sizes = []
    relation_sizes = []
    for indices in groups.values():
        bucket = [queries[index] for index in indices]
        first = bucket[0]
        assert not set(first['constants']) & {'f', 'e1', 'e2', 'e3'}
        for query in bucket:
            assert all(0 <= int(value) < self.num_entities for value in query['constants'].values())
            if query['type'] == 'profile':
                assert query['hp_ids'] and len(set(query['hp_ids'])) == len(query['hp_ids'])
                assert set(query['hp_ids']) <= self.allowed_hp_ids
                assert sorted(query['constants'].values()) == sorted(query['hp_ids'])
                assert all(atom[2] == 'e1' for atom in query['atoms'][:-1])
                assert query['atoms'][-1][0:3:2] == ['e1', 'f']
        terms = {tail for head, role, tail in first['atoms']} | {head for head, role, tail in first['atoms']}
        assert 'f' in terms
        constants = {}
        for term in sorted(terms):
            if term in first['constants']:
                begin = len(entity_ids)
                entity_ids.extend(query['constants'][term] for query in bucket)
                constants[term] = len(entity_sizes)
                entity_sizes.append(len(bucket))
            elif term != 'f':
                assert term.startswith('e') and term[1:].isdigit()
        atoms = {}
        role_slices = {}

        for index, (head, role, tail) in enumerate(first['atoms']):
            name = 'r'+str(index)
            atoms[name] = SimpleNamespace(head=SimpleNamespace(name=head), tail=SimpleNamespace(name=tail), relation=name, negated=False)
            roles = [query['atoms'][index][1] for query in bucket]
            assert all(isinstance(value, int) and 0 <= value < self.num_relations for value in roles)
            begin = len(relation_ids)
            relation_ids.extend(roles)
            role_slices[name] = len(relation_sizes)
            relation_sizes.append(len(bucket))
        plans.append((indices, terms, constants, atoms, role_slices, len(terms-set(first['constants']))))
    entities = self.nbp.get_entity_emb(entity_ids).split(entity_sizes) if entity_ids else ()
    relations = self.nbp.get_relation_emb(relation_ids).split(relation_sizes) if relation_ids else ()
    results = {}
    for indices, terms, constants, atoms, role_slices, rounds in plans:
        local = {}
        for term in sorted(terms):
            if term in constants:
                local[term] = entities[constants[term]]
            elif term == 'f':
                local[term] = self.layer.free_embedding
            else:
                local[term] = self.layer.existential_embedding
        pred = {name: relations[index] for name, index in role_slices.items()}
        inverse = {}
        for _ in range(rounds):
            local = self.layer(local, atoms, pred, inverse)
        values = local['f']
        for position, index in enumerate(indices):
            results[index] = values[position]
    return torch.stack([results[index] for index in range(len(queries))])


def install(model):
    """Install only on the pinned original model, without changing parameter state keys."""
    source = Path(model.encode.__func__.__code__.co_filename)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_MODEL_SHA256
    references = source.parent.parent/'references'
    pins = {'official-clmpt.py': '9f709cae7c91116a53364b6c5127bc83b64a309a9f1d0c64c6a48684bf717c23', 'official-complex.py': 'ec1704cb593dc8a4e47374f8c93f2e3bce0078eb7924bd66186b44acd8adfaf2', 'official-transformer.py': '88b205f6fce3971551a888df352c743c202a1fc6e7baca31d80e32ee8a58e9f2'}
    for name, digest in pins.items():
        assert hashlib.sha256((references/name).read_bytes()).hexdigest() == digest
    assert not hasattr(model, '_unqualified_packed_lookup_variant')
    model.encode = MethodType(packed_encode, model)
    model._unqualified_packed_lookup_variant = True
    return model
