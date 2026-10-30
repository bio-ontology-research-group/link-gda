"""Check the captured matched-KGE packages and exercise their CPU runtime."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text())
    if platform.python_version() != capture['python']:
        raise ValueError('Python version differs from the training environment')
    expected = {canonical(d['name']): d['version'] for d in capture['distributions']}
    distributions = list(importlib.metadata.distributions())
    installed = {canonical(d.metadata['Name']): d.version for d in distributions}
    if len(expected) != len(capture['distributions']) or len(installed) != len(distributions):
        raise ValueError('Duplicate Python distribution identities')
    if installed != expected:
        differences = {name: (expected.get(name), installed.get(name))
            for name in sorted(expected.keys() | installed.keys())
            if expected.get(name) != installed.get(name)}
        raise ValueError('Python distributions differ: ' + str(differences))
    records = {r['name']: r for path in (Path(sys.prefix) / 'conda-meta').glob('*.json')
               for r in [json.loads(path.read_text())]}
    if set(records) != {r['name'] for r in capture['conda']}:
        raise ValueError('Conda package set differs')
    for record in capture['conda']:
        for field in ['version', 'build', 'md5', 'sha256']:
            if record[field] != records[record['name']].get(field):
                raise ValueError('Conda package identity differs: ' + record['name'] + '/' + field)

    import numpy as np
    import torch
    from pykeen.models import ConvKB, TransD
    from pykeen.triples import TriplesFactory
    torch.set_num_threads(1)
    np.testing.assert_array_equal(torch.tensor(np.array([1., 2.])).numpy(), [1., 2.])
    factory = TriplesFactory.from_labeled_triples(
        np.array([['g0', 'r', 'p0'], ['g1', 'r', 'p1']]), create_inverse_triples=True)
    checked = {}
    for model_type, options in [(TransD, {'relation_dim': 8}),
                                (ConvKB, {'num_filters': 2, 'hidden_dropout_rate': 0.})]:
        model = model_type(triples_factory=factory, embedding_dim=8, random_seed=0, **options)
        scores = model.score_hrt(factory.mapped_triples)
        if scores.shape != (2, 1) or not torch.isfinite(scores).all():
            raise ValueError('Invalid smoke-test scores: ' + model_type.__name__)
        (-scores.mean()).backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Invalid smoke-test gradients: ' + model_type.__name__)
        checked[model_type.__name__] = dict(finite_scores=True, finite_gradients=True)

    import mowl
    mowl.init_jvm('512m')
    from java.lang import System
    from org.semanticweb.owlapi.apibinding import OWLManager
    java_home = Path(str(System.getProperty('java.home'))).resolve()
    java_home.relative_to(Path(sys.prefix).resolve())
    java_version = str(System.getProperty('java.version'))
    if not java_version.startswith('17.'):
        raise ValueError('Expected the pinned OpenJDK 17 runtime')
    manager = OWLManager.createOWLOntologyManager()
    ontology = manager.createOntology()
    if ontology.getAxiomCount() != 0:
        raise ValueError('Unexpected initial OWL ontology')
    result = dict(status='passed', capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), conda_packages=len(records), python_distributions=len(installed),
        all_package_identities_equal=True, torch=str(torch.__version__), numpy=np.__version__,
        pykeen=importlib.metadata.version('pykeen'), models=checked,
        numpy_torch_interchange=True, mowl_java_owlapi=True,
        java_version=java_version, java_from_reconstructed_environment=True,
        scope='Exact captured package identities; real CPU TransD/ConvKB scoring and gradients; NumPy interchange and Java OWLAPI. GPU trajectory and full prediction parity remain separate checks.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
