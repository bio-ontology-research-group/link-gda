"""Immutable source and cross-seed selection gates for query campaigns."""
from pathlib import Path
from .common import read, sha, write


def source_text(code):
    code = Path(code)
    files = sorted((code / 'phenoquery').glob('*.py')) + sorted(
        code / 'baselines' / name for name in
        ['run_query_campaign.py', 'run_controlled_query.py', 'run_phenoquery.py', 'run_ultra_matched.py'])
    return ''.join(f'{sha(p)}  {p.relative_to(code)}\n' for p in files)


def verify_source(root, code):
    if (Path(root) / 'source.sha256').read_text() != source_text(code):
        raise ValueError('Executed source differs from frozen campaign source')


def verify_gates(root):
    root = Path(root)
    plan = read(root / 'CONTROLLED.json')
    for initialization in plan['initializations']:
        gate = read(root / f'GATE-{initialization}.json')
        if gate['source_sha256'] != plan['source_sha256'] or gate['steps'] != 4:
            raise ValueError('Technical gate does not cover this production source')
        if gate.get('validation_boundary',{}).get('steps') != 4:
            raise ValueError('Technical gate lacks validation-boundary resumption')
        for name, digest in gate['artifacts'].items():
            if sha(root / name) != digest:
                raise ValueError(f'Technical gate artifact changed: {name}')


def freeze_global(root):
    root = Path(root)
    plan = read(root / 'CONTROLLED.json')
    selections = {}
    count = 0
    for name in plan['campaigns']:
        child = root / name
        selection = read(child / 'SELECTION.json')
        if len(selection['runs']) != plan['runs_per_campaign']:
            raise ValueError(f'Incomplete selection: {name}')
        for run, record in selection['runs'].items():
            folder = child / 'training' / run
            if sha(folder / 'COMPLETE.json') != record['completion_sha256']:
                raise ValueError(f'Changed completion: {name}/{run}')
            completed = read(folder / 'COMPLETE.json')
            for file, key in [('best.pt', 'checkpoint_sha256'),
                              ('initial.pt', 'initial_checkpoint_sha256'),
                              ('validation.npz', 'validation_sha256'),
                              ('initial-validation.npz', 'initial_validation_sha256')]:
                if sha(folder / file) != completed[key]:
                    raise ValueError(f'Changed artifact: {name}/{run}/{file}')
            count += 1
        selections[name] = sha(child / 'SELECTION.json')
    if count != plan['expected_runs']:
        raise ValueError('Incomplete global selection')
    result = {'plan_sha256': sha(root / 'CONTROLLED.json'), 'runs': count,
              'selections': selections, 'selection_data': 'validation only'}
    destination = root / 'GLOBAL-SELECTION.json'
    if destination.exists() and read(destination) != result:
        raise ValueError('Global selection changed')
    write(destination, result)
    return sha(destination)


def verify_global(child, expected_sha):
    child = Path(child)
    root = child.parent
    path = root / 'GLOBAL-SELECTION.json'
    if sha(path) != expected_sha:
        raise ValueError('Global selection checksum mismatch')
    selection = read(path)
    plan = read(root / 'CONTROLLED.json')
    if selection['plan_sha256'] != sha(root / 'CONTROLLED.json'):
        raise ValueError('Controlled plan changed')
    if selection['runs'] != plan['expected_runs'] or set(selection['selections']) != set(plan['campaigns']):
        raise ValueError('Incomplete global selection')
    for name, digest in selection['selections'].items():
        if sha(root / name / 'SELECTION.json') != digest:
            raise ValueError('A campaign selection changed after the global freeze')
    if child.name not in selection['selections']:
        raise ValueError('Campaign is absent from global selection')
