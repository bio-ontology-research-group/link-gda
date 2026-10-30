import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('matched_manifest', ROOT / 'reproducibility/build_matched_kge_manifest.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotations = tmp_path / 'gene_phenotypes.csv'
    annotations.write_text('Gene,Phenotype\ng0,MP:0000001\n')
    root, data = tmp_path / 'campaign', tmp_path / 'data'
    frozen = dict(folds=list(range(10)), models=list(m.MODELS), graphs=list(m.GRAPHS), training={})
    for fold in range(10):
        d = data / f'fold{fold}'
        (d / 'raw').mkdir(parents=True)
        for file in m.INPUTS:
            (d / file).write_text(f'synthetic {fold} {file}')
        (d / 'candidates.txt').write_text(''.join(f'g{i}\n' for i in range(13625)))
        for graph in m.GRAPHS:
            for model in m.MODELS:
                folder = m.directory(root, fold, graph, model)
                folder.mkdir(parents=True)
                settings = dict(model=model, dimension=400 if model == 'INDIGENA' else 200,
                    seed=0, epochs=1000, validation_frequency=20, batch_size=65536,
                    sub_batch_size=8192, profile_epochs=20, checkpoint_frequency=20,
                    expected_candidates=13625, primary=m.PRIMARY[model], early_stopping=False,
                    t1_sampling='candidate-typed, known training positives excluded', device='cuda')
                identity = dict(fold=fold, graph=graph, settings=settings,
                    inputs={name: m.sha(d / name) for name in m.INPUTS},
                    protocol_sha256=m.sha(m.PROTOCOL), source={'fixture.py': 'synthetic source'},
                    torch='fixture', pykeen='fixture', nodes=15000, forward_triples=20000)
                if model == 'INDIGENA': identity['gene_phenotypes_sha256'] = m.sha(annotations)
                if model in ('PPOOL', 'CONVKB'):
                    choice = 'raw_head' if model == 'PPOOL' else 'calibrated_mr'
                    identity['warm_start_sha256'] = m.sha(m.directory(root, fold, graph, 'U1') / f'selected_{choice}.pt')
                m.write(folder / 'identity.json', identity)
                frequency, epochs = (1, 20) if model == 'PPOOL' else (20, 1000)
                metrics = dict(raw={'hits@1': .01, 'hits@10': .1, 'mrr': .03, 'mr': 500.},
                               calibrated={'hits@1': .02, 'hits@10': .2, 'mrr': .05, 'mr': 400.})
                best = {}
                for choice in m.CHOICES:
                    mode, criterion = choice.split('_')
                    v = metrics[mode]
                    key = [-v['mr']] if criterion == 'mr' else [v['hits@10'], v['hits@1'], v['mrr'], -v['mr']]
                    best[choice] = dict(epoch=frequency, key=key, metrics=v, checkpoint=f'selected_{choice}.pt')
                    (folder / f'selected_{choice}.pt').write_bytes(f'synthetic checkpoint {fold}/{graph}/{model}/{choice}'.encode())
                    # This manifest checks hashes; matrix parsing belongs to the shared evaluator.
                    (folder / f'scores-{choice}.npz').write_bytes(f'synthetic score {fold}/{graph}/{model}/{choice}'.encode())
                m.write(folder / 'validation.json', dict(frequency=frequency, best=best,
                    curve=[dict(epoch=e, metrics=metrics) for e in range(frequency, epochs+1, frequency)]))
                training = dict(epochs=epochs, complete=True)
                if model == 'PPOOL': training['history'] = [{} for _ in range(epochs)]
                else: training['early_stopped'] = False
                m.write(folder / 'TRAINED.json', dict(identity_sha256=m.sha(folder / 'identity.json'),
                    selected=best, files={p.name: m.sha(p) for p in [folder / 'validation.json', *folder.glob('*.pt')]},
                    training=training, primary=m.PRIMARY[model]))
                frozen['training'][str(folder.relative_to(root))] = m.sha(folder / 'TRAINED.json')
    m.write(root / 'FROZEN.json', frozen)
    for run, digest in frozen['training'].items():
        folder = root / run
        m.write(folder / 'SCORED.json', dict(training_sha256=digest, freeze_sha256=m.sha(root / 'FROZEN.json'),
            primary=m.PRIMARY[folder.name], files={p.name: m.sha(p) for p in folder.glob('scores-*.npz')}))
    return root, data


def rebind(root, folder):
    complete = m.read(folder / 'TRAINED.json')
    complete['identity_sha256'] = m.sha(folder / 'identity.json')
    complete['files'] = {name: m.sha(folder / name) for name in complete['files']}
    m.write(folder / 'TRAINED.json', complete)
    frozen = m.read(root / 'FROZEN.json')
    frozen['training'][str(folder.relative_to(root))] = m.sha(folder / 'TRAINED.json')
    m.write(root / 'FROZEN.json', frozen)
    for run, digest in frozen['training'].items():
        p = root / run / 'SCORED.json'; score = m.read(p)
        score.update(training_sha256=digest, freeze_sha256=m.sha(root / 'FROZEN.json'))
        m.write(p, score)


def test_full_manifest_preserves_all_choices_primary_rules_and_relocation(tmp_path):
    root, data = fixture(tmp_path / 'original')
    result = m.build(root, data, tmp_path / 'original/manifest.json', data.parent / 'gene_phenotypes.csv')
    assert len(result['models']) == 40 and len(result['evidence']) == 100
    assert sum(len(model['folds']) for model in result['models']) == 400
    assert sum(model['primary'] for model in result['models']) == 10
    assert {family: value['scoring'] for family, value in result['primary_comparisons'].items()} == {
        'U1': 'calibrated', 'T1': 'raw', 'INDIGENA': 'raw', 'CONVKB': 'calibrated', 'PPOOL': 'raw'}
    shutil.copytree(tmp_path / 'original', tmp_path / 'relocated')
    assert result == m.build(tmp_path / 'relocated/campaign', tmp_path / 'relocated/data', tmp_path / 'relocated/manifest.json', tmp_path / 'relocated/gene_phenotypes.csv')


@pytest.mark.parametrize('problem', ['final-validation', 'later-tie', 'partial-freeze', 'wrong-budget',
    'warm-start', 'source', 'input', 'annotations', 'corrupt-score', 'score-freeze', 'partial-scores'])
def test_semantically_wrong_or_corrupt_campaign_is_rejected(tmp_path, problem):
    root, data = fixture(tmp_path)
    folder = m.directory(root, 0, 'original', 'CONVKB' if problem == 'warm-start' else 'U1')
    if problem in ('final-validation', 'later-tie'):
        curve = m.read(folder / 'validation.json')
        if problem == 'final-validation': curve['curve'].pop()
        else:
            curve['best']['raw_head']['epoch'] = 40
            trained = m.read(folder / 'TRAINED.json'); trained['selected'] = curve['best']
            m.write(folder / 'TRAINED.json', trained)
        m.write(folder / 'validation.json', curve); rebind(root, folder)
    elif problem in ('wrong-budget', 'warm-start', 'source'):
        identity = m.read(folder / 'identity.json')
        if problem == 'wrong-budget': identity['settings']['epochs'] = 500
        elif problem == 'warm-start': identity['warm_start_sha256'] = 'wrong parent'
        else: identity['source'] = {'changed.py': 'different source'}
        m.write(folder / 'identity.json', identity); rebind(root, folder)
    elif problem == 'partial-freeze':
        frozen = m.read(root / 'FROZEN.json'); frozen['training'].pop('fold9/nobridge/PPOOL')
        m.write(root / 'FROZEN.json', frozen)
    elif problem == 'input': (data / 'fold0/test.tsv').write_text('changed')
    elif problem == 'annotations': (data.parent / 'gene_phenotypes.csv').write_text('changed')
    elif problem == 'corrupt-score': (folder / 'scores-raw_head.npz').write_bytes(b'changed')
    else:
        score = m.read(folder / 'SCORED.json')
        if problem == 'score-freeze': score['freeze_sha256'] = 'wrong freeze'
        else: score['files'].pop('scores-calibrated_mr.npz')
        m.write(folder / 'SCORED.json', score)
    with pytest.raises(ValueError):
        m.build(root, data, tmp_path / 'manifest.json', data.parent / 'gene_phenotypes.csv')
