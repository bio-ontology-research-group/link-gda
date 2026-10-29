import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from analysis.summarize_matched_kge import summarize
from phenoquery.common import read, sha, write
from reproduce.artifacts import fetch, pack
from test_matched_kge_manifest import fixture as training_fixture
from test_matched_kge_report import evaluated_fixture

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'reproducibility'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('build_matched_kge_scores')
tables = load('generate_matched_kge_tables')


@pytest.fixture(scope='module')
def complete_campaign(tmp_path_factory, evaluated_fixture):
    base = tmp_path_factory.mktemp('matched-score-export')
    root, data = training_fixture(base)
    shutil.rmtree(data)
    shutil.copytree(evaluated_fixture/'workspace/data', data)
    model_scores = read(evaluated_fixture/'manifest.json')['models']
    frozen = read(root/'FROZEN.json'); frozen['schema'] = 1
    for run in frozen['training']:
        folder = root/run
        fold, graph, family = run.split('/')
        identity = read(folder/'identity.json')
        identity['settings']['expected_candidates'] = 17
        identity['inputs'] = {name:sha(data/fold/name) for name in identity['inputs']}
        write(folder/'identity.json', identity)
        trained = read(folder/'TRAINED.json')
        trained['identity_sha256'] = sha(folder/'identity.json')
        write(folder/'TRAINED.json', trained)
        frozen['training'][run] = sha(folder/'TRAINED.json')
        for model in model_scores:
            if model['family'] == family and model['graph'] == graph:
                original = evaluated_fixture/model['folds'][fold[4:]]['path']
                shutil.copyfile(original, folder/f"scores-{model['choice']}.npz")
    write(root/'FROZEN.json', frozen)
    for run, digest in frozen['training'].items():
        folder = root/run
        scored = read(folder/'SCORED.json')
        scored.update(training_sha256=digest, freeze_sha256=sha(root/'FROZEN.json'),
                      files={p.name:sha(p) for p in folder.glob('scores-*.npz')})
        write(folder/'SCORED.json', scored)
    manifest = root/'score-manifest.json'
    write(manifest, m.build_manifest(root, data, manifest, base/'gene_phenotypes.csv', expected_candidates=17))
    result = subprocess.run([sys.executable, str(ROOT/'code/analysis/evaluate_stepwise_bias.py'),
        '--base-dir', str(base), '--score-manifest', str(manifest), '--out-json', str(base/'evaluation.json'),
        '--ranks-dir', str(base/'ranks'), '--workers', '2'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    write(base/'summary.json', summarize(manifest, base/'evaluation.json', base/'ranks', data,
                                        expected_candidates=17))
    return base


def export(base, work, **kwargs):
    return m.build(base/'campaign', base/'data', base/'campaign/score-manifest.json',
        base/'gene_phenotypes.csv', base/'evaluation.json', base/'summary.json', base/'ranks',
        work/'verification', workers=2, expected_candidates=17, **kwargs)


def test_roundtrip_replays_every_condition_and_both_tables_without_originals(complete_campaign, tmp_path):
    base = tmp_path/'original'; shutil.copytree(complete_campaign, base)
    (base/'campaign/private-unlisted').write_text('not a release input')
    plan = export(base, tmp_path, max_part_bytes=1_000_000)
    expected = plan['provenance']['expected_files']
    assert len(expected) == 886
    assert sum('/scores-' in p for p in expected) == 400
    assert not any(p.endswith('.pt') or 'private-unlisted' in p for p in expected)
    assert sum(p.endswith('/TRAINED.json') for p in expected) == 100
    assert sum(p.endswith('/validation.json') for p in expected) == 100
    plan_file, archives = tmp_path/'plan.json', tmp_path/'archives'
    write(plan_file, plan); pack(plan_file, archives, archives.as_uri())
    assert m.verify_packed(plan_file, archives/'release-manifest.json')['files'] == 886
    restored = tmp_path/'restored'
    fetch(archives/'release-manifest.json', restored, tmp_path/'downloads')
    shutil.rmtree(base)
    assert not any(p.is_symlink() for p in restored.rglob('*'))
    p = plan['provenance']
    proof = m.replay(restored/p['score_manifest'], restored/p['data_root'],
        restored/p['reference_report'], restored/p['reference_summary'], restored/p['reference_ranks'],
        tmp_path/'fresh-replay', workers=2, expected_candidates=17)
    assert proof['models'] == 40 and proof['conditions_per_model'] == 42
    assert proof['primary_comparisons'] == 10 and proof['rank_files'] == 20
    reference = restored/p['reference_summary']; actual = tmp_path/'fresh-replay/summary.json'
    assert reference.read_bytes() == actual.read_bytes()
    expected_tables = tables.render(read(reference), sha(reference))
    assert len(expected_tables) == 2
    assert expected_tables == tables.render(read(actual), sha(actual))


@pytest.mark.parametrize('problem', ['nonprimary-condition', 'nonprimary-rank', 'summary',
                                    'checkpoint', 'score', 'validation', 'manifest-location'])
def test_export_rejects_invalid_campaign_or_analysis(complete_campaign, tmp_path, problem):
    base = tmp_path/'original'; shutil.copytree(complete_campaign, base)
    if problem == 'nonprimary-condition':
        path = base/'evaluation.json'; report = read(path)
        report['MatchedKGE (U1, original, raw_head)']['S0_Raw_All']['mr'] += 1
        write(path, report)
    elif problem == 'nonprimary-rank':
        path = base/'ranks/fold0-ranks.npz'
        with np.load(path) as z: arrays = {k:z[k] for k in z.files}
        arrays['MatchedKGE (U1, original, raw_head)::r_s0'][0] += 1
        np.savez_compressed(path, **arrays)
    elif problem == 'summary':
        path = base/'summary.json'; result = read(path)
        result['contrasts'][0]['upper'] += 1; write(path, result)
    elif problem == 'manifest-location':
        path = base/'campaign/score-manifest.json'; path.rename(base/'score-manifest.json')
        with pytest.raises(ValueError, match='directly inside'):
            m.build(base/'campaign', base/'data', base/'score-manifest.json', base/'gene_phenotypes.csv',
                base/'evaluation.json', base/'summary.json', base/'ranks', tmp_path/'verification')
        return
    else:
        names = dict(checkpoint='selected_raw_head.pt', score='scores-raw_head.npz', validation='validation.json')
        (base/'campaign/fold0/original/U1'/names[problem]).write_bytes(b'changed')
    with pytest.raises(ValueError):
        export(base, tmp_path)


def test_archive_verification_rejects_mutation_after_export(complete_campaign, tmp_path):
    base = tmp_path/'original'; shutil.copytree(complete_campaign, base)
    plan = export(base, tmp_path)
    file = base/'campaign/fold0/original/U1/scores-raw_head.npz'
    file.write_bytes(b'changed after planning')
    plan_file, archives = tmp_path/'plan.json', tmp_path/'archives'
    write(plan_file, plan); pack(plan_file, archives, archives.as_uri())
    with pytest.raises(ValueError, match='Packed files differ'):
        m.verify_packed(plan_file, archives/'release-manifest.json')
