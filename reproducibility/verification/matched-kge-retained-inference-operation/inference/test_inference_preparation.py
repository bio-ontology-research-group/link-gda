"""Verify full campaign coverage, frozen options and retained-output checks."""
import copy
import ast
import importlib.util
from pathlib import Path
import shutil

import pytest
from test_matched_kge_manifest import fixture

spec = importlib.util.spec_from_file_location('matched_public_inference', Path(__file__).with_name('run-inference.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


@pytest.fixture(scope='module')
def campaign(tmp_path_factory):
    return fixture(tmp_path_factory.mktemp('inference-campaign'))


def test_complete_design_and_original_scorer_options(campaign):
    root,data = campaign
    runs = [r for fold in range(10) for r in m.runs_for_fold(fold)]
    assert len(runs) == len(set(runs)) == 100
    assert set(runs) == set(m.read(root/'FROZEN.json')['training'])
    for run in runs:
        identity = m.read(root/run/'identity.json')
        command = m.score_command(identity, root, data, data.parent/'gene_phenotypes.csv')
        options = dict(zip(command[3::2],command[4::2]))
        assert options['--phase'] == 'score'
        assert options['--fold'] == str(identity['fold'])
        assert options['--graph'] == identity['graph']
        for name in ('model','device','dimension','seed','epochs','profile_epochs',
                     'validation_frequency','checkpoint_frequency','batch_size','sub_batch_size','expected_candidates'):
            assert options['--'+name.replace('_','-')] == str(identity['settings'][name])
        assert options['--root'] == str(root) and options['--data-root'] == str(data)


@pytest.mark.parametrize('field,value', [('dimension',17),('epochs',20),('device','cpu'),('early_stopping',True)])
def test_refuses_changed_fixed_training_settings(campaign, field, value):
    root,data = campaign
    identity = m.read(root/'fold0/original/U1/identity.json')
    identity['settings'][field] = value
    with pytest.raises(AssertionError):
        m.score_command(identity, root, data, data.parent/'gene_phenotypes.csv')


def test_clean_start_rejects_native_scores_and_partial_predictions(campaign,tmp_path):
    source,_ = campaign
    root=tmp_path/'runs'; shutil.copytree(source/'fold0',root/'fold0')
    with pytest.raises(AssertionError): m.clean_start(root,0)
    for run in m.runs_for_fold(0):
        folder=root/run
        (folder/'SCORED.json').unlink()
        for path in folder.glob('scores-*'): path.unlink()
    m.clean_start(root,0)
    (root/'fold0/original/U1/scores-raw_head.npz.partial').write_bytes(b'unattributed output')
    with pytest.raises(AssertionError): m.clean_start(root,0)


def test_completed_output_checks_all_four_choices(campaign):
    root,_=campaign
    record=m.completed_run(root,'fold0/original/U1',m.sha(root/'FROZEN.json'))
    assert len(record['files']) == 4


@pytest.mark.parametrize('change',['score-bytes','missing-choice','freeze','training','identity','symlink'])
def test_retained_outputs_cannot_drift(campaign,tmp_path,change):
    source,_=campaign
    relative='fold0/original/U1'
    root=tmp_path/'runs'; folder=root/relative
    shutil.copytree(source/relative,folder)
    frozen_digest=m.sha(source/'FROZEN.json')
    scored=m.read(folder/'SCORED.json')
    if change=='score-bytes': (folder/'scores-raw_head.npz').write_bytes(b'changed')
    elif change=='missing-choice':
        scored['files'].pop('scores-raw_head.npz'); m.write(folder/'SCORED.json',scored)
    elif change=='freeze':
        scored['freeze_sha256']='0'*64; m.write(folder/'SCORED.json',scored)
    elif change=='training':
        scored['training_sha256']='0'*64; m.write(folder/'SCORED.json',scored)
    elif change=='identity': (folder/'identity.json').write_text('{}')
    else:
        path=folder/'scores-raw_head.npz'; target=tmp_path/'outside.npz'
        path.rename(target); path.symlink_to(target)
    with pytest.raises((AssertionError,ValueError)):
        m.completed_run(root,relative,frozen_digest)


def real_members():
    return m.read(Path(__file__).with_name('test-member-inventory.json'))


def sample():
    return m.read(Path(__file__).with_name('validation-verification.json'))


def test_every_public_member_is_hashed_by_at_least_one_fold():
    files = real_members()
    groups = [m.members_for_fold(files, fold) for fold in range(10)]
    assert set().union(*(g.keys() for g in groups)) == set(files)
    assert all(len(g) == 87 for g in groups)
    assert sum(sum(n.startswith('matched-kge/') and n.endswith('.pt') for n in g) for g in groups) == 400
    shared = set.intersection(*(set(g) for g in groups))
    assert shared == {'matched-kge/FROZEN.json', 'matched-kge/inputs/PROVENANCE.json',
                      'matched-kge/inputs/gene_phenotypes.csv'}


@pytest.mark.parametrize('change', ['missing', 'traversal', 'absolute', 'prediction', 'digest', 'size', 'fold'])
def test_rejects_incomplete_or_unsafe_relay_inventory(change):
    files = real_members()
    name = 'matched-kge/FROZEN.json'
    fold = 0
    if change == 'missing': files.pop(name)
    elif change == 'traversal': files['matched-kge/../FROZEN.json'] = files.pop(name)
    elif change == 'absolute': files['/' + name] = files.pop(name)
    elif change == 'prediction': files['matched-kge/fold0/original/U1/SCORED.json'] = files.pop(name)
    elif change == 'digest': files[name]['sha256'] = 'unverified'
    elif change == 'size': files[name]['bytes'] = -1
    else: fold = 10
    with pytest.raises(AssertionError): m.members_for_fold(files, fold)


def test_actual_completed_h200_sample_is_accepted():
    p = sample()
    m.validate_sample(p, p['source_proof_sha256'])


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'job', 'threads', 'source', 'metrics'])
def test_rejects_invalid_validation_evidence(change):
    p = sample()
    digest = p['source_proof_sha256']
    if change == 'missing': p['records'].pop()
    elif change == 'duplicate': p['records'][0] = copy.deepcopy(p['records'][1])
    elif change == 'job': p['job_id'] += 1
    elif change == 'threads': p['cpu_threads'] = 4
    elif change == 'source': digest = '0'*64
    else: p['records'][0]['actual']['raw']['mr'] += 1
    with pytest.raises(AssertionError): m.validate_sample(p, digest)


def runtime():
    p = sample()['runtime']
    return dict(torch=p['torch'], numpy=p['numpy'], pykeen=p['pykeen'], gpu=p['device'],
                capability=p['capability'], threads=8, cpus=8,
                matmul_tf32=p['tf32_matmul'], cudnn_tf32=p['tf32_cudnn'])


def test_accepts_the_measured_h200_runtime():
    m.validate_runtime(runtime(), sample())


@pytest.mark.parametrize('field,value', [('torch','2.9'), ('numpy','1.26'), ('pykeen','1.10'),
                                        ('gpu','NVIDIA A100'), ('capability',[8,0]),
                                        ('threads',4), ('cpus',4), ('matmul_tf32',True),
                                        ('cudnn_tf32',False)])
def test_rejects_changed_runtime(field, value):
    r = runtime(); r[field] = value
    with pytest.raises(AssertionError): m.validate_runtime(r, sample())


def test_original_scorer_and_output_checks_are_unchanged():
    root = Path(__file__).resolve().parent
    before = (root.parent/'fixtures/prior-inference.py').read_text()
    after = (root/'run-inference.py').read_text()
    def function(text, name):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(text, node)
    for name in ('runs_for_fold', 'score_command', 'clean_start', 'completed_run'):
        assert function(before, name) == function(after, name)
    def loop(text):
        return text[text.index('    for relative in runs_for_fold(fold):', text.index('def main():')):
                    text.index("    assert sha(runs/'FROZEN.json') == frozen_digest")]
    assert loop(before) == loop(after)
