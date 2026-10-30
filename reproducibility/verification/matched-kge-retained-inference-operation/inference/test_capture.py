"""Capture gates tested against retained production schemas and bad inventories."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
spec = importlib.util.spec_from_file_location('capture_matched', ROOT/'capture-native.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def accounting():
    return '\n'.join(f'123_{i}|{200+i}|COMPLETED|0:0' for i in range(3))


def test_complete_accounting():
    assert capture.native_tasks(accounting(), 123) == {fold: 200 + fold % 3 for fold in range(10)}


def test_actual_launcher_runs_every_fold_once_in_three_allocations():
    text = (ROOT/'run-inference.sbatch').read_text()
    assert '#SBATCH --array=0-2\n' in text
    opening = text[text.index('for (('):].splitlines()[0]
    observed = []
    for task in range(3):
        program = f'SLURM_ARRAY_TASK_ID={task}\n' + opening + '\nprintf "%s\\n" "$fold"\ndone\n'
        actual = subprocess.check_output(['bash', '-c', program], text=True)
        folds = [int(x) for x in actual.splitlines()]
        assert folds == list(range(task, 10, 3))
        observed.extend(folds)
    assert sorted(observed) == list(range(10))


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'failed', 'pending', 'exit',
                                  'wrong_array', 'extra', 'raw_reused', 'raw_zero', 'bad_suffix'])
def test_accounting_rejects_incomplete_or_wrong_identity(kind):
    lines = accounting().splitlines()
    if kind == 'missing': lines.pop()
    if kind == 'duplicate': lines.append(lines[0])
    if kind == 'failed': lines[0] = lines[0].replace('COMPLETED', 'FAILED')
    if kind == 'pending': lines[0] = lines[0].replace('COMPLETED', 'PENDING')
    if kind == 'exit': lines[0] = lines[0].replace('0:0', '1:0')
    if kind == 'wrong_array': lines[0] = lines[0].replace('123_', '124_')
    if kind == 'extra': lines.append('123_10|210|COMPLETED|0:0')
    if kind == 'raw_reused': lines[0] = lines[0].replace('|200|', '|201|')
    if kind == 'raw_zero': lines[0] = lines[0].replace('|200|', '|0|')
    if kind == 'bad_suffix': lines[0] = lines[0].replace('123_0', '123_0_0')
    with pytest.raises(ValueError): capture.native_tasks('\n'.join(lines), 123)


def production_records(fold):
    retained = json.loads((BASE/'fixtures/native-score-metadata.json').read_text())['files']
    records = []
    for run in sorted(capture.run_names(fold)):
        actual = retained[run+'/SCORED.json']
        scored = json.loads(actual['contents'])
        records.append(dict(run=run, scored_sha256=actual['sha256'], files=scored['files']))
    return dict(status='inference_completed', fold=fold, training_runs=10, score_files=40, records=records)


@pytest.mark.parametrize('fold', range(10))
def test_all_production_folds_have_expected_score_schema(fold):
    complete = production_records(fold)
    assert len(capture.validate_records(complete, fold)) == 10


@pytest.mark.parametrize('kind', ['missing_run', 'duplicate_run', 'wrong_fold', 'missing_score',
                                  'extra_score', 'digest', 'score_count', 'not_complete'])
def test_score_inventory_rejects_mutated_production(kind):
    d = copy.deepcopy(production_records(0))
    if kind == 'missing_run': d['records'].pop()
    if kind == 'duplicate_run': d['records'][0] = d['records'][1]
    if kind == 'wrong_fold': d['records'][0]['run'] = d['records'][0]['run'].replace('fold0', 'fold1')
    if kind == 'missing_score': d['records'][0]['files'].pop('scores-raw_mr.npz')
    if kind == 'extra_score': d['records'][0]['files']['extra.npz'] = 'a'*64
    if kind == 'digest': d['records'][0]['scored_sha256'] = '../wrong'
    if kind == 'score_count': d['score_files'] = 39
    if kind == 'not_complete': d['status'] = 'running'
    with pytest.raises(ValueError): capture.validate_records(d, 0)


@pytest.fixture
def workspace(tmp_path):
    manifest = json.loads((ROOT/'checkpoint-release-manifest.json').read_text())
    public = {n:r for c in manifest['components'] for n,r in c['files'].items()}
    public.update(json.loads((ROOT/'INPUTS_REFERENCE.json').read_text())['files'])
    assert len(public) == 843
    retained = json.loads((BASE/'fixtures/native-score-metadata.json').read_text())['files']
    # Real published input names and native score names; zero-sized test placeholders only.
    names = set(public) | {'matched-kge/'+n for n in retained
                          if n.endswith('/SCORED.json') or '/scores-' in n}
    names |= {'matched-kge/'+run+'/run.lock' for f in range(10) for run in capture.run_names(f)}
    assert len(names) == 1443
    for name in names:
        p = tmp_path/name; p.parent.mkdir(parents=True, exist_ok=True); p.touch()
    return tmp_path, public


def test_actual_closed_inventory(workspace):
    capture.workspace_inventory(*workspace)


@pytest.mark.parametrize('kind', ['missing', 'extra', 'partial', 'symlink', 'nonempty_lock'])
def test_workspace_rejects_inventory_changes(workspace, kind):
    root, public = workspace
    p = root/'matched-kge/fold0/original/U1/scores-raw_mr.npz'
    if kind == 'missing': p.unlink()
    if kind == 'extra': (root/'unexpected').touch()
    if kind == 'partial': p.with_suffix('.partial').touch()
    if kind == 'nonempty_lock': (root/'matched-kge/fold0/original/U1/run.lock').write_text('unexpected content')
    if kind == 'symlink':
        p.unlink(); p.symlink_to(root/'matched-kge/FROZEN.json')
    with pytest.raises(ValueError): capture.workspace_inventory(root, public)
