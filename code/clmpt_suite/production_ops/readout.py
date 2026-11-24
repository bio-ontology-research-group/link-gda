"""Pool and label-mask readouts of the saved TEST score matrices written by query/code/test_eval.py (numpy, no model).

For each view (initial, raw-selected, calibrated-selected, terminal) the one saved matrix `test_real` (TEST cases x the
bank's candidate genes) is read once. Calibrated scores are computed ONCE on the full matrix with the worker's own
metrics.calibrate (own leave-one-out over every TEST disease); nothing is recalibrated after any filtering. Each pool is
a subset of the same columns (main: the bank pool; excluded: secondary_full_go 18,904 and primary_no_mgi 5,696), ranked
with the worker's metrics.measure (unfiltered, deterministic average ties). Each label mask restricts only which true
associations are summarized ('all'; excluded also 's_supported', the native expression-supported targets): competitor
columns are never removed for a mask. Associations whose gene is outside a pool are counted and excluded from that pool.
The full pool with all labels must reproduce test_eval.py's own metrics exactly; the readout refuses otherwise.
Raw and calibrated metrics are reported at the same checkpoint, and the separately selected points (raw view in raw
scores, calibrated view in calibrated scores) are listed. Masks are recorded by hash for later baseline replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

VIEWS = ('initial', 'raw', 'calibrated', 'terminal')


def digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            value.update(block)
    return value.hexdigest()


def pools_and_masks(cases, candidate_ids):
    pools = {'bank_full': list(candidate_ids)}
    if 'candidate_views' in cases:
        pools = {name: list(values) for name, values in sorted(cases['candidate_views'].items())}
        assert set(pools['secondary_full_go']) == set(candidate_ids), 'full-GO view must equal the scored columns'
    masks = {'all': [list(row['gene_ids']) for row in cases['cases']]}
    if all('s_supported_gene_ids' in row for row in cases['cases']):
        masks['s_supported'] = [[gene for gene in row['gene_ids'] if gene in set(row['s_supported_gene_ids'])] for row in cases['cases']]
    return pools, masks


def readout(metrics, real, cases, candidate_ids, seen):
    """All pools x masks x score types from one matrix, calibration first on the full matrix."""
    scored = {'raw': np.asarray(real, dtype=np.float64)}
    scored['calibrated'] = metrics.calibrate(scored['raw'])
    position = {int(gene): index for index, gene in enumerate(candidate_ids)}
    pools, masks = pools_and_masks(cases, candidate_ids)
    result = {}
    for pool, genes in pools.items():
        columns = [position[int(gene)] for gene in genes]
        inside = set(int(gene) for gene in genes)
        for mask, answers in masks.items():
            kept_rows, kept_answers, outside = [], [], 0
            for row, labels in enumerate(answers):
                usable = [gene for gene in labels if int(gene) in inside]
                outside += len(labels) - len(usable)
                if usable:
                    kept_rows.append(row)
                    kept_answers.append(usable)
            entry = {'associations_in_mask': sum(len(labels) for labels in answers), 'associations_outside_pool': outside,
                     'associations_ranked': sum(len(labels) for labels in kept_answers), 'cases_ranked': len(kept_rows), 'pool_size': len(genes)}
            for score, matrix in scored.items():
                value = metrics.measure(matrix[np.ix_(kept_rows, columns)], kept_answers, [int(gene) for gene in genes], seen_genes=seen)
                entry[score] = {key: value[key] for key in value if key != 'ranks'}
                entry[score]['ranks'] = value['ranks']
            result['%s/%s' % (pool, mask)] = entry
    return result, pools, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--worker-code', required=True, help='query/code directory whose metrics.py is used')
    parser.add_argument('--inference-dir', required=True, help='test_eval.py output directory')
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--bank-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.worker_code).resolve()))
    import metrics
    output = Path(args.output)
    assert not output.exists()
    inference, test_dir = Path(args.inference_dir), Path(args.test_dir)
    cases = json.loads((test_dir / 'test-cases.json').read_text())
    bank = json.loads((Path(args.bank_dir) / 'bank.json').read_text())
    seen = {gene for row in bank['training_profiles'] for gene in row['answers']}
    record = {'test_cases_sha256': sha(test_dir / 'test-cases.json'), 'test_complete_sha256': sha(test_dir / 'COMPLETE.json'), 'metrics_sha256': sha(Path(metrics.__file__)),
              'calibration': 'own leave-one-out over every TEST disease on the full matrix, computed once before any pool or mask restriction',
              'ranking': 'unfiltered, deterministic average ties, worker metrics.measure', 'views': {}}
    for view in VIEWS:
        archive = inference / ('test-%s.npz' % view)
        if not archive.exists():
            continue
        saved = np.load(archive)
        real, candidate_ids = saved['test_real'], [int(gene) for gene in saved['candidate_ids']]
        assert candidate_ids == [int(gene) for gene in cases['candidate_ids']] and real.shape == (len(cases['cases']), len(candidate_ids))
        values, pools, masks = readout(metrics, real, cases, candidate_ids, seen)
        own = json.loads((inference / ('test-%s.json' % view)).read_text())['metrics']
        full = 'secondary_full_go/all' if 'candidate_views' in cases else 'bank_full/all'
        for score in ('raw', 'calibrated'):
            assert values[full][score]['answer_micro'] == own[score]['answer_micro'], 'readout disagrees with test_eval on the full pool (%s, %s)' % (view, score)
        record['views'][view] = {'scores_sha256': sha(archive), 'shape': list(real.shape), 'readouts': values}
        record['pools'] = {name: {'size': len(genes), 'genes_sha256': digest(sorted(int(gene) for gene in genes))} for name, genes in pools.items()}
        record['masks'] = {name: {'associations': sum(len(labels) for labels in answers), 'labels_sha256': digest(answers)} for name, answers in masks.items()}
    assert record['views'], 'no saved TEST matrix found'
    keys = sorted(next(iter(record['views'].values()))['readouts'])
    record['separately_selected'] = {key: {'raw_selected_raw': record['views'].get('raw', {}).get('readouts', {}).get(key, {}).get('raw', {}).get('answer_micro'),
                                           'calibrated_selected_calibrated': record['views'].get('calibrated', {}).get('readouts', {}).get(key, {}).get('calibrated', {}).get('answer_micro')}
                                     for key in keys}
    record['replay_note'] = 'pools and masks are identified by hash so F, S, FS and baseline replays can be summarized on identical columns and labels'
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'views': sorted(record['views']), 'readouts': keys}))


if __name__ == '__main__':
    main()
