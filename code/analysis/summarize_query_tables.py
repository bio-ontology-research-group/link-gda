#!/usr/bin/env python3
"""Verify retained query scores/ranks and derive every historical query-table cell."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phenoquery.common import ARMS, cal, pairs, ranks, read, sha, write

MATCHED_ARMS = [f'{graph}_{readout}_{stage}' for graph in ['original','augmented']
                for readout in ['pool','and'] for stage in ['frozen','adapted']]
DIAGNOSTIC_ARMS = ['original_pool_frozen','original_pool_adapted',
                   'augmented_pool_adapted','augmented_and_adapted']


def metrics(values):
    values = np.asarray(values, dtype=float)
    if not len(values): return None
    if not np.isfinite(values).all() or (values < 1).any(): raise ValueError('Invalid ranks')
    return {'mr':float(values.mean()), 'mrr':float((1/values).mean()),
            **{f'hits@{k}':float((values <= k).mean()) for k in [1,10,100]}}


def metadata(data):
    genes = (data/'candidates.txt').read_text().splitlines()
    train, validation, test = [pairs(data/(split+'.tsv')) for split in ['training','validation','test']]
    sets = [{d for _,d in rows} for rows in [train, validation, test]]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise ValueError('Disease leakage')
    if len(genes) != len(set(genes)) or len(test) != len(set(test)): raise ValueError('Repeated identities')
    if not {g for rows in [train,validation,test] for g,_ in rows} <= set(genes):
        raise ValueError('Positive gene outside candidates')
    profiles = {d:frozenset(ps) for d,ps in read(data/'profiles.json').items()}
    if any(not profiles[d] for ds in sets for d in ds): raise ValueError('Empty phenotype profile')
    seen_genes = frozenset(g for g,_ in train)
    candidate_seen = np.array([g in seen_genes for g in genes])
    seen = np.array([g in seen_genes for g,_ in test])
    training_by_gene = defaultdict(set)
    training_profiles = set()
    for g,d in train:
        training_by_gene[g].add(profiles[d]); training_profiles.add(profiles[d])
    positives = defaultdict(set)
    for g,d in test: positives[d].add(g)
    jaccard = np.array([max((len(profiles[d]&q)/len(profiles[d]|q)
                             for q in training_by_gene.get(g,())), default=0) for g,d in test])
    strata = {'seen_high': seen & (jaccard>=.5),
               'seen_medium': seen & (jaccard>=.2) & (jaccard<.5),
               'seen_low': seen & (jaccard>0) & (jaccard<.2),
               'seen_zero': seen & (jaccard==0), 'unseen': ~seen,
               'multi_gene': np.array([len(positives[d])>1 for _,d in test]),
               'training_profile_match': np.array([profiles[d] in training_profiles for _,d in test])}
    return genes, test, seen, candidate_seen, strata


def diagnostics(scores, di, gi, candidate_seen):
    full = rankdata(-scores, method='average', axis=1)
    other = (full.sum(0)[gi]-full[di,gi])/(len(scores)-1)
    within = []
    for row,column in zip(di,gi):
        values = scores[row,candidate_seen==candidate_seen[column]]
        truth = scores[row,column]
        within.append(1+np.count_nonzero(values>truth)+.5*(np.count_nonzero(values==truth)-1))
    return np.array(within), other


def summarize(base, matched, eq, folds):
    collected = defaultdict(list)
    checksums = {}
    fold_metadata = []
    for fold in folds:
        data = base/'data'/f'fold{fold}'
        genes, rows, seen, candidate_seen, strata = metadata(data)
        diseases = sorted({d for _,d in rows})
        glookup, dlookup = {g:i for i,g in enumerate(genes)}, {d:i for i,d in enumerate(diseases)}
        gi, di = np.array([glookup[g] for g,_ in rows]), np.array([dlookup[d] for _,d in rows])
        fold_metadata.append({'fold':fold, 'pairs':len(rows), 'diseases':len(diseases),
            'candidates':len(genes), 'seen_pairs':int(seen.sum()), 'unseen_pairs':int((~seen).sum()),
            'seen_candidates':int(candidate_seen.sum()), 'unseen_candidates':int((~candidate_seen).sum()),
            'strata':{k:int(v.sum()) for k,v in strata.items()}})
        for filename in ['training.tsv','validation.tsv','test.tsv','profiles.json','candidates.txt']:
            checksums[f'benchmark/data/fold{fold}/{filename}'] = sha(data/filename)
        bundles = [(co,arm,matched/co/f'score-fold{fold}'/arm)
                   for co in ['general','query'] for arm in MATCHED_ARMS]
        bundles += [('phenoquery',arm,eq/'test'/f'fold{fold}'/arm) for arm in ARMS]
        for cohort,arm,directory in bundles:
            complete = read(directory/'COMPLETE.json')
            prefix = (f'eq/test/fold{fold}/{arm}' if cohort=='phenoquery'
                      else f'matched/{cohort}/score-fold{fold}/{arm}')
            for name in ['scores.npz','ranks.npz']:
                digest = sha(directory/name)
                if digest != complete['files'][name]: raise ValueError(f'Checksum mismatch: {prefix}/{name}')
                checksums[prefix+'/'+name] = digest
            checksums[prefix+'/COMPLETE.json'] = sha(directory/'COMPLETE.json')
            with np.load(directory/'scores.npz') as z:
                np.testing.assert_array_equal(z['genes'].astype(str), genes)
                np.testing.assert_array_equal(z['diseases'].astype(str), diseases)
                values = z['scores'].copy()
            with np.load(directory/'ranks.npz') as z:
                np.testing.assert_array_equal(z['genes'].astype(str), [g for g,_ in rows])
                np.testing.assert_array_equal(z['diseases'].astype(str), [d for _,d in rows])
                np.testing.assert_array_equal(z['seen'], seen)
                saved = {mode:z[mode].copy() for mode in ['raw','calibrated']}
                if cohort != 'phenoquery':
                    np.testing.assert_array_equal(z['primary'], saved[complete['primary_mode']])
            mode_outputs = {}
            for mode,scores in [('raw',values), ('calibrated',cal(values))]:
                regenerated = ranks(scores, diseases, genes, rows)
                np.testing.assert_array_equal(regenerated, saved[mode])
                record = {'ranks':regenerated, 'seen':seen, 'strata':strata,
                          'metrics':metrics(regenerated)}
                if cohort!='phenoquery' and arm in DIAGNOSTIC_ARMS:
                    within, other = diagnostics(scores, di, gi, candidate_seen)
                    record['diagnostics'] = {'within_seen_mr':float(within[seen].mean()),
                        'within_unseen_mr':float(within[~seen].mean()),
                        'actual_mr':float(regenerated.mean()), 'other_query_mr':float(other.mean())}
                mode_outputs[mode] = record
            collected[cohort+'/'+arm].append({'fold':fold, 'modes':mode_outputs,
                'primary_mode':'calibrated' if cohort=='phenoquery' else complete['primary_mode']})
        print('VERIFIED_QUERY_FOLD', fold, len(bundles), flush=True)
    result = {}
    for key, records in collected.items():
        modes = {}
        for mode in ['raw','calibrated']:
            mm = [r['modes'][mode] for r in records]
            all_ranks = np.concatenate([r['ranks'] for r in mm])
            all_seen = np.concatenate([r['seen'] for r in mm])
            output = {'pooled':metrics(all_ranks), 'median':float(np.median(all_ranks)),
                'equal_fold':{k:float(np.mean([r['metrics'][k] for r in mm])) for k in mm[0]['metrics']},
                'fold_mr_sd':float(np.std([r['metrics']['mr'] for r in mm],ddof=1)),
                'seen':metrics(all_ranks[all_seen]), 'unseen':metrics(all_ranks[~all_seen]),
                'strata':{name:metrics(np.concatenate([r['ranks'][r['strata'][name]] for r in mm]))
                           for name in mm[0]['strata']}}
            if 'diagnostics' in mm[0]:
                output['diagnostics_equal_fold'] = {k:float(np.mean([r['diagnostics'][k] for r in mm]))
                                                     for k in mm[0]['diagnostics']}
            modes[mode] = output
        primary = [r['primary_mode'] for r in records]
        if len(set(primary)) != 1: raise ValueError('Table requires explicit per-fold mixed scoring modes')
        result[key] = {'primary_mode':primary[0], 'modes':modes}
    counts = {k:sum(m[k] for m in fold_metadata) for k in ['pairs','diseases','seen_pairs','unseen_pairs']}
    counts['strata'] = {k:sum(m['strata'][k] for m in fold_metadata) for k in fold_metadata[0]['strata']}
    return {'schema':1, 'scope':'Historical seed0 campaigns; PhenoQuery initializes from ultra_4g',
        'source_sha256':{Path(__file__).name:sha(__file__), 'phenoquery/common.py':sha(Path(__file__).resolve().parents[1]/'phenoquery/common.py')},
        'input_sha256':checksums, 'folds':fold_metadata, 'counts':counts, 'results':result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--matched',type=Path,required=True)
    p.add_argument('--eq',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--folds',type=int,nargs='+',default=list(range(10)))
    a = p.parse_args()
    if len(a.folds)<2 or len(set(a.folds))!=len(a.folds):p.error('At least two distinct folds required')
    if a.output.exists():p.error('Refusing to overwrite existing report')
    write(a.output, summarize(a.base,a.matched,a.eq,a.folds))


if __name__ == '__main__': main()
