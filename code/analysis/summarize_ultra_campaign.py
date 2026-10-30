#!/usr/bin/env python3
"""Independently verify and describe frozen ULTRA experiments (CPU analysis).

This script never selects configurations or edits the manuscript. It requires
all ten completed score bundles before loading any outcome matrix.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def pairs_from(path):
    with Path(path).open() as handle:
        reader=csv.DictReader(handle,delimiter='\t')
        if not {'Gene','Disease'}<=set(reader.fieldnames or []):raise ValueError(f'Missing pair columns: {path}')
        pairs=[(r['Gene'],r['Disease']) for r in reader]
    if not pairs or len(pairs)!=len(set(pairs)):raise ValueError('Empty or duplicate association rows')
    return pairs


def calibrated(scores):
    """One reference per unique disease; leave current disease out, centered moments."""
    scores=np.asarray(scores,dtype=np.float64)
    if len(scores)<2:raise ValueError('Calibration requires at least two disease queries')
    centered=scores-scores[:1];n=len(scores)-1
    mean=(centered.sum(axis=0)-centered)/n
    variance=(np.square(centered).sum(axis=0)-np.square(centered))/n-np.square(mean)
    return (centered-mean)/(np.sqrt(np.maximum(variance,0))+1e-12)


def rank_pairs(scores,diseases,genes,pairs):
    di={d:i for i,d in enumerate(diseases)};gi={g:i for i,g in enumerate(genes)}
    ranks=[]
    for g,d in pairs:
        values=scores[di[d]];truth=values[gi[g]]
        ranks.append(1+np.count_nonzero(values>truth)+(np.count_nonzero(values==truth)-1)/2)
    return np.asarray(ranks,dtype=float)


def metrics(ranks):
    ranks=np.asarray(ranks,dtype=float)
    if not len(ranks) or not np.isfinite(ranks).all() or (ranks<1).any():raise ValueError('Invalid ranks')
    return {'hits@1':float((ranks<=1).mean()),'hits@10':float((ranks<=10).mean()),
            'mrr':float((1/ranks).mean()),'mr':float(ranks.mean())}


def verify_checksums(directory):
    manifest=read(directory/'checksums.json')
    required={'scores.npz','pairs.tsv','candidates.txt','metrics.json'}
    if not required<=set(manifest):raise ValueError(f'Incomplete checksum manifest: {directory}')
    for name,expected in manifest.items():
        if Path(name).name!=name:raise ValueError('Nonlocal manifest path')
        if sha(directory/name)!=expected:raise ValueError(f'Checksum mismatch: {directory/name}')
    return {'manifest_sha256':sha(directory/'checksums.json'),'files':manifest}


def derangement(n,seed):
    if n<2:raise ValueError('Wrong-query control needs at least two diseases')
    rng=np.random.default_rng(seed)
    for _ in range(10000):
        p=rng.permutation(n)
        if np.all(p!=np.arange(n)):return p
    raise RuntimeError('Could not generate a derangement')


def load_scores(directory,seed=20261023):
    directory=Path(directory);provenance=verify_checksums(directory)
    with np.load(directory/'scores.npz',allow_pickle=False) as data:
        scores=np.asarray(data['scores'],dtype=np.float64);diseases=data['diseases'].astype(str).tolist();genes=data['genes'].astype(str).tolist()
    if len(set(diseases))!=len(diseases) or len(set(genes))!=len(genes):raise ValueError('Duplicate disease/candidate IDs')
    if scores.shape!=(len(diseases),len(genes)) or not np.isfinite(scores).all():raise ValueError('Invalid score matrix')
    if genes!=(directory/'candidates.txt').read_text().splitlines():raise ValueError('Candidate ordering mismatch')
    pairs=pairs_from(directory/'pairs.tsv')
    if set(d for _,d in pairs)!=set(diseases) or not {g for g,_ in pairs}<=set(genes):raise ValueError('Pair/matrix identities disagree')
    settings={'raw':rank_pairs(scores,diseases,genes,pairs),'calibrated':rank_pairs(calibrated(scores),diseases,genes,pairs)}
    saved=read(directory/'metrics.json')
    for arm,ranks in settings.items():
        reported=saved[arm][0] if isinstance(saved[arm],list) else saved[arm]
        for key,value in metrics(ranks).items():
            if not np.isclose(value,reported[key],atol=1e-12,rtol=1e-12):raise ValueError(f'Recomputed metric differs: {directory} {arm} {key}')
    permutation=derangement(len(diseases),seed)
    # This is an evaluated-query-panel, label-free estimate of candidate prior.
    mean_scores=np.broadcast_to(scores.mean(axis=0),scores.shape)
    controls={'wrong_query':rank_pairs(scores[permutation],diseases,genes,pairs),
              'candidate_mean':rank_pairs(mean_scores,diseases,genes,pairs)}
    return {'genes':genes,'diseases':diseases,'pairs':pairs,'settings':settings,'controls':controls,
            'control_seed':seed,'permutation_sha256':hashlib.sha256(permutation.tobytes()).hexdigest(),'provenance':provenance}


def cluster_interval(differences,diseases,repeats=10000,seed=20261023,confidence=.9875):
    """Resample diseases; retain all association rows within every selected disease."""
    differences=np.asarray(differences,dtype=float)
    if len(differences)!=len(diseases) or not len(differences) or not np.isfinite(differences).all():raise ValueError('Invalid paired differences')
    if repeats<1:raise ValueError('Positive bootstrap replicate count required')
    unique,inverse=np.unique(diseases,return_inverse=True)
    sums=np.bincount(inverse,weights=differences);counts=np.bincount(inverse)
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,repeats,100):
        ix=rng.integers(0,len(unique),size=(min(100,repeats-start),len(unique)))
        draws.extend((sums[ix].sum(axis=1)/counts[ix].sum(axis=1)).tolist())
    alpha=1-confidence
    return {'difference':float(differences.mean()),'interval':np.quantile(draws,[alpha/2,1-alpha/2]).tolist(),
            'confidence_level':confidence,'bootstrap_replicates':repeats,'seed':seed,'disease_clusters':len(unique),
            'estimator':'association-weighted difference; paired disease-cluster resampling'}


def align(left,right):
    if set(left['pairs'])!=set(right['pairs']):raise ValueError('Comparison associations differ')
    lookup={key:i for i,key in enumerate(right['pairs'])}
    return np.array([lookup[k] for k in left['pairs']])


def combined(folds,setting):
    pairs=[];ranks=[];diseases=set();genes=None
    for fold in folds:
        if diseases&set(fold['diseases']):raise ValueError('Disease appears in multiple outer folds')
        diseases.update(fold['diseases'])
        if genes is not None and genes!=fold['genes']:raise ValueError('Candidate order differs across folds')
        genes=fold['genes'];pairs.extend(fold['pairs']);ranks.extend(fold['settings'][setting])
    return {'pairs':pairs,'ranks':np.asarray(ranks),'genes':genes}


def describe(folds,setting):
    pooled=combined(folds,setting);per_fold=[metrics(f['settings'][setting]) for f in folds]
    disease_values={}
    for (g,d),rank in zip(pooled['pairs'],pooled['ranks']):disease_values.setdefault(d,[]).append(rank)
    per_disease=[metrics(v) for v in disease_values.values()]
    return {'pairs':len(pooled['pairs']),'diseases':len(disease_values),'candidates':len(pooled['genes']),
            'pooled_associations':metrics(pooled['ranks']),
            'equal_fold':{k:float(np.mean([m[k] for m in per_fold])) for k in per_fold[0]},
            'equal_disease':{k:float(np.mean([m[k] for m in per_disease])) for k in per_fold[0]},
            'fold_metrics':per_fold}


ARTIFACTS=('zero','pool','link','profile')


def check_ready(root):
    """Only metadata/hashes are opened here, never outer metrics or scores."""
    root=Path(root);selections=[];digests=[]
    for stage in (1,2,3):
        path=root/f'selection-stage{stage}.json'
        if not path.is_file():raise ValueError('All three selection freezes required before outcome inspection')
        obj=read(path)
        if obj.get('stage')!=stage or obj.get('selection_data')!='inner-validation-only' or obj.get('scoring') not in ('raw','calibrated'):
            raise ValueError('Invalid selection freeze')
        selections.append(obj);digests.append(sha(path))
    primary={name:s['scoring'] for name,s in zip(ARTIFACTS[1:],selections)}
    state=root/'campaign-state.json'
    if state.exists():
        hashes=read(state).get('selection_hashes',{})
        if any(hashes.get(str(i),d)!=d for i,d in enumerate(digests,1)):raise ValueError('Dispatcher selection checksum mismatch')
    bundles=[]
    for fold in range(10):
        path=root/'runs'/f'score-fold{fold}'/'completion.json'
        if not path.is_file():raise ValueError(f'All ten score completions required; missing fold {fold}')
        obj=read(path);prep=read(root/'data'/f'fold{fold}'/'preparation.json')
        if obj.get('phase')!='score' or obj.get('fold')!=fold or obj.get('identity')!=prep or prep.get('fold')!=fold:
            raise ValueError('Score completion identity mismatch')
        if obj.get('technical_pilot',False):raise ValueError('Technical pilot cannot enter final assessment')
        cfg=path.parent/'config.json'
        if cfg.exists() and read(cfg).get('technical_pilot',False):raise ValueError('Technical pilot cannot enter final assessment')
        if obj.get('selection_sha256')!=digests or obj.get('primary_scoring')!=primary:raise ValueError('Score completion differs from frozen selection')
        if sorted(obj.get('artifacts',[]))!=sorted(ARTIFACTS):raise ValueError('Exactly four score artifacts required')
        hashes=obj.get('checkpoint_sha256',{})
        if set(hashes)!=set(ARTIFACTS) or any(len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in hashes.values()):raise ValueError('Invalid checkpoint hashes')
        if hashes['zero']!=hashes['pool']:raise ValueError('Frozen pooling must use zero-shot checkpoint')
        for name in ARTIFACTS:
            manifest=path.parent/name/'checksums.json'
            if obj.get('artifact_checksums_sha256',{}).get(name)!=sha(manifest):raise ValueError('Artifact manifest not bound to score completion')
            declared=read(manifest)
            if declared.get('candidates.txt')!=prep.get('candidates_sha256') or declared.get('pairs.tsv')!=prep.get('test_sha256'):
                raise ValueError('Scored candidates/test pairs differ from prepared inputs')
        bundles.append({'fold':fold,'completion_sha256':sha(path),'checkpoint_sha256':hashes})
    return selections,digests,bundles


def resolve_old(path,override):
    path=Path(path)
    if path.exists():return path
    if override is not None:
        candidate=Path(override)/path.name
        if candidate.exists():return candidate
    raise FileNotFoundError(path)


def load_indigena(directory,predictions,reference):
    """Recompute both historical score modes, retaining only a unique disease panel."""
    folds=[];sources=[]
    for fold in range(10):
        path=Path(directory)/f'fold_{fold}.json';obj=read(path);prov=obj['provenance']
        paths={k:resolve_old(prov[k]['path'],predictions) for k in ('predictions','candidates')}
        for k,p in paths.items():
            if sha(p)!=prov[k]['sha256']:raise ValueError(f'INDIGENA source hash mismatch: {p}')
        genes=paths['candidates'].read_text().splitlines()
        if genes!=reference[fold]['genes']:raise ValueError('INDIGENA candidate order differs')
        pairs=[];by_disease={}
        with paths['predictions'].open() as handle:
            for line in handle:
                cells=line.rstrip('\n').split('\t')
                if len(cells)!=len(genes)+3:raise ValueError('Historical prediction width mismatch')
                g,d=cells[:2];ix=int(cells[2]);values=np.asarray(cells[3:],float)
                if not 0<=ix<len(genes) or genes[ix]!=g or not np.isfinite(values).all():raise ValueError('Invalid historical prediction row')
                if d in by_disease and not np.array_equal(values,by_disease[d]):raise ValueError('Inconsistent repeated disease scores')
                by_disease[d]=values;pairs.append((g,d))
        if len(set(pairs))!=len(pairs):raise ValueError('Duplicate historical pairs')
        stored=[(v['gene'],v['disease']) for v in obj['keys']]
        if pairs!=stored or set(pairs)!=set(reference[fold]['pairs']):raise ValueError('Historical cohort differs')
        diseases=sorted(by_disease);scores=np.stack([by_disease[d] for d in diseases]);settings={}
        for mode,values in [('raw',scores),('calibrated',calibrated(scores))]:
            ranks=rank_pairs(values,diseases,genes,pairs);key='disease_loo' if mode=='calibrated' else mode
            if not np.array_equal(ranks,obj['settings'][key]['ranks']):raise ValueError('Historical ranks do not reproduce')
            settings[mode]=ranks
        folds.append({'genes':genes,'diseases':diseases,'pairs':pairs,'settings':settings})
        sources.append({'metrics_sha256':sha(path),'metrics_path':str(path),'verified_sources':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()}})
    return folds,sources


def summarize(root,indigena_rawsel=None,indigena_calsel=None,rawsel_predictions=None,calsel_predictions=None,repeats=10000,seed=20261023):
    root=Path(root);selections,digests,bundles=check_ready(root)
    all_folds={};summary={};controls={};provenance={};reference=None
    for name in ARTIFACTS:
        folds=[load_scores(root/'runs'/f'score-fold{f}'/name,seed+f) for f in range(10)]
        if reference is not None:
            for left,right in zip(reference,folds):
                if left['genes']!=right['genes'] or left['diseases']!=right['diseases'] or set(left['pairs'])!=set(right['pairs']):raise ValueError('Artifact cohorts differ')
        else:reference=folds
        all_folds[name]=folds;summary[name]={mode:describe(folds,mode) for mode in ('raw','calibrated')}
        controls[name]={control:{'pooled':metrics(np.concatenate([x['controls'][control] for x in folds])),
            'fold_metrics':[metrics(x['controls'][control]) for x in folds]} for control in ('wrong_query','candidate_mean')}
        controls[name]['permutations']=[{'fold':f,'seed':x['control_seed'],'sha256':x['permutation_sha256']} for f,x in enumerate(folds)]
        controls[name]['training_status']=[]
        for f,x in enumerate(folds):
            train=root/'data'/f'fold{f}'/'training.tsv';prep=read(train.parent/'preparation.json')
            if sha(train)!=prep['training_sha256']:raise ValueError('Training identity mismatch')
            seen={g for g,_ in pairs_from(train)};mask=np.array([g in seen for g,_ in x['pairs']])
            controls[name]['training_status'].append({'fold':f,'training_sha256':sha(train),
                'candidate_seen':sum(g in seen for g in x['genes']),'candidate_unseen':sum(g not in seen for g in x['genes']),
                'groups':{label:{'pairs':int(use.sum()),'raw_hits@10':float((x['settings']['raw'][use]<=10).mean()) if use.any() else None} for label,use in [('seen',mask),('unseen',~mask)]}})
        provenance[name]=[x['provenance'] for x in folds]
    mode=selections[2]['scoring'];left=combined(all_folds['profile'],mode)
    comparators={'zero/raw':combined(all_folds['zero'],'raw')};historical={}
    for label,directory,predictions in [('INDIGENA_rawsel',indigena_rawsel,rawsel_predictions),('INDIGENA_calsel',indigena_calsel,calsel_predictions)]:
        if directory is None:historical[label]={'status':'missing','reason':'Location not supplied'};continue
        try:folds,sources=load_indigena(directory,predictions,reference)
        except FileNotFoundError as exc:
            historical[label]={'status':'missing','reason':str(exc)};continue
        historical[label]={'status':'verified','sources':sources}
        modes=('raw',) if label=='INDIGENA_rawsel' else ('raw','calibrated')
        summary[label]={m:describe(folds,m) for m in modes}
        if label=='INDIGENA_rawsel':comparators[label+'/raw']=combined(folds,'raw')
    differences=[]
    for name,right in comparators.items():
        ranks=right['ranks'][align(left,right)]
        for endpoint,k in [('hits@1',1),('hits@10',10)]:
            delta=(left['ranks']<=k).astype(float)-(ranks<=k).astype(float)
            differences.append({'left':'profile/'+mode,'right':name,'endpoint':endpoint,
                **cluster_interval(delta,[d for _,d in left['pairs']],repeats,seed,.9875)})
    return {'schema_version':1,'selections':selections,'selection_sha256':digests,'verified_bundles':bundles,
        'primary_scoring':{'zero':'raw',**{n:s['scoring'] for n,s in zip(ARTIFACTS[1:],selections)}},
        'summary':summary,'primary_differences':differences,'historical':historical,'controls':controls,'verified_artifacts':provenance,
        'independent_baselines':{'status':'not run; audit only'},
        'inference':{'complete_primary_family':len(differences)==4,'per_interval_confidence':.9875,'nominal_familywise_confidence':.95,
            'family':'Stage3 selected scoring vs matched zero/raw and historical INDIGENA raw-MR-selection/raw: H1 and H10',
            'bootstrap_method':'paired percentile disease-cluster; association-weighted point estimate',
            'units':'Proportions; multiply by 100 for percentage points',
            'limitations':['Exploratory follow-up: previous outcomes informed development.',
                'Bootstrap conditions on fitted models; training and selection uncertainty are not captured.',
                'Historical INDIGENA uses its historical validation and graph protocol; it is not newly matched retraining.',
                'Raw/calibrated stage sensitivities and INDIGENA calsel are descriptive, not extra primary hypotheses.',
                'Wrong-query control uses one fixed seeded derangement per fold; no permutation significance test.',
                'Candidate marginal averages raw scores over the evaluated query panel without gene labels.',
                'Nominal simultaneous coverage uses Bonferroni-adjusted approximate percentile intervals.']}}


def write_report(report,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'metrics.tsv').open('w') as handle:
        writer=csv.writer(handle,delimiter='\t');writer.writerow(['artifact','scoring','weighting','H1','H10','MRR','MR','pairs','diseases'])
        for artifact,settings in report['summary'].items():
            for scoring,values in settings.items():
                for weighting in ('pooled_associations','equal_fold','equal_disease'):
                    m=values[weighting];writer.writerow([artifact,scoring,weighting,m['hits@1'],m['hits@10'],m['mrr'],m['mr'],values['pairs'],values['diseases']])
    with (out/'comparisons.tsv').open('w') as handle:
        writer=csv.writer(handle,delimiter='\t');writer.writerow(['left','right','endpoint','difference','lower','upper','confidence'])
        for d in report['primary_differences']:writer.writerow([d['left'],d['right'],d['endpoint'],d['difference'],*d['interval'],d['confidence_level']])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path)
    parser.add_argument('--indigena-rawsel-metrics',type=Path);parser.add_argument('--indigena-calsel-metrics',type=Path)
    parser.add_argument('--indigena-rawsel-predictions-dir',type=Path);parser.add_argument('--indigena-calsel-predictions-dir',type=Path)
    parser.add_argument('--bootstrap-replicates',type=int,default=10000);parser.add_argument('--seed',type=int,default=20261023)
    args=parser.parse_args()
    report=summarize(args.root,args.indigena_rawsel_metrics,args.indigena_calsel_metrics,
        args.indigena_rawsel_predictions_dir,args.indigena_calsel_predictions_dir,args.bootstrap_replicates,args.seed)
    output=args.out or args.root/'report';write_report(report,output)
    print(json.dumps({'status':'complete' if report['inference']['complete_primary_family'] else 'incomplete-primary-comparators','report':str(output)}))
    if not report['inference']['complete_primary_family']:raise SystemExit(2)

if __name__=='__main__':main()
