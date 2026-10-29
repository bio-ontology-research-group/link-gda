"""Verify the complete matched KGE campaign and expose its 40 validation choices."""
import argparse
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from phenoquery.common import read, sha, write
from training.run_matched_kge import GRAPHS, MODELS, PRIMARY, directory, validate_completion

CHOICES = ('raw_mr', 'raw_head', 'calibrated_mr', 'calibrated_head')
INPUTS = ('raw/train.txt', 'candidates.txt', 'profiles.json', 'training.tsv', 'validation.tsv', 'test.tsv')
PROTOCOL = Path(__file__).resolve().parent / 'protocols/matched-kge-20261029.md'


def verify_selections(folder, complete, identity):
    """Recompute strict validation choices, retaining the first epoch on ties."""
    settings = identity['settings']
    profile = settings['model'] == 'PPOOL'
    epochs = settings['profile_epochs'] if profile else settings['epochs']
    frequency = 1 if profile else settings['validation_frequency']
    state = complete['training']
    if (state['epochs'] != epochs or state.get('early_stopped', False)
            or state['complete'] is not True or (profile and len(state['history']) != epochs)):
        raise ValueError('Training did not complete its fixed epoch budget')
    curve = read(folder / 'validation.json')
    if (curve['frequency'] != frequency
            or [item['epoch'] for item in curve['curve']] != list(range(frequency, epochs + 1, frequency))
            or epochs % frequency):
        raise ValueError('Validation trajectory is incomplete')
    best = {}
    for item in curve['curve']:
        for mode in ('raw', 'calibrated'):
            values = item['metrics'][mode]
            if (not all(math.isfinite(values[k]) for k in ('hits@1', 'hits@10', 'mrr', 'mr'))
                    or not 0 <= values['hits@1'] <= values['hits@10'] <= 1
                    or not 0 < values['mrr'] <= 1 or values['mr'] < 1):
                raise ValueError('Invalid validation metrics')
            for criterion in ('mr', 'head'):
                choice = f'{mode}_{criterion}'
                key = [-values['mr']] if criterion == 'mr' else [values['hits@10'], values['hits@1'], values['mrr'], -values['mr']]
                if choice not in best or key > best[choice]['key']:
                    best[choice] = dict(epoch=item['epoch'], key=key, metrics=values,
                                        checkpoint=f'selected_{choice}.pt')
    for records in (complete['selected'], curve['best']):
        normalized = {k: {**v, 'checkpoint': Path(v['checkpoint']).name} for k, v in records.items()}
        if normalized != best:
            raise ValueError('Selected checkpoint differs from the fixed validation rule')
    if set(complete['files']) != {'validation.json', *(v['checkpoint'] for v in best.values())}:
        raise ValueError('Training completion does not bind all selected artifacts')
    return best


def build(root, data_root, output, gene_phenotypes, expected_candidates=13625):
    root, data_root, output = Path(root).resolve(), Path(data_root).resolve(), Path(output).resolve()
    relative = lambda p: os.path.relpath(p, output.parent)
    frozen = read(root / 'FROZEN.json')
    expected_runs = {f'fold{fold}/{graph}/{model}' for fold in range(10) for graph in GRAPHS for model in MODELS}
    if (frozen['folds'] != list(range(10)) or frozen['models'] != list(MODELS)
            or frozen['graphs'] != list(GRAPHS) or set(frozen['training']) != expected_runs):
        raise ValueError('Require the global freeze for all 100 training runs')
    freeze_sha = sha(root / 'FROZEN.json')
    protocol_sha = sha(PROTOCOL)
    annotation_sha = sha(gene_phenotypes)
    models, identities, evidence, benchmarks, common_settings = {}, {}, {}, {}, {}
    common_source = None
    for fold in range(10):
        data = data_root / f'fold{fold}'
        inputs = {name: sha(data / name) for name in INPUTS}
        genes = (data / 'candidates.txt').read_text().splitlines()
        if len(genes) != expected_candidates or len(set(genes)) != len(genes):
            raise ValueError('Require the complete expected gene candidate pool')
        benchmarks[str(fold)] = inputs
        for graph in GRAPHS:
            for model in MODELS:
                folder = directory(root, fold, graph, model)
                run = str(folder.relative_to(root))
                if sha(folder / 'TRAINED.json') != frozen['training'][run]:
                    raise ValueError('Training completion changed after global freeze')
                complete = validate_completion(folder)
                identity = read(folder / 'identity.json')
                settings = identity['settings']
                expected = dict(model=model, dimension=400 if model == 'INDIGENA' else 200,
                    seed=0, epochs=1000, validation_frequency=20, batch_size=65536,
                    sub_batch_size=8192, profile_epochs=20, checkpoint_frequency=20,
                    expected_candidates=expected_candidates, primary=PRIMARY[model], early_stopping=False,
                    t1_sampling='candidate-typed, known training positives excluded')
                if any(settings.get(k) != v for k, v in expected.items()):
                    raise ValueError('Training settings differ from the production protocol')
                if (identity['fold'] != fold or identity['graph'] != graph or identity['inputs'] != inputs
                        or identity['protocol_sha256'] != protocol_sha or complete['primary'] != PRIMARY[model]):
                    raise ValueError('Training/evaluation identity mismatch')
                if model == 'INDIGENA' and identity['gene_phenotypes_sha256'] != annotation_sha:
                    raise ValueError('Frozen gene phenotype annotations changed')
                source = {k: identity[k] for k in ('source', 'torch', 'pykeen', 'protocol_sha256')}
                if common_source is not None and source != common_source:
                    raise ValueError('Training source/environment differs across runs')
                common_source = source
                if model in common_settings and settings != common_settings[model]:
                    raise ValueError('Model settings differ across folds or graphs')
                common_settings[model] = settings
                selected = verify_selections(folder, complete, identity)
                scored = read(folder / 'SCORED.json')
                if (scored['freeze_sha256'] != freeze_sha or scored['training_sha256'] != frozen['training'][run]
                        or scored['primary'] != PRIMARY[model]
                        or set(scored['files']) != {f'scores-{choice}.npz' for choice in CHOICES}):
                    raise ValueError('Scored outputs do not match the frozen choices')
                for choice in CHOICES:
                    file = folder / f'scores-{choice}.npz'
                    digest = scored['files'][file.name]
                    if sha(file) != digest:
                        raise ValueError('Completed score file changed')
                    label = f'MatchedKGE ({model}, {graph}, {choice})'
                    record = models.setdefault(label, dict(name=label, family=model, graph=graph,
                        validation_choice=choice, primary=choice == PRIMARY[model], folds={}))
                    record['folds'][str(fold)] = dict(path=relative(file), sha256=digest)
                identities[(fold, graph, model)] = identity
                evidence[run] = dict(training_sha256=frozen['training'][run],
                    identity_sha256=complete['identity_sha256'], scored_sha256=sha(folder / 'SCORED.json'),
                    validation_sha256=complete['files']['validation.json'], selected=selected,
                    nodes=identity['nodes'], forward_triples=identity['forward_triples'],
                    epochs=complete['training']['epochs'])
        for model in MODELS:
            left, right = (identities[(fold, g, model)] for g in GRAPHS)
            if left.get('gene_phenotypes_sha256') != right.get('gene_phenotypes_sha256'):
                raise ValueError('Annotation resources differ across graphs')
            for graph in GRAPHS:
                if model in ('CONVKB', 'PPOOL'):
                    parent = evidence[f'fold{fold}/{graph}/U1']
                    choice = 'calibrated_mr' if model == 'CONVKB' else 'raw_head'
                    checkpoint = directory(root, fold, graph, 'U1') / parent['selected'][choice]['checkpoint']
                    if identities[(fold, graph, model)]['warm_start_sha256'] != sha(checkpoint):
                        raise ValueError('Warm start differs from the prescribed U1 choice')
    comparisons = {model: dict(choice=PRIMARY[model],
        scoring='calibrated' if PRIMARY[model].startswith('calibrated') else 'raw',
        **{graph: f'MatchedKGE ({model}, {graph}, {PRIMARY[model]})' for graph in GRAPHS}) for model in MODELS}
    return dict(schema=1, folds=list(range(10)), training_runs=100, score_files=400,
        freeze_sha256=freeze_sha, protocol_sha256=protocol_sha, builder_sha256=sha(__file__),
        gene_phenotypes_sha256=annotation_sha,
        benchmark=benchmarks, source=common_source, settings=common_settings,
        evidence=evidence, primary_comparisons=comparisons, models=list(models.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gene-phenotypes', type=Path, required=True)
    a = p.parse_args()
    result = build(a.root, a.data_root, a.output, a.gene_phenotypes)
    if a.output.exists() and read(a.output) != result:
        raise ValueError('Existing manifest differs; choose a new output path')
    write(a.output, result)
    print('Verified 100 training runs, 400 score files and 40 validation choices')


if __name__ == '__main__':
    main()
