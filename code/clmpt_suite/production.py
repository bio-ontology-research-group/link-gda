"""Production suite plan (Stages 3-6) and its readiness gate. Writes a plan; never stages, releases or submits.

plan       109 pretraining units (Orix ComplEx) and 185 query units (IBEX GNN) under the full-pipeline seed policy:
           Stage 3 PFS method/curriculum suite on folds 0-9 at seed 0 plus seeds 1-9 on fold 0 (19 pairs, five
           conditions sharing each pair's initializer); Stage 4 P/PF/PS and Stage 5 F/S/FS on folds 0-9 at seed 0;
           Stage 6 excluded-cohort F/S/FS on the fixed split with seeds 0-9. Each query unit depends on the COMPLETE
           delivery of its own pretraining unit, never on a Slurm job of it, so there is no all-pretrain barrier and a
           repeated pretraining segment cannot release its child early.
readiness  checks the blockers that must clear before any production campaign is staged and reports each one;
           anything not verifiable from files is reported as unresolved, not assumed.
"""
import argparse
import hashlib
import json
from pathlib import Path


CONDITIONS = (('full13-shared', 'full13', 'shared'), ('core3-shared', 'core3', 'shared'), ('clinical-only-shared', 'aux_off', 'shared'),
              ('full13-independent', 'full13', 'independent'), ('full13-mean', 'full13', 'mean'))
FULL = CONDITIONS[0]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def pretrain_unit(identifier, stage, cohort, modality, fold, seed):
    return {'id': identifier, 'kind': 'pretrain', 'cluster': 'orix', 'stage': stage, 'cohort': cohort, 'modality': modality, 'fold': fold, 'seed': seed,
            'graph_input': {'cohort': cohort, 'modality': modality, 'fold': fold, 'source': 'data worker modality manifest (unresolved until delivered)'},
            'budget': 'fixed optimizer-update budget of the frozen common recipe, identical across modality graphs', 'reuse': None}


def query_unit(identifier, stage, parent, cohort, modality, fold, seed, condition, pools=None, target_subset=None):
    name, curriculum, encoder = condition
    unit = {'id': identifier, 'kind': 'query', 'cluster': 'ibex', 'stage': stage, 'depends_on': parent, 'release_when': 'parent COMPLETE.json delivered and admitted by hash',
            'cohort': cohort, 'modality': modality, 'fold': fold, 'seed': seed, 'condition': name, 'curriculum': curriculum, 'encoder': encoder,
            'steps': 'common recipe budget (20000)', 'selection': 'raw and calibrated checkpoints selected separately on VALIDATION; terminal kept', 'test_evaluation': False}
    if pools:
        unit.update({'ranking_pools': pools, 'pools_are_readouts_of_one_trajectory': True})
    if target_subset:
        unit['target_subset'] = target_subset
    return unit


def plan():
    pretrain, query = {}, []
    for fold, seed in [(fold, 0) for fold in range(10)] + [(0, seed) for seed in range(1, 10)]:
        parent = 'main-PFS-f%d-ps%d' % (fold, seed)
        pretrain[parent] = pretrain_unit(parent, 3, 'main', 'PFS', fold, seed)
        for condition in CONDITIONS:
            query.append(query_unit('main-PFS-f%d-qs%d-%s' % (fold, seed, condition[0]), 3, parent, 'main', 'PFS', fold, seed, condition))
    for stage, modalities in ((4, ('P', 'PF', 'PS')), (5, ('F', 'S', 'FS'))):
        for modality in modalities:
            for fold in range(10):
                parent = 'main-%s-f%d-ps0' % (modality, fold)
                pretrain[parent] = pretrain_unit(parent, stage, 'main', modality, fold, 0)
                query.append(query_unit('main-%s-f%d-qs0-%s' % (modality, fold, FULL[0]), stage, parent, 'main', modality, fold, 0, FULL))
    for modality in ('F', 'S', 'FS'):
        for seed in range(10):
            parent = 'excluded-%s-ps%d' % (modality, seed)
            pretrain[parent] = pretrain_unit(parent, 6, 'excluded', modality, None, seed)
            query.append(query_unit('excluded-%s-qs%d-%s' % (modality, seed, FULL[0]), 6, parent, 'excluded', modality, None, seed, FULL, pools=[5696, 18904],
                                    target_subset='expression-supported true targets; candidate pools unchanged' if modality == 'S' else 'frozen 425 associations'))
    for unit in query:
        if unit['id'] == 'main-PFS-f0-qs0-full13-shared':
            unit['hpo_reuse_candidate'] = 'the common-recipe HPO winner may replace this unit only after identical source, input, initializer, schedule and RNG identities are verified'
    counts = {'pretrain': len(pretrain), 'query': len(query), 'query_transformer': sum(unit['encoder'] != 'mean' for unit in query),
              'query_plain_mean': sum(unit['encoder'] == 'mean' for unit in query), 'by_stage': {}}
    for unit in list(pretrain.values()) + query:
        counts['by_stage'].setdefault(str(unit['stage']), {}).setdefault(unit['kind'], 0)
        counts['by_stage'][str(unit['stage'])][unit['kind']] += 1
    assert counts['pretrain'] == 109 and counts['query'] == 185 and counts['query_plain_mean'] == 19
    return {'schema': 'clmpt-suite-production-plan/1', 'launchable': False, 'seed_scope': 'full-pipeline', 'counts': counts,
            'pretrain': list(pretrain.values()), 'query': query,
            'held': {'test_replay': 'blocked until every production selection receipt is frozen; never automatic during training',
                     'second_full_recipe': 'only by explicit user decision if the raw and calibrated HPO winners differ',
                     'deferred_analyses': ['Robert replay analyses', 'baseline replay', 'fold-0 HPO-validation-excluded sensitivity', 'fixed-TRAIN calibration sensitivity', 'initial-checkpoint TEST row']},
            'rules': {'excluded_initializers': 'trained separately; never reused from main units by modality name',
                      'rq2_fallback': 'deterministic first-eligible phenotype fallback, from the data worker manifest',
                      'excluded_validation': 'excluded 333-pair validation selector with full-GO ranking, distinct from main',
                      'scheduling': 'query unit released by its own parent delivery; no global pretraining barrier'}}


def readiness(hpo_campaign, data_manifest, pretrain_interface, query_protocol):
    blockers, verified = [], {}
    recipe = Path(hpo_campaign) / 'decisions' / 'recipe.json' if hpo_campaign else None
    frozen = Path(hpo_campaign) / 'decisions' / 'recipe.frozen.json' if hpo_campaign else None
    if recipe and frozen.exists() and json.loads(frozen.read_text())['decision_sha256'] == sha(recipe):
        value = json.loads(recipe.read_text())
        verified['common_recipe'] = {'leaderboard': value['common_leaderboard'], 'production_recipe': value['production_recipe'], 'sha256': sha(recipe),
                                     'other_view_recipe_differs': value['other_view_recipe_differs']}
    else:
        blockers.append('common recipe not frozen in the HPO campaign')
    if data_manifest and Path(data_manifest).exists():
        verified['data_manifest'] = {'path': str(data_manifest), 'sha256': sha(data_manifest), 'note': 'presence only; per-unit graph and bank identities are checked when a production campaign is staged'}
    else:
        blockers.append('data worker modality, fold, cohort and pool manifests not delivered')
    if not (pretrain_interface and Path(pretrain_interface).exists()):
        blockers.append('pretraining worker config interface and Orix controller not delivered')
    if query_protocol and Path(query_protocol).exists():
        protocol = json.loads(Path(query_protocol).read_text())
        missing = [name for key, names in (('curricula', ('full13', 'core3', 'aux_off')), ('encoders', ('shared', 'independent', 'mean')))
                   for name in names if name not in protocol.get(key, [])]
        if missing or sorted(protocol.get('seeds', [])) != list(range(10)):
            blockers.append('production protocol does not admit every curriculum, encoder and seeds 0-9: %s' % missing)
        if not protocol.get('production_recipe'):
            blockers.append('production query protocol has no frozen production_recipe (copy decisions/recipe.json production_recipe into a NEW protocol)')
    else:
        blockers.append('production query protocol (curricula, encoders, seeds 0-9, production_recipe) not written')
    blockers += ['qualification of independent branches, plain-mean aggregation, clinical-only denominator 128 and per-stage seed streams with resume (not run)',
                 'excluded 333-pair validation selector and full-GO ranking verification (not run)',
                 'storage and timing qualification for production configurations (HPO qualification covers only Full13/shared/seed 0)',
                 'production stager, advancer DAG and Orix-to-IBEX delivery bridge (not implemented in this package)']
    return {'ready': False, 'blockers': blockers, 'verified': verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    write = commands.add_parser('plan')
    write.add_argument('--output', required=True)
    check = commands.add_parser('readiness')
    check.add_argument('--hpo-campaign')
    check.add_argument('--data-manifest')
    check.add_argument('--pretrain-interface')
    check.add_argument('--query-protocol')
    args = parser.parse_args()
    if args.command == 'plan':
        value = plan()
        output = Path(args.output)
        assert not output.exists()
        output.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        print(json.dumps(value['counts'], indent=2, sort_keys=True))
    else:
        print(json.dumps(readiness(args.hpo_campaign, args.data_manifest, args.pretrain_interface, args.query_protocol), indent=2, sort_keys=True))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
