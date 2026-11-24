"""Workstation side of the production freeze: real HPO outcome -> v2 protocols -> reuse audit -> pinned deployment bundle.

check-hpo   Validate the HPO decision files fetched from the IBEX HPO campaign (bridge.py fetch-hpo): fetch hashes match
            the remote hashes, the campaign is DONE, recipe.json is frozen and bound to final.json and to the user's
            selection decision (common leaderboard CALIBRATED), the worker's leaderboard.py freeze receipt agrees with
            it (crosscheck agreement, no mismatch), neither receipt is synthetic, and both separately selected winners
            are present. Nothing downstream runs without a passed check.
protocols   Run the query worker's make_protocol.py generate (FROZEN only) and validate on the fetched freeze receipt,
            the selection decision, the v3 main manifest and the three excluded entries.
reuse-audit FROZEN_HPO_ARTIFACT audit (reuse.py) of the calibrated winner's initializer for main-PFS-f0-ps0 and, only if
            that passes, of the winner trajectory for main-PFS-f0-qs0-full13-shared, against a template config the
            worker builds for that unit. Writes REUSE.json; an absent or failed audit means no reuse.
bundle      Pin the committed production source (executed files clean, HEAD pushed) and write BUNDLE.json plus the exact
            coordinator commands for source checkout and input transfer. --rehearsal skips the Git check and marks the
            bundle so no host will release it.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUITE, Refusal, check, read_json, sha, write_exclusive, write_replace  # noqa: E402
import reuse as R  # noqa: E402
import units as U  # noqa: E402

HPO_FILES = ('DONE.json', 'recipe.json', 'recipe.frozen.json', 'recipe.crosscheck.json', 'recipe.crosscheck-agreement.json', 'final.json', 'final.frozen.json',
             'selection-decision.json', 'selection-decision.receipt.json', 'screen.json', 'frozen-manifest.json')
EXECUTED = ('production_ops/common.py', 'production_ops/units.py', 'production_ops/campaign.py', 'production_ops/controller.py', 'production_ops/site-policy.json',
            'production_ops/jobs/orix-job.sh', 'production_ops/jobs/ibex-job.sh', 'production_ops/jobs/ibex-inference-job.sh', 'production_ops/build_query_config.py',
            'production_ops/reuse.py', 'production_ops/readout.py', 'production_ops/export_terminal.py', 'production_ops/inference_job.py', 'production.py',
            'pretrain/train.py', 'pretrain/complex_model.py', 'pretrain/state_compare.py', 'pretrain/make_config.py')
V3 = 'data/clmpt-suite-inputs-20261123-v3'


def check_hpo(args):
    folder = Path(args.hpo_dir)
    fetch = read_json(folder / 'FETCH-RECEIPT.json')
    problems = []
    listed = set(fetch.get('remote_decisions_listing', []))
    for name in HPO_FILES:
        path = folder / name
        remote = fetch['files'].get(name)
        if not path.exists() or remote is None or sha(path) != remote:
            problems.append('fetched file missing or differs from its remote hash: %s' % name)
    if any(name.endswith('crosscheck-mismatch.json') for name in listed):
        problems.append('the HPO campaign holds a selector cross-check mismatch')
    check(not problems, 'HPO outcome refused: %s' % problems)
    recipe, frozen, theirs = read_json(folder / 'recipe.json'), read_json(folder / 'recipe.frozen.json'), read_json(folder / 'recipe.crosscheck.json')
    agreement, final, final_frozen = read_json(folder / 'recipe.crosscheck-agreement.json'), read_json(folder / 'final.json'), read_json(folder / 'final.frozen.json')
    user = Path(args.user_decision)
    decision = read_json(user)
    checks = {
        'recipe_frozen_binding': frozen['decision_sha256'] == sha(folder / 'recipe.json') and frozen['target'] == 'recipe',
        'final_frozen_binding': final_frozen['decision_sha256'] == sha(folder / 'final.json') and final_frozen['target'] == 'final',
        'recipe_binds_final': recipe['final_sha256'] == sha(folder / 'final.json'),
        'recipe_binds_campaign_decision': recipe['decision_sha256'] == sha(folder / 'selection-decision.json'),
        'campaign_decision_is_user_decision': sha(folder / 'selection-decision.json') == sha(user),
        'calibrated_common_leaderboard': decision['common_leaderboard'] == 'calibrated' == recipe['common_leaderboard'],
        'no_calibration_presumption': recipe['calibration_benefit_presumed'] is False,
        'no_test_access': recipe['test_access'] is False and theirs['test_access'] is False and final['test_access'] is False,
        'worker_freeze_receipt': theirs['command'] == 'freeze' and theirs['decision'] == decision and not theirs.get('synthetic') and not recipe.get('synthetic'),
        'worker_and_root_recipe_agree': theirs['production_recipe'] == recipe['production_recipe'],
        'crosscheck_agreement': agreement['differences'] == [] and agreement['crosscheck_sha256'] == sha(folder / 'recipe.crosscheck.json'),
        'both_winners_present': set(final['winners']) == {'raw', 'calibrated'} and set(theirs['separately_selected_winners']) == {'raw', 'calibrated'},
        'campaign_done': read_json(folder / 'DONE.json').get('recipe_decision_sha256') == sha(folder / 'recipe.json'),
    }
    receipt = {'checks': checks, 'passed': all(checks.values()), 'hpo_dir': str(folder.resolve()), 'files': {name: sha(folder / name) for name in HPO_FILES},
               'production_recipe': recipe['production_recipe'], 'raw_winner': final['winners']['raw'], 'calibrated_winner': final['winners']['calibrated'],
               'other_view_recipe_differs': recipe['other_view_recipe_differs'], 'hpo_campaign': fetch.get('campaign'), 'time': time.time(),
               'note': 'raw and calibrated winners stay separate; the common recipe trains every production unit; per-run raw and calibrated checkpoints are both kept'}
    write_exclusive(args.output, receipt)
    print(json.dumps({'passed': receipt['passed'], 'checks': checks}, indent=2))
    return 0 if receipt['passed'] else 1


def protocols(args):
    prod = Path(args.prod)
    hpo = Path(args.hpo_dir)
    receipt = read_json(args.hpo_check)
    check(receipt['passed'] is True and receipt['files']['recipe.crosscheck.json'] == sha(hpo / 'recipe.crosscheck.json'), 'HPO check not passed or not for these files')
    source = Path(args.source_root).resolve()
    worker = source / 'code' / 'clmpt_suite' / 'query' / 'code' / 'make_protocol.py'
    out = prod / 'protocols'
    excluded = [str(source / V3 / 'excluded' / 'entries' / ('excluded-%s.json' % modality)) for modality in ('F', 'S', 'FS')]
    generate = [args.python, str(worker), 'generate', '--query-protocol', str(source / 'code/clmpt_suite/query/protocols/suite-query-v1.json'),
                '--pretrain-protocol', str(source / 'code/clmpt_suite/pretrain/protocols/suite-pretrain-v1.json'), '--freeze', str(hpo / 'recipe.crosscheck.json'),
                '--decision', str(hpo / 'selection-decision.json'), '--manifest', str(source / V3 / 'MANIFEST-final-20261123.json'),
                '--root', str(source), '--output-query', str(out / 'suite-query-v2.json'), '--output-pretrain', str(out / 'suite-pretrain-v2.json')]
    for path in excluded:
        generate += ['--excluded-entry', path]
    first = subprocess.run(generate, capture_output=True, text=True)
    write_exclusive(out / 'generate-run.json', {'command': generate, 'returncode': first.returncode, 'stdout': first.stdout[-20000:], 'stderr': first.stderr[-8000:]})
    check(first.returncode == 0, 'make_protocol generate refused: %s' % first.stderr[-3000:])
    query, pretrain = read_json(out / 'suite-query-v2.json'), read_json(out / 'suite-pretrain-v2.json')
    check(query['status'] == pretrain['status'] == 'FROZEN', 'generated protocols are not FROZEN')
    check(query['production_recipe']['common_leaderboard'] == 'calibrated', 'production recipe is not the calibrated winner')
    validate = [args.python, str(worker), 'validate', '--query-protocol', str(out / 'suite-query-v2.json'), '--pretrain-protocol', str(out / 'suite-pretrain-v2.json'),
                '--output', str(out / 'validate-receipt.json')]
    second = subprocess.run(validate, capture_output=True, text=True)
    write_exclusive(out / 'validate-run.json', {'command': validate, 'returncode': second.returncode, 'stdout': second.stdout[-20000:], 'stderr': second.stderr[-8000:]})
    check(second.returncode == 0 and read_json(out / 'validate-receipt.json')['passed'] is True, 'make_protocol validate failed')
    summary = {'query_v2_sha256': sha(out / 'suite-query-v2.json'), 'pretrain_v2_sha256': sha(out / 'suite-pretrain-v2.json'),
               'validate_receipt_sha256': sha(out / 'validate-receipt.json'), 'hpo_check_sha256': sha(args.hpo_check), 'production_recipe': query['production_recipe']}
    write_exclusive(out / 'PROTOCOLS.json', summary)
    print(json.dumps(summary, indent=2))
    return 0


def query_template(args, prod, source, export):
    """The production config the reused query unit would get, built by the worker's make_config (no export on this host)."""
    plan = U.plan()['query'][R.QUERY_UNIT]
    protocol = read_json(prod / 'protocols' / 'suite-query-v2.json')
    recipe = protocol['production_recipe']
    inputs, _ = input_records(source)
    folder = prod / 'reuse-template'
    folder.mkdir(parents=True, exist_ok=True)
    request = {'output': str(folder / 'template-config.json'), 'protocol': str(prod / 'protocols' / 'suite-query-v2.json'), 'experiment_id': R.QUERY_UNIT,
               'data_dir': str(source / V3 / inputs[plan['entry']]['bank']['relative']), 'pretrained': export['ibex_path'], 'pretrained_sha256': export['sha256'],
               'initializer': {'taxonomy_mode': 'original', 'cohort': 'main', 'modality': 'PFS', 'fold': 0, 'pretrain_seed': 0, **recipe['initializer_recipe']},
               'seed': plan['seed'], 'sampler_seed': plan['sampler_seed'], 'encoder': plan['encoder'], 'curriculum': plan['curriculum'],
               'learning_rate': recipe['learning_rate'], 'temperature': recipe['temperature'], 'maximum_steps': 20000, 'evaluation_every_updates': 500, 'maximum_seconds': 13200}
    write_replace(folder / 'request.json', request)
    for leftover in (folder / 'template-config.json', folder / 'receipt.json'):
        if leftover.exists():
            leftover.unlink()
    result = subprocess.run([args.python, '-I', str(Path(__file__).resolve().parent / 'build_query_config.py'), '--worker-code', str(source / 'code/clmpt_suite/query/code'),
                             '--request', str(folder / 'request.json'), '--receipt', str(folder / 'receipt.json'), '--skip-pretrained-hash'], capture_output=True, text=True)
    check(result.returncode == 0, 'template config refused by the worker: %s' % result.stderr[-2000:])
    template = read_json(folder / 'template-config.json')
    expected = {key: template[key] for key in R.QUERY_SCIENTIFIC}
    expected.update(initializer=template['initializer'], template=template, template_keys=sorted(template))
    return expected, {name: template['source_hashes'][name] for name in R.QUERY_SOURCES}, template['reference_hashes']


def reuse_audit(args):
    """FROZEN_HPO_ARTIFACT audit from fetched evidence (bridge.py fetch-reuse-evidence). Missing evidence or any failed
    check means the planned unit trains; REUSE.json records exactly what is reused and why."""
    prod, source = Path(args.prod), Path(args.source_root).resolve()
    hpo = prod / 'hpo'
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else hpo / 'reuse'
    fetched = read_json(evidence_dir / 'FETCH-RECEIPT.json')
    for name, digest in fetched['files'].items():
        check(sha(evidence_dir / name) == digest, 'reuse evidence differs from its remote hash: %s' % name)
    protocols = read_json(prod / 'protocols' / 'PROTOCOLS.json')
    pretrain_v2 = read_json(prod / 'protocols' / 'suite-pretrain-v2.json')
    production = protocols['production_recipe']
    inputs, _ = input_records(source)
    recipe = read_json(Path(args.recipe) if args.recipe else hpo / 'recipe.json')
    hpo_manifest = read_json(Path(args.hpo_manifest) if args.hpo_manifest else hpo / 'frozen-manifest.json')
    export_name = 'update-%07d.pt' % production['initializer_recipe']['initializer_update_budget']
    evidence = {'recipe': recipe, 'hpo_manifest': hpo_manifest, 'admission': read_json(evidence_dir / 'ADMISSION.json'), 'admission_sha256': sha(evidence_dir / 'ADMISSION.json'),
                'identity': read_json(evidence_dir / 'identity.json'), 'identity_sha256': sha(evidence_dir / 'identity.json'), 'config': read_json(evidence_dir / 'config.json'),
                'config_sha256': sha(evidence_dir / 'config.json'), 'export_name': export_name}
    pretrain = R.audit_pretrain(evidence, inputs, production, pretrain_v2['fixed'])
    result = {'pretrain': pretrain, 'query': None, 'time': time.time(), 'planned_units_unchanged': {'pretrain': 109, 'query': 185},
              'policy': 'reuse is admitted only for main-PFS-f0-ps0 and main-PFS-f0-qs0-full13-shared as FROZEN_HPO_ARTIFACT; planned counts stay 109/185 and reuse is counted separately'}
    if pretrain['passed']:
        identity, complete = read_json(evidence_dir / 'trajectory-identity.json'), read_json(evidence_dir / 'trajectory-COMPLETE.json')
        expected, sources, references = query_template(args, prod, source, pretrain['export'])
        query = R.audit_query(identity, complete, pretrain, expected, sources, references)
        query.update({'training_dir': fetched['winner_training_dir'], 'complete_sha256': fetched['files']['trajectory-COMPLETE.json'],
                      'identity_sha256': fetched['files']['trajectory-identity.json']})
        result['query'] = query
    write_exclusive(prod / 'REUSE.json', result)
    print(json.dumps({'pretrain_reused': pretrain['passed'], 'query_reused': bool(result['query'] and result['query']['passed']),
                      'failed_checks': [key for key, value in pretrain['checks'].items() if not value] + ([key for key, value in result['query']['checks'].items() if not value] if result['query'] else [])}, indent=2))
    return 0


def git(root, *arguments):
    return subprocess.run(['git', '-C', str(root)] + list(arguments), capture_output=True, text=True, timeout=60, check=True).stdout.strip()


def source_pin(source, revision, rehearsal):
    suite = source / 'code' / 'clmpt_suite'
    files = list(EXECUTED) + ['query/code/' + path.name for path in sorted((suite / 'query' / 'code').iterdir()) if path.is_file() and path.suffix == '.py']
    hashes = {name: sha(suite / name) for name in files}
    if rehearsal:
        return {'revision': revision, 'files': hashes, 'verified': False, 'rehearsal': True}
    head = git(source, 'rev-parse', 'HEAD')
    check(head == revision, 'HEAD %s differs from --revision %s' % (head, revision))
    check(git(source, 'rev-parse', '@{upstream}') == head, 'HEAD is not the pushed upstream commit')
    status = git(source, 'status', '--porcelain', '--untracked-files=all', '--', *('code/clmpt_suite/' + name for name in files))
    check(not status, 'executed production files are uncommitted or modified: %s' % status.splitlines())
    return {'revision': revision, 'files': hashes, 'verified': True, 'rehearsal': False}


def input_records(source):
    manifest = read_json(source / V3 / 'MANIFEST-final-20261123.json')
    check(manifest['format'] == 'clmpt-suite-inputs-v1' and manifest['complete'] is True and len(manifest['entries']) == 70, 'main manifest incomplete')
    entries = {name: dict(entry) for name, entry in manifest['entries'].items()}
    for modality in ('F', 'S', 'FS'):
        entries['excluded-%s' % modality] = read_json(source / V3 / 'excluded' / 'entries' / ('excluded-%s.json' % modality))
    records = {}
    for name, entry in sorted(entries.items()):
        record = {'cohort': entry.get('cohort', 'main'), 'modality': entry['modality'], 'fold': entry['fold']}
        for kind in ('graph', 'bank', 'test'):
            directory = Path(entry['%s_dir' % kind])
            relative = str(directory.relative_to(V3)) if not directory.is_absolute() else str(directory.relative_to(source / V3))
            record[kind] = {'relative': relative, 'complete_sha256': entry['%s_complete_sha256' % kind]}
            check(sha(source / V3 / relative / 'COMPLETE.json') == record[kind]['complete_sha256'], 'local %s COMPLETE differs: %s' % (kind, name))
        record['pretraining'] = entry['pretraining']
        records[name] = record
    return records, {'manifest_sha256': sha(source / V3 / 'MANIFEST-final-20261123.json'), 'root_sha256': sha(source / V3 / 'ROOT.json')}


def bundle(args):
    prod, source = Path(args.prod), Path(args.source_root).resolve()
    out = prod / 'bundle'
    check(not (out / 'BUNDLE.json').exists(), 'bundle exists; bundles are write-once')
    summary = read_json(prod / 'protocols' / 'PROTOCOLS.json')
    for name, key in (('suite-query-v2.json', 'query_v2_sha256'), ('suite-pretrain-v2.json', 'pretrain_v2_sha256')):
        check(sha(prod / 'protocols' / name) == summary[key], 'protocol changed after generation: %s' % name)
        write_exclusive(out / name, (prod / 'protocols' / name).read_bytes())
    pin = source_pin(source, args.revision, args.rehearsal)
    plan = U.plan()
    inputs, identity = input_records(source)
    for unit in list(plan['pretrain'].values()) + list(plan['query'].values()):
        check(unit['entry'] in inputs, 'unit reads an unknown entry: %s' % unit['id'])
    reuse = read_json(prod / 'REUSE.json') if (prod / 'REUSE.json').exists() else {'pretrain': None, 'query': None, 'note': 'no audit: no reuse'}
    hpo_check_path = Path(args.hpo_check) if args.hpo_check else prod / 'hpo-check.json'
    hpo_check = dict(read_json(hpo_check_path), sha256=sha(hpo_check_path))
    check(args.rehearsal or (hpo_check['passed'] is True and hpo_check['sha256'] == summary['hpo_check_sha256']), 'the HPO check is not passed or not the one the protocols were generated from')
    policy = read_json(Path(__file__).resolve().parent / 'site-policy.json')
    reused = [record['unit'] for record in (reuse.get('pretrain'), reuse.get('query')) if record and record.get('passed')]
    value = {'schema': 'clmpt-suite-production-bundle/1', 'created': time.time(), 'source': pin, 'rehearsal': bool(args.rehearsal),
             'protocols': {'query': {'name': 'suite-query-v2.json', 'sha256': summary['query_v2_sha256']}, 'pretrain': {'name': 'suite-pretrain-v2.json', 'sha256': summary['pretrain_v2_sha256']}},
             'production_recipe': summary['production_recipe'], 'hpo_check_sha256': summary['hpo_check_sha256'], 'hpo_check': hpo_check, 'inputs': inputs, 'inputs_identity': identity,
             'units': plan, 'summary_groups': U.summaries(plan), 'reuse': reuse, 'site_policy_sha256': sha(Path(__file__).resolve().parent / 'site-policy.json'),
             'site_policy': policy, 'counts': {'pretrain': len(plan['pretrain']), 'query': len(plan['query']), 'reused_units_counted_separately': reused,
                                               'inference_readouts': len(plan['query']), 'inference_note': 'one short TEST readout job per query unit; not a training trajectory'},
             'test': {'status': 'per_unit_release', 'rule': 'after the HPO common recipe is globally frozen (this bundle) and each unit\'s validation selections are frozen; '
                                                           'query/code/test_eval.py with a per-unit RELEASED file; readout.py adds pools and S labels from the same saved matrices'},
             'deferred': {'robert_replay': 'not started (score-analysis branch over the archived TEST matrices and readout masks)', 'native_baseline_replay': 'not started'}}
    write_exclusive(out / 'BUNDLE.json', value)
    write_exclusive(out / 'REQUIRED-COMMANDS.md', commands(args, value).encode())
    print(json.dumps({'bundle': str(out / 'BUNDLE.json'), 'sha256': sha(out / 'BUNDLE.json'), 'rehearsal': value['rehearsal'], 'counts': value['counts']}, indent=2))
    return 0


def commands(args, value):
    revision = value['source']['revision']
    q = shlex.quote
    return '\n'.join([
        '# Coordinator commands for the production deployment (generated; not executed by the ops worker)', '',
        'Bundle: `%s`, source revision `%s`.' % (q(str(Path(args.prod) / 'bundle' / 'BUNDLE.json')), revision), '',
        '## Source checkouts (separate from the frozen HPO checkout)', '', '```bash',
        'orixlogin "git clone --single-branch <remote> ~/Git/link-gda-clmpt-suite-prod && git -C ~/Git/link-gda-clmpt-suite-prod checkout %s"' % revision,
        'ssh zhapacfp@glogin.ibex.kaust.edu.sa "git clone --single-branch <remote> /ibex/user/zhapacfp/link-gda-clmpt-suite-prod-src && git -C /ibex/user/zhapacfp/link-gda-clmpt-suite-prod-src checkout %s"' % revision,
        '```', '', '## Inputs (v3 root only, without work/; COMPLETE hashes are re-verified by campaign.py stage)', '', '```bash',
        'rsync -a --exclude work/ %s/ zhapacfp@orix-login.kaust.edu.sa:/mnt/data/u/zhapacfp/link-gda/inputs/clmpt-suite-inputs-20261123-v3/   # graphs (Orix needs graphs/, bases/, excluded/)' % V3,
        'rsync -a --exclude work/ --exclude graphs/ %s/ zhapacfp@glogin.ibex.kaust.edu.sa:/ibex/user/zhapacfp/link-gda-clmpt-suite-inputs-20261123-v3/   # banks and TEST only' % V3,
        '```', '', 'Use the rsync shell of the bridge config for Orix. Then run `campaign.py stage` on each host (README).', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('check-hpo')
    p.add_argument('--hpo-dir', required=True)
    p.add_argument('--user-decision', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('protocols')
    for name in ('--prod', '--hpo-dir', '--hpo-check', '--source-root', '--python'):
        p.add_argument(name, required=True)
    p = sub.add_parser('reuse-audit')
    p.add_argument('--prod', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--python', required=True, help='interpreter with torch for the worker template config')
    p.add_argument('--evidence-dir', help='default <prod>/hpo/reuse')
    p.add_argument('--recipe', help='default <prod>/hpo/recipe.json')
    p.add_argument('--hpo-manifest', help='default <prod>/hpo/frozen-manifest.json')
    p = sub.add_parser('bundle')
    p.add_argument('--prod', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--hpo-check', help='default <prod>/hpo-check.json')
    p.add_argument('--rehearsal', action='store_true')
    args = parser.parse_args()
    try:
        return {'check-hpo': check_hpo, 'protocols': protocols, 'reuse-audit': reuse_audit, 'bundle': bundle}[args.command](args)
    except Refusal as error:
        print(json.dumps({'refused': str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
