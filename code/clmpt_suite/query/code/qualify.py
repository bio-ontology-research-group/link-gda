"""Stage-0 qualification diagnostics for the suite query and pretraining engines.

Every subcommand writes one JSON receipt with an explicit 'passed' flag and the measured numbers; none of
them trains a reportable model or reads TEST.

streams       CPU. Same bank and sampler seed: clinical draws (profiles, positives, 128 negatives) identical
              across full13, core3 and aux_off for N updates; full13 and core3 ontology draws differ; another
              sampler seed changes clinical and ontology draws; save/restore mid-stream reproduces later draws.
              --synthetic builds a small in-memory bank for a local syntax/logic check.
construction  Seeded construction parity: Suite shared at seed 0 equals the frozen-trainer construction
              (seed 0, original model with packed lookup) parameter for parameter; independent equals shared;
              mean equals shared minus the Transformer; a different seed changes the query-stage parameters
              and leaves the pretrained ComplEx tables unchanged.
scores        At matching weights (seed-0 construction): suite shared scores vs the frozen model's
              QueryModel.scores on validation cases; suite shared loss vs frozen QueryModel.loss on one drawn
              update (eval mode); independent profile score equals the mean of per-branch cosines computed
              branch by branch; mean-encoder 1p output equals the hand-computed mean of message and state;
              clinical-only loss equals the clinical rows' summed cross-entropy divided by 128.
resume        Runs the real worker (query or pretrain train.py) twice from one qualification config:
              uninterrupted, and paused at a step then resumed in a fresh process; compares the terminal
              full states, update logs and (query) saved score matrices.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[2] / 'clmpt_boundary_queries' / 'query' / 'code'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n')
    print(json.dumps({'receipt': str(path), 'passed': value['passed']}))
    return 0 if value['passed'] else 1


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_bank(seed=0):
    """Small consistent bank: 3 ontology patterns, 40 training profiles, 300 classes, 400 genes."""
    rng = np.random.default_rng(seed)
    classes, genes = list(range(1000, 1300)), list(range(2000, 2400))
    ontology = []
    for kind in ('1p', 'pip', 'ppip', 'P(P(Anchor))'):
        for _ in range(25):
            answers = sorted(map(int, rng.choice(classes, size=3, replace=False)))
            ontology.append({'type': kind, 'domain': 'class', 'answers': answers, 'negative_known': answers, 'sampleable': True})
    profiles = []
    for _ in range(40):
        answers = sorted(map(int, rng.choice(genes, size=2, replace=False)))
        profiles.append({'type': 'profile', 'domain': 'gene', 'answers': answers, 'negative_known': answers, 'sampleable': True})
    return {'ontology_queries': ontology, 'training_profiles': profiles, 'class_ids': classes, 'candidate_ids': genes,
            'ontology_patterns': [{'type': kind} for kind in ('1p', 'pip', 'ppip', 'P(P(Anchor))')]}


def streams(args):
    from sampler import Sampler
    protocol = json.loads(Path(args.protocol).read_text())
    if args.synthetic:
        bank = synthetic_bank()
        mixtures = {'full13': dict(protocol['mixtures']['F13_sqrt'], patterns=['1p', 'P(P(Anchor))', 'pip', 'ppip']), 'core3': protocol['mixtures']['C3_sqrt'], 'aux_off': None}
    else:
        bank = json.loads((Path(args.bank) / 'bank.json').read_text())
        mixtures = {name: (protocol['mixtures'][entry['mixture_id']] if entry['mixture_id'] else None) for name, entry in protocol['curricula'].items()}
    seed, other = args.sampler_seed, args.other_sampler_seed
    samplers = {name: Sampler(bank, mixture, seed) for name, mixture in mixtures.items()}
    alternate = Sampler(bank, mixtures['full13'], other)
    replay = Sampler(bank, mixtures['full13'], seed)
    receipts = {name: [] for name in samplers}
    alternate_receipts, replay_receipts = [], []
    saved = None
    for step in range(args.steps):
        for name, sampler in samplers.items():
            receipts[name].append(sampler.draw(step)[3])
        alternate_receipts.append(alternate.draw(step)[3])
        if step == args.steps // 2:
            saved = copy.deepcopy(samplers['full13'].state())
    replay.restore(saved)
    for step in range(args.steps // 2 + 1, args.steps):
        replay_receipts.append(replay.draw(step)[3])
    clinical = {name: [row['clinical_draw_sha256'] for row in rows] for name, rows in receipts.items()}
    ontology = {name: [row['query_indices'][:len(row['query_indices']) - 32] for row in rows] for name, rows in receipts.items()}
    checks = {
        'clinical_identical_full13_core3': clinical['full13'] == clinical['core3'],
        'clinical_identical_full13_aux_off': clinical['full13'] == clinical['aux_off'],
        'aux_off_draws_no_ontology_rows': all(not rows for rows in ontology['aux_off']),
        'ontology_differs_full13_core3': ontology['full13'] != ontology['core3'],
        'core3_only_core_patterns': all(set(row['ontology_pattern_counts']) <= {'1p', 'pip', 'ppip'} for row in receipts['core3']),
        'other_seed_changes_clinical': all(a != b['clinical_draw_sha256'] for a, b in zip(clinical['full13'], alternate_receipts)),
        'other_seed_changes_ontology': ontology['full13'] != [row['query_indices'][:-32] for row in alternate_receipts],
        'restore_reproduces_later_draws': [row['draw_sha256'] for row in replay_receipts] == [row['draw_sha256'] for row in receipts['full13'][args.steps // 2 + 1:]],
    }
    checks['gene_domain_source_follows_bank'] = (samplers['full13'].gene_domain['source'] == 'training_gene_ids') == ('training_gene_ids' in bank)
    if args.synthetic:
        extended = synthetic_bank()
        extra = list(range(5000, 5100))
        extended['training_gene_ids'] = extended['candidate_ids'] + extra
        extended['training_profiles'][0] = dict(extended['training_profiles'][0], answers=[extra[0]], negative_known=[extra[0]])
        explicit = Sampler(extended, None, seed)
        universe, candidates = set(explicit.gene_domain['ids']), set(extended['candidate_ids'])
        negatives, positives = set(), set()
        for step in range(args.steps):
            _, positive, negative, _ = explicit.draw(step)
            positives.update(positive)
            negatives.update(value for row in negative for value in row)
        checks['explicit_domain_source'] = explicit.gene_domain['source'] == 'training_gene_ids' and explicit.gene_domain['outside_candidate_pool'] == len(extra)
        checks['explicit_domain_negatives_inside_universe'] = negatives <= universe
        checks['explicit_domain_negatives_reach_outside_candidates'] = bool(negatives - candidates)
        checks['explicit_domain_keeps_outside_pool_positive'] = extra[0] in positives
        try:
            Sampler(dict(extended, training_gene_ids=extended['candidate_ids']), None, seed)
            checks['positive_outside_universe_refused'] = False
        except AssertionError:
            checks['positive_outside_universe_refused'] = True
    return write(args.output, {'subcommand': 'streams', 'synthetic': args.synthetic, 'steps': args.steps, 'sampler_seed': seed, 'other_sampler_seed': other,
                               'checks': checks, 'passed': all(checks.values()), 'sampler_source_sha256': file_sha(HERE / 'sampler.py'),
                               'gene_domain': {key: value for key, value in samplers['full13'].gene_domain.items() if key != 'ids'},
                               'first_clinical_draws': {name: values[:3] for name, values in clinical.items()}})


def domain(args):
    """Protocol record for one bank: counts and clinical TRAIN gene universe, recomputed from bank.json."""
    from sampler import domain_identity, gene_domain
    directory = Path(args.bank)
    manifest = json.loads((directory / 'COMPLETE.json').read_text())
    files_ok = {name: file_sha(directory / name) == (digest['sha256'] if isinstance(digest, dict) else digest) for name, digest in manifest['files'].items()}
    bank = json.loads((directory / 'bank.json').read_text())
    identity = domain_identity(gene_domain(bank))
    universe = set(gene_domain(bank)['ids'])
    record = {'validation_cases': len(bank['validation']), 'validation_associations': sum(len(row['gene_ids']) for row in bank['validation']),
              'candidate_genes': len(bank['candidate_ids']), 'entity_count': bank.get('num_entities'), 'clinical_training_gene_domain': identity}
    checks = {'bank_files_match_complete': all(files_ok.values()),
              'complete_domain_declaration_consistent': manifest.get('training_gene_domain', identity) == identity,
              'training_positives_inside_universe': all(set(row['answers']) <= universe for row in bank['training_profiles']),
              'validation_targets_inside_candidates': all(set(row['gene_ids']) <= set(bank['candidate_ids']) for row in bank['validation']),
              'universe_inside_entity_table': 'num_entities' in bank and 0 <= min(universe) and max(universe) < bank['num_entities'],
              'no_test_in_bank': 'test' not in bank and 'test_cases' not in bank}
    return write(args.output, {'subcommand': 'domain', 'bank': str(directory.resolve()), 'bank_complete_sha256': file_sha(directory / 'COMPLETE.json'),
                               'protocol_record_fields': record, 'checks': checks, 'passed': all(checks.values())})


def load_bank(path):
    return json.loads((Path(path) / 'bank.json').read_text())


def frozen_reference(pretrained, bank, temperature, device):
    """Construction exactly as the frozen trainer: global seed 0, then original model with packed lookup."""
    import torch
    from original_model import QueryModel as OriginalQueryModel
    from packed_encode import install
    for name in ('original_model.py', 'packed_encode.py'):
        assert file_sha(FROZEN / name) == file_sha(HERE / name), name
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = OriginalQueryModel(pretrained, bank['hp_ids'], device=device, temperature=temperature)
    return install(model)


def suite_model(pretrained, bank, encoder, seed, temperature, device):
    import torch
    from model import SuiteQueryModel
    from train import seed_everything
    seed_everything(seed)
    return SuiteQueryModel(pretrained, bank['hp_ids'], encoder, device=device, temperature=temperature)


def state_equal(left, right, skip=()):
    import torch
    keys = [key for key in left if not key.startswith(skip)]
    differing = [key for key in keys if key not in right or not torch.equal(left[key].cpu(), right[key].cpu())]
    return {'compared': len(keys), 'differing': differing[:50], 'equal': not differing}


def construction(args):
    import torch
    from train import pin_numerics
    if args.device == 'cuda':
        pin_numerics()
    bank = load_bank(args.bank)
    pretrained = torch.load(args.pretrained, map_location='cpu', weights_only=False)['model']
    reference = frozen_reference(pretrained, bank, args.temperature, args.device).state_dict()
    shared0 = suite_model(pretrained, bank, 'shared', 0, args.temperature, args.device).state_dict()
    independent0 = suite_model(pretrained, bank, 'independent', 0, args.temperature, args.device).state_dict()
    mean0 = suite_model(pretrained, bank, 'mean', 0, args.temperature, args.device).state_dict()
    shared1 = suite_model(pretrained, bank, 'shared', args.other_seed, args.temperature, args.device).state_dict()
    query_stage = [key for key in shared0 if not key.startswith(('nbp.', 'layer.nbp.'))]
    random_init = [key for key in query_stage if shared0[key].numel() > 1 and not bool((shared0[key] == shared0[key].flatten()[0]).all())]
    constant_init = sorted(set(query_stage) - set(random_init))
    checks = {
        'shared_seed0_equals_frozen_construction': state_equal(reference, shared0)['equal'] and set(reference) == set(shared0),
        'independent_seed0_equals_shared': state_equal(shared0, independent0)['equal'] and set(shared0) == set(independent0),
        'mean_has_no_transformer': not any(key.startswith('layer.transformer.') for key in mean0),
        'mean_seed0_equals_shared_without_transformer': state_equal(mean0, shared0)['equal'],
        'other_seed_changes_every_randomly_initialized_tensor': bool(random_init) and all(not torch.equal(shared0[key].cpu(), shared1[key].cpu()) for key in random_init),
        'other_seed_keeps_constant_initialized_tensors': all(torch.equal(shared0[key].cpu(), shared1[key].cpu()) for key in constant_init),
        'other_seed_keeps_pretrained_tables': all(torch.equal(shared0[key].cpu(), shared1[key].cpu()) for key in shared0 if key.startswith(('nbp.', 'layer.nbp.'))),
    }
    return write(args.output, {'subcommand': 'construction', 'device': args.device, 'other_seed': args.other_seed, 'pretrained_sha256': file_sha(args.pretrained),
                               'query_stage_tensors': len(query_stage), 'randomly_initialized': random_init, 'constant_initialized': constant_init,
                               'checks': checks, 'passed': all(checks.values())})


def scores(args):
    import torch
    import torch.nn.functional as F
    from evaluate import profile_query
    from sampler import Sampler
    from train import pin_numerics
    if args.device == 'cuda':
        pin_numerics()
    protocol = json.loads(Path(args.protocol).read_text())
    bank = load_bank(args.bank)
    pretrained = torch.load(args.pretrained, map_location='cpu', weights_only=False)['model']
    reference = frozen_reference(pretrained, bank, args.temperature, args.device).eval()
    shared = suite_model(pretrained, bank, 'shared', 0, args.temperature, args.device).eval()
    independent = suite_model(pretrained, bank, 'independent', 0, args.temperature, args.device).eval()
    mean = suite_model(pretrained, bank, 'mean', 0, args.temperature, args.device).eval()
    rows = bank['validation'][:args.cases]
    queries = [profile_query(row['hp_ids'], bank['relations']) for row in rows]
    candidates = torch.tensor(bank['candidate_ids'], device=args.device)
    measured = {}
    with torch.no_grad():
        old = reference.scores(queries, candidates)
        new = shared.profile_scores(queries, candidates)
        measured['shared_vs_frozen_scores_max_abs'] = float((old - new).abs().max())
        measured['shared_vs_frozen_scores_bitwise'] = bool(torch.equal(old, new))
        measured['shared_vs_frozen_row_argsort_equal'] = bool(torch.equal(torch.argsort(old, dim=1, stable=True), torch.argsort(new, dim=1, stable=True)))
        sampler = Sampler(bank, protocol['mixtures']['F13_sqrt'], 0)
        drawn, positive, negative, _ = sampler.draw(0)
        positive, negative = torch.tensor(positive, device=args.device), torch.tensor(negative, device=args.device)
        old_loss = reference.loss(drawn, positive, negative)
        new_loss = shared.suite_loss(drawn, positive, negative, 96, True)
        measured['shared_vs_frozen_loss'] = [float(old_loss), float(new_loss)]
        measured['shared_vs_frozen_loss_bitwise'] = bool(torch.equal(old_loss, new_loss))
        representation = shared.encode(drawn)
        target = shared.nbp._entity_embedding(positive)
        negatives = shared.nbp._entity_embedding(negative)
        logits = torch.cat([F.cosine_similarity(representation, target, dim=-1)[:, None], F.cosine_similarity(representation[:, None, :], negatives, dim=-1)], dim=1) / shared.temperature
        per_row = F.cross_entropy(logits, torch.zeros(len(drawn), dtype=torch.long, device=args.device), reduction='none')
        clinical_only = shared.suite_loss(drawn[96:], positive[96:], negative[96:], 0, False)
        measured['aux_off_loss_vs_clinical_sum_over_128'] = [float(clinical_only), float(per_row[96:].sum() / 128)]
        averaged = independent.profile_scores(queries[:8], candidates)
        manual = []
        from model import branch_queries
        for query in queries[:8]:
            parts = branch_queries(query)
            branch = F.normalize(independent.encode(parts), dim=-1) @ F.normalize(independent.nbp._entity_embedding(candidates), dim=-1).T
            manual.append(branch.mean(0))
        measured['independent_vs_branchwise_mean_cosine_max_abs'] = float((averaged - torch.stack(manual)).abs().max())
        representation = independent.profile_representation(queries[:8])
        measured['independent_representation_norm_range'] = [float(representation.norm(dim=-1).min()), float(representation.norm(dim=-1).max())]
        one_hop = next(query for query in drawn[:96] if query['type'] == '1p' and len(query['atoms']) == 1)
        head, role, tail = one_hop['atoms'][0]
        assert 'f' in (head, tail) and len(one_hop['constants']) == 1
        anchor_emb = mean.nbp.get_entity_emb([next(iter(one_hop['constants'].values()))])
        relation = mean.nbp.get_relation_emb([role])
        message = mean.nbp.estimate_tail_emb(anchor_emb, relation) if tail == 'f' else mean.nbp.estimate_head_emb(anchor_emb, relation)
        expected = (message + mean.layer.free_embedding.expand_as(message)) / 2
        measured['mean_1p_atom'] = one_hop['atoms']
        measured['mean_1p_vs_hand_mean_max_abs'] = float((mean.encode([one_hop]) - expected).abs().max())
    tolerance = args.tolerance
    checks = {
        'shared_scores_match_frozen': measured['shared_vs_frozen_scores_max_abs'] <= tolerance,
        'shared_loss_matches_frozen': abs(measured['shared_vs_frozen_loss'][0] - measured['shared_vs_frozen_loss'][1]) <= tolerance,
        'aux_off_loss_is_clinical_sum_over_128': abs(np.subtract(*measured['aux_off_loss_vs_clinical_sum_over_128'])) <= tolerance,
        'independent_score_is_mean_branch_cosine': measured['independent_vs_branchwise_mean_cosine_max_abs'] <= tolerance,
        'mean_1p_is_mean_of_message_and_state': measured.get('mean_1p_vs_hand_mean_max_abs', float('inf')) <= tolerance,
    }
    return write(args.output, {'subcommand': 'scores', 'device': args.device, 'cases': len(rows), 'tolerance': tolerance, 'measured': measured, 'checks': checks,
                               'passed': all(checks.values()), 'note': 'bitwise flags are reported, not required; tolerance is the acceptance threshold'})


def resume(args):
    import torch
    from state_compare import compare
    worker = (HERE / 'train.py') if args.engine == 'query' else (HERE.parents[1] / 'pretrain' / 'train.py')
    root = Path(args.workdir)
    assert not root.exists()
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0')
    straight, paused = root / 'uninterrupted', root / 'paused-resumed'
    calls = []
    for command in ([str(worker), '--config', args.config, '--output', str(straight)],
                    [str(worker), '--config', args.config, '--output', str(paused), '--pause-at-step', str(args.pause_at_step)],
                    [str(worker), '--config', args.config, '--output', str(paused)]):
        completed = subprocess.run([sys.executable] + command, env=environment)
        calls.append({'command': command, 'returncode': completed.returncode})
    expected_codes = [0, 75, 0]
    result = {'subcommand': 'resume', 'engine': args.engine, 'calls': calls, 'return_codes_ok': [call['returncode'] for call in calls] == expected_codes}
    if result['return_codes_ok']:
        terminal = {}
        for name, directory in (('uninterrupted', straight), ('paused', paused)):
            progress = json.loads((directory / 'progress.json').read_text())
            terminal[name] = torch.load(directory / progress['checkpoint'], map_location='cpu', weights_only=False)
        fields = [key for key in terminal['uninterrupted'] if key not in ('identity_sha256', 'log_bytes', 'log_sha256', 'curve', 'pending_checkpoint', 'initial_checkpoint', 'retained_best')]
        exact = compare({key: terminal['uninterrupted'][key] for key in fields}, {key: terminal['paused'][key] for key in fields})
        result['terminal_state_comparison'] = exact
        straight_log = [json.loads(line) for line in (straight / 'updates.jsonl').read_text().splitlines()]
        paused_log = [json.loads(line) for line in (paused / 'updates.jsonl').read_text().splitlines()]
        drop = ('update_seconds', 'gpu_peak_bytes')
        result['update_logs_equal_except_timing'] = [{k: v for k, v in row.items() if k not in drop} for row in straight_log] == [{k: v for k, v in row.items() if k not in drop} for row in paused_log]
        if args.engine == 'query':
            matrices = {}
            for path in sorted((straight / 'fits').glob('*.npz')):
                left, right = np.load(path), np.load(paused / 'fits' / path.name)
                matrices[path.name] = all(np.array_equal(left[key], right[key]) for key in left.files)
            result['score_matrices_equal'] = matrices
            result['selection_equal'] = json.loads((straight / 'selection.json').read_text()) == json.loads((paused / 'selection.json').read_text())
        result['passed'] = exact['exact'] and result['update_logs_equal_except_timing'] and all(result.get('score_matrices_equal', {'_': True}).values()) and result.get('selection_equal', True)
    else:
        result['passed'] = False
    result['scope'] = 'same device and process environment; equality here is not a cross-GPU guarantee'
    return write(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('streams')
    p.add_argument('--protocol', required=True)
    p.add_argument('--bank')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--sampler-seed', type=int, default=0)
    p.add_argument('--other-sampler-seed', type=int, default=1)
    p.add_argument('--output', required=True)
    for name in ('construction', 'scores'):
        p = sub.add_parser(name)
        p.add_argument('--protocol', required=True)
        p.add_argument('--bank', required=True)
        p.add_argument('--pretrained', required=True)
        p.add_argument('--temperature', type=float, default=.05)
        p.add_argument('--device', default='cuda')
        p.add_argument('--other-seed', type=int, default=1)
        p.add_argument('--cases', type=int, default=64)
        p.add_argument('--tolerance', type=float, default=1e-5)
        p.add_argument('--output', required=True)
    p = sub.add_parser('domain')
    p.add_argument('--bank', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('resume')
    p.add_argument('--engine', choices=['query', 'pretrain'], required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--pause-at-step', type=int, required=True)
    p.add_argument('--workdir', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'streams':
        assert args.synthetic or args.bank
    return {'streams': streams, 'domain': domain, 'construction': construction, 'scores': scores, 'resume': resume}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
