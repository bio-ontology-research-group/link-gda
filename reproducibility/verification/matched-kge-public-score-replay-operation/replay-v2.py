"""Replay every matched score condition, planned comparison and table from public files."""
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent/'full-matched-public-replay-v1'
SOURCE = SHARED/'source'
COMMIT = '68101ed08e3e271f5f2276724f8a5338b9cf32a2'
sys.path[:0] = [str(SOURCE/'code'), str(SOURCE/'reproducibility')]
from phenoquery.common import read, sha, write
from build_matched_kge_scores import replay
from generate_matched_kge_tables import render


def compare_tables(reference, actual, expected_hashes):
    assert set(reference) == set(actual) == set(expected_hashes) and len(reference) == 2
    for name,old in reference.items():
        assert hashlib.sha256(old.encode()).hexdigest() == expected_hashes[name], name
        old_comment,separator,old_body = old.partition('\n')
        new_comment,new_separator,new_body = actual[name].partition('\n')
        assert separator == new_separator == '\n'
        for comment in (old_comment,new_comment):
            assert comment.startswith('% Summary SHA256: ')
            digest = comment.removeprefix('% Summary SHA256: ')
            assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        assert old_body == new_body, name


def main():
    started = time.time()
    plan = read(ROOT/'replay-plan-v2.json')
    source,download = read(SHARED/'SOURCE_READY.json'),read(ROOT/'DOWNLOAD_COMPLETE.json')
    assert source['status'] == download['status'] == 'passed'
    assert source['commits']['source'] == download['source_commit'] == COMMIT
    assert plan['source_proof_sha256'] == download['source_proof_sha256'] == sha(SHARED/'SOURCE_READY.json')
    assert plan['script_sha256'] == sha(Path(__file__)) and plan['sbatch_sha256'] == sha(ROOT/'replay-v2.sbatch')
    assert download['script_sha256'] == sha(ROOT/'download-v2.py')
    assert download['download_plan_sha256'] == sha(ROOT/'download-plan-v2.json')
    assert download['manifest_sha256'] == plan['manifest_sha256'] == sha(ROOT/'release-manifest.json')
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == COMMIT
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True)
    for name,digest in source['analysis_files'].items():
        assert sha(SOURCE/name) == digest, name
    workspace, output = ROOT/'workspace',ROOT/'replayed'
    assert not output.exists() and not (ROOT/'REPLAY_COMPLETE.json').exists()
    assert not any(path.is_symlink() for path in workspace.rglob('*'))
    p = read(ROOT/'release-manifest.json')['provenance']
    for name,record in p['expected_files'].items():
        path = workspace/name
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
    reference = workspace/p['reference_summary']
    assert sha(reference) == plan['summary_sha256'] == p['replay']['summary_sha256']
    assert sha(workspace/p['reference_report']) == plan['report_sha256'] == p['replay']['report_sha256']
    assert p['freeze_sha256'] == plan['freeze_sha256'] == download['freeze_sha256']
    result = replay(workspace/p['score_manifest'],workspace/p['data_root'],
        workspace/p['reference_report'],reference,workspace/p['reference_ranks'],output,workers=4)
    assert result['status'] == 'passed' and result['summary_values_exact']
    assert result['models'] == 40 and result['conditions_per_model'] == 42 and result['primary_comparisons'] == 10
    original_tables = render(read(reference),sha(reference))
    actual_tables = render(read(output/'summary.json'),sha(output/'summary.json'))
    compare_tables(original_tables,actual_tables,plan['primary_tables_sha256'])
    tables = output/'tables'; tables.mkdir()
    for name,value in actual_tables.items(): (tables/name).write_text(value)
    bias = output/'bias-tables'
    subprocess.run([sys.executable,str(SOURCE/'reproducibility/generate_production_bias_tables.py'),
        '--report-json',str(output/'evaluation.json'),'--score-manifest',str(workspace/p['score_manifest']),
        '--out-dir',str(bias)],check=True)
    bias_files = {path.name:sha(path) for path in bias.iterdir()}
    assert sum(name.endswith('.tex') for name in bias_files) == 5
    assert bias_files == plan['bias_files_sha256']
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True)
    proof = dict(status='passed',source_commit=COMMIT,score_files=400,variants=40,conditions=42,
        primary_comparisons=10,source_proof_sha256=sha(SHARED/'SOURCE_READY.json'),
        download_proof_sha256=sha(ROOT/'DOWNLOAD_COMPLETE.json'),manifest_sha256=sha(ROOT/'release-manifest.json'),
        replay_plan_sha256=sha(ROOT/'replay-plan-v2.json'),script_sha256=sha(Path(__file__)),
        score_replay_proof_sha256=sha(output/'verification.json'),report_sha256=sha(output/'evaluation.json'),
        summary_sha256=sha(output/'summary.json'),tables={name:sha(tables/name) for name in actual_tables},
        bias_files=bias_files,elapsed_seconds=time.time()-started,
        table_comparison='Exact bodies and validated source-summary checksum comments; rank-container recompression may change the summary hash.',
        scope='Fresh public scores reproduce all 40 x 42 conditions, every retained rank array, ten planned comparisons, two primary tables and all five bias panels/metrics. Independent model inference remains separate.')
    write(ROOT/'REPLAY_COMPLETE.json',proof)
    print(proof,flush=True)


if __name__ == '__main__': main()
