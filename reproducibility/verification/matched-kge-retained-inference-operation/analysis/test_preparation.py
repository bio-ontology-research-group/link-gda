import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('matched_operation', Path(__file__).with_name('verify-and-analyze.py'))
operation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operation)


def accounting():
    return '\n'.join(f'123_{fold}|{1000+fold}|COMPLETED|0:0' for fold in range(3))


def test_complete_native_accounting():
    assert operation.native_tasks(accounting(), 123) == {f:1000+f%3 for f in range(10)}


@pytest.mark.parametrize('change', [
    lambda x: '\n'.join(x.splitlines()[:-1]),
    lambda x: x+'\n'+x.splitlines()[0],
    lambda x: x.replace('123_2', '123_3'),
    lambda x: x.replace('123_2', '124_2'),
    lambda x: x.replace('123_2', '123_2.batch'),
    lambda x: x.replace('|1002|', '|1001|'),
    lambda x: x.replace('|1002|', '|invalid|'),
    lambda x: x.replace('COMPLETED', 'RUNNING', 1),
    lambda x: x.replace('0:0', '1:0', 1),
    lambda x: '',
])
def test_incomplete_or_inconsistent_accounting_rejected(change):
    with pytest.raises(ValueError): operation.native_tasks(change(accounting()), 123)


OLD, NEW = 'a'*64, 'b'*64
PREFIX = '% Summary SHA256: '


def test_bound_header_relocation():
    operation.compare_table_text(PREFIX+OLD+'\nvalue 1.25\n', PREFIX+NEW+'\nvalue 1.25\n', [(PREFIX,OLD,NEW)])


@pytest.mark.parametrize('left,right', [
    (PREFIX+OLD+'\nvalue 1.25\n', PREFIX+NEW+'\nvalue 1.26\n'),
    (PREFIX+OLD+'\n'+OLD+'\n', PREFIX+NEW+'\n'+NEW+'\n'),
    ('body\n'+PREFIX+OLD+'\n', 'body\n'+PREFIX+NEW+'\n'),
    (PREFIX+OLD+'\n', PREFIX+OLD+'\n'),
    (PREFIX+OLD+'\n'+PREFIX+OLD+'\n', PREFIX+NEW+'\n'+PREFIX+NEW+'\n'),
    (PREFIX+OLD+'\nvalue\n', PREFIX+NEW+'\nvalue'),
])
def test_header_normalization_cannot_hide_body_or_provenance_changes(left,right):
    with pytest.raises(ValueError): operation.compare_table_text(left,right,[(PREFIX,OLD,NEW)])


def fixture_exports(root):
    (root/'primary').mkdir(); (root/'bias').mkdir()
    for name in ('matched-kge-primary.tex', 'matched-kge-design.tex'):
        (root/'primary'/name).write_text('table')
    outputs = {}
    for name in [*(f'matched-bias-{f}.tex' for f in operation.FAMILIES), 'matched-bias-metrics.csv']:
        p=root/'bias'/name; p.write_text('values'); outputs[name]=operation.sha(p)
    inputs = {k:OLD for k in ('report_sha256','manifest_sha256','metadata_sha256','generator_sha256')}
    proof = dict(inputs,variants=40,conditions=42,outputs=outputs)
    (root/'bias/matched-bias-export.json').write_text(json.dumps(proof))
    return inputs


def test_complete_exports(tmp_path):
    inputs=fixture_exports(tmp_path)
    assert len(operation.verify_exports(tmp_path,inputs)['hashes']) == 9


@pytest.mark.parametrize('change', ['missing','extra','symlink','changed_csv','wrong_input','incomplete'])
def test_export_corruption_rejected(tmp_path,change):
    inputs=fixture_exports(tmp_path)
    if change == 'missing': (tmp_path/'primary/matched-kge-design.tex').unlink()
    if change == 'extra': (tmp_path/'extra.txt').write_text('extra')
    if change == 'symlink':
        p=tmp_path/'primary/matched-kge-design.tex'; p.unlink(); p.symlink_to('matched-kge-primary.tex')
    if change == 'changed_csv': (tmp_path/'bias/matched-bias-metrics.csv').write_text('different')
    if change == 'wrong_input': inputs['manifest_sha256']=NEW
    if change == 'incomplete':
        p=tmp_path/'bias/matched-bias-export.json'; proof=json.loads(p.read_text()); proof['variants']=39
        p.write_text(json.dumps(proof))
    with pytest.raises(ValueError): operation.verify_exports(tmp_path,inputs)


def test_two_bias_headers_are_bound_in_order():
    headers=[('% Report SHA256: ',OLD,NEW),('% Score manifest SHA256: ',NEW,OLD)]
    a=''.join(p+x+'\n' for p,x,y in headers)+'metrics\n'
    b=''.join(p+y+'\n' for p,x,y in headers)+'metrics\n'
    operation.compare_table_text(a,b,headers)
    with pytest.raises(ValueError): operation.compare_table_text(a,b,list(reversed(headers)))
