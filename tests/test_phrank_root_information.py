import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pytest
from phrank import Phrank

SOURCE = Path(__file__).resolve().parents[1]/'code/baselines/run_matched_phrank_phenorank.py'
spec = importlib.util.spec_from_file_location('phrank_refit_roots', SOURCE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_native_roots_and_singletons_match_author_information_content():
    closures = [{'a','root'}, {'b','root'}, {'singleton'}, set()]
    parents = defaultdict(list, {'a':['root'], 'b':['root']})
    terms = ['root','a','b','singleton','absent']; index = {t:i for i,t in enumerate(terms)}
    _, expected = Phrank.compute_information_content(
        {str(i):c for i,c in enumerate(closures) if c}, parents)
    actual = adapter.compute_phrank_mic(closures, terms, index, parents)
    np.testing.assert_allclose(actual, [expected.get(t,0.) for t in terms], rtol=1e-7, atol=1e-7)
    assert actual[index['root']] > 0 and actual[index['singleton']] > 0
    legacy = adapter.compute_phrank_mic(closures, terms, index, parents, root_policy='legacy-zero')
    np.testing.assert_array_equal(legacy, [0.,1.,1.,0.,0.])
    with pytest.raises(ValueError, match='root policy'):
        adapter.compute_phrank_mic(closures, terms, index, parents, root_policy='unknown')


def test_phrank_only_exports_native_and_explicit_legacy_modes(tmp_path, monkeypatch):
    data = tmp_path/'data/fold0'; (data/'raw').mkdir(parents=True)
    h='http://purl.obolibrary.org/obo/HP_'; m='http://purl.obolibrary.org/obo/MP_'
    (data/'candidates.txt').write_text('g1\ng2\ng3\n')
    (data/'profiles.json').write_text(json.dumps({'d1':[h+'a'],'d2':[h+'b'],
                                               'q1':[h+'a'],'q2':[h+'b']}))
    (data/'training.tsv').write_text('Gene\tDisease\ng1\td1\ng1\td2\n')
    (data/'test.tsv').write_text('Gene\tDisease\ng1\tq1\ng2\tq2\n')
    triples=[(h+'a','http://subclassof',h+'root'),(h+'b','http://subclassof',h+'root'),
             (m+'x','http://subclassof',m+'root'),(m+'y','http://subclassof',m+'root'),
             ('g1','has_phenotype',m+'x'),('g2','has_phenotype',m+'y'),('g3','has_phenotype',m+'y')]
    (data/'raw/train.txt').write_text(''.join('\t'.join(row)+'\n' for row in triples))
    monkeypatch.setattr(adapter,'DATA_DIR',data.parent)
    monkeypatch.setattr(adapter,'PHRANK_ONLY',True)
    # No PhenoRank resources exist: a Phrank-only invocation must not touch them.
    monkeypatch.setattr(adapter,'PHENORANK_RES_DIR',tmp_path/'missing-resources')
    for mode in ['native','legacy-zero']:
        monkeypatch.setattr(adapter,'OUT_ROOT',tmp_path/mode)
        monkeypatch.setattr(adapter,'PHRANK_ROOT_POLICY',mode)
        adapter.process_fold(0)
        out=tmp_path/mode/'fold0';complete=json.loads((out/'COMPLETE.json').read_text())
        assert complete['methods']==['phrank'] and complete['phrank_root_policy']==mode
        assert len(complete['outputs'])==10 and not list(out.glob('phenorank*'))
        for name,digest in complete['outputs'].items():
            assert adapter.sha256(out/name)==digest
            with np.load(out/name) as z:
                assert z['scores'].shape==(2,3) and np.isfinite(z['scores']).all()
                assert z['genes'].tolist()==['g1','g2','g3']
        with pytest.raises(FileExistsError):adapter.process_fold(0)
    for readout in ['direct','disease']:
        name=f'phrank_upheno_hm_{readout}_raw_scores.npz'
        with np.load(tmp_path/'native/fold0'/name) as native, np.load(tmp_path/'legacy-zero/fold0'/name) as legacy:
            np.testing.assert_allclose(native['scores'][:,0],math.log2(3),rtol=1e-7)
            np.testing.assert_array_equal(legacy['scores'],np.zeros((2,3)))
