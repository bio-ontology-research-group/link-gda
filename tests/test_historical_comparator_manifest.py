import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location('historical_manifest', ROOT/'reproducibility/build_historical_comparator_manifest.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


def test_overlap_mask_uses_only_permitted_scores_and_background(tmp_path):
    genes=['g0','g1','g2']; (tmp_path/'candidates.txt').write_text('\n'.join(genes)+'\n')
    np.savez_compressed(tmp_path/'test-phenotypes.npz',scores=[[10.,1.,0.],[20.,3.,0.],[30.,5.,0.]],phenotypes=['p0','p1','p2'])
    profiles={'train':['p0','p1'],'all':['p0','p1','p2'],'d0':['p0'],'d1':['p0','p2'],'d2':['p1']}
    actual=m.masked_scores(tmp_path,genes,['d0','d1','d2'],[('g0','train'),('g2','all')],profiles)
    np.testing.assert_array_equal(actual,[[30.,1.,0.],[30.,3.,0.],[30.,3.,0.]])
    (tmp_path/'candidates.txt').write_text('g1\ng0\ng2\n')
    with pytest.raises(ValueError):m.masked_scores(tmp_path,genes,['d0'],[('g0','train')],profiles)


def fixture(root):
    genes=['g0','g1','g2']
    for fold in range(10):
        data=root/f'data/fold{fold}';(data/'raw').mkdir(parents=True)
        (data/'raw/train.txt').write_text('g0\tcauses_phenotype\tp0\n')
        (data/'candidates.txt').write_text('\n'.join(genes)+'\n')
        for split,rows in [('training',[('g0','train')]),('validation',[('g1','val')]),('test',[('g0','d0'),('g2','d1')])]:
            (data/(split+'.tsv')).write_text('Gene\tDisease\n'+''.join(f'{g}\t{d}\n' for g,d in rows))
        m.write(data/'profiles.json',{'train':['p0'],'val':['p1'],'d0':['p0'],'d1':['p1']})
        m.write(data/'preparation.json',dict(fold=fold,**{key+'_sha256':m.sha(data/name) for name,key in m.INPUTS.items()}))
        cache=root/f'phrank-inspired/ultra-cache/fold{fold}';cache.mkdir(parents=True)
        (cache/'candidates.txt').write_bytes((data/'candidates.txt').read_bytes())
        np.savez_compressed(cache/'test-phenotypes.npz',scores=[[3.,1.,2.],[1.,3.,2.]],phenotypes=['p0','p1'])
        files={p.name:m.sha(p) for p in cache.iterdir()}
        m.write(cache/'checksums.json',files)
        m.write(cache/'completion.json',dict(status='complete',fold=fold,files=files,
            preparation_sha256=m.sha(data/'preparation.json'),checksums_sha256=m.sha(cache/'checksums.json')))
        sources=m.source_files(root,root/'comparators/local-npz',fold)
        assert len(sources)==24
        for _,path in sources:
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,scores=[[3.,1.,2.],[1.,3.,2.]],genes=genes,diseases=['d0','d1'])


def test_all_historical_variants_and_deterministic_mask_export(tmp_path):
    fixture(tmp_path)
    output=tmp_path/'export/manifest.json'
    result=m.build(tmp_path,tmp_path/'comparators/local-npz',output,expected_candidates=3)
    assert len(result['models'])==25 and sum(len(v['folds']) for v in result['models'])==250
    assert result==m.build(tmp_path,tmp_path/'comparators/local-npz',output,expected_candidates=3)
    masked=next(v for v in result['models'] if v['name']=='ULTRA (Direct Overlap Masked)')
    for entry in masked['folds'].values():
        with np.load(output.parent/entry['path']) as z:np.testing.assert_array_equal(z['scores'],[[1.,1.,2.],[1.,3.,2.]])


@pytest.mark.parametrize('problem',['prepared-input','cache','axes','missing-fold','derived-score'])
def test_rejects_changed_inputs_cache_axes_or_partial_data(tmp_path,problem):
    fixture(tmp_path);output=tmp_path/'export/manifest.json'
    if problem=='prepared-input':(tmp_path/'data/fold0/training.tsv').write_text('Gene\tDisease\ng1\ttrain\n')
    elif problem=='cache':(tmp_path/'phrank-inspired/ultra-cache/fold0/test-phenotypes.npz').write_bytes(b'changed')
    elif problem=='missing-fold':(tmp_path/'comparators/local-npz/phive/fold_9.npz').unlink()
    elif problem=='axes':
        np.savez_compressed(tmp_path/'comparators/local-npz/phive/fold_0.npz',scores=np.zeros((2,3)),genes=['g0','g0','g2'],diseases=['d0','d1'])
    else:
        m.build(tmp_path,tmp_path/'comparators/local-npz',output,expected_candidates=3)
        np.savez_compressed(output.parent/'derived-overlap/fold0.npz',scores=np.zeros((2,3)),genes=['g0','g1','g2'],diseases=['d0','d1'])
    with pytest.raises((ValueError,FileNotFoundError)):
        m.build(tmp_path,tmp_path/'comparators/local-npz',output,expected_candidates=3)
