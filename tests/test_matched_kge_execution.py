"""Real small-model test of both graphs, all KGE families and the global score gate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def test_all_matched_families_train_freeze_score_and_resume(tmp_path):
    data=tmp_path/'data/fold0';(data/'raw').mkdir(parents=True)
    genes=[f'g{i}' for i in range(80)]
    (data/'candidates.txt').write_text('\n'.join(genes)+'\n')
    triples=[[g,'causes_phenotype',f'p{i%2}'] for i,g in enumerate(genes)]
    triples += [[f'g{i}','associated_with',f'd{i}'] for i in [0,1]]
    triples += [[f'd{i}','has_symptom',f'p{i}'] for i in [0,1]]
    (data/'raw/train.txt').write_text(''.join('\t'.join(row)+'\n' for row in triples))
    (data/'profiles.json').write_text(json.dumps({f'd{i}':[f'p{i%2}'] for i in range(6)}))
    for split,indices in [('training',[0,1]),('validation',[2,3]),('test',[4,5])]:
        pd.DataFrame({'Gene':[f'g{i}' for i in indices],'Disease':[f'd{i}' for i in indices]}).to_csv(data/(split+'.tsv'),sep='\t',index=False)
    annotations=tmp_path/'gene_phenotypes.csv'
    pd.DataFrame({'Gene':genes,'Phenotype':[f'p{i%2}' for i in range(80)]}).to_csv(annotations,index=False)
    root=tmp_path/'run'
    base=[sys.executable,str(ROOT/'code/training/run_matched_kge.py'),'--root',str(root),
          '--data-root',str(tmp_path/'data'),'--fold','0','--device','cpu','--dimension','4',
          '--epochs','2','--profile-epochs','2','--validation-frequency','1',
          '--checkpoint-frequency','1','--batch-size','16','--sub-batch-size','8',
          '--expected-candidates','80','--gene-phenotypes',str(annotations)]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def invoke(extra,success=True):
        run=subprocess.run(base+extra,env=env,capture_output=True,text=True,timeout=120)
        if success:assert run.returncode==0,run.stdout+'\n'+run.stderr
        else:assert run.returncode!=0
        return run
    invoke(['--phase','freeze','--folds','0'],False)
    for graph in ['original','nobridge']:
        for model in ['U1','T1','INDIGENA','CONVKB','PPOOL']:
            extra=['--phase','train','--graph',graph,'--model',model]
            if model=='T1' and graph=='nobridge':
                invoke(extra+['--stop-after','1'])
                assert not (root/f'fold0/{graph}/{model}/TRAINED.json').exists()
            invoke(extra)
            assert not list(root.rglob('scores-*.npz'))
            invoke(['--phase','score','--graph',graph,'--model',model],False)
    invoke(['--phase','freeze','--folds','0'])
    frozen=json.loads((root/'FROZEN.json').read_text());assert len(frozen['training'])==10
    for graph in ['original','nobridge']:
        for model in ['U1','T1','INDIGENA','CONVKB','PPOOL']:
            invoke(['--phase','score','--graph',graph,'--model',model])
            out=root/f'fold0/{graph}/{model}'
            assert len(list(out.glob('scores-*.npz')))==4
            for path in out.glob('scores-*.npz'):
                with np.load(path) as z:
                    assert z['scores'].shape==(2,80) and z['train_scores'].shape==(2,80)
                    assert np.isfinite(z['scores']).all()
                    np.testing.assert_array_equal(z['genes'],genes)
    altered=root/'fold0/nobridge/T1/scores-raw_head.npz';altered.write_bytes(b'corrupted')
    result=invoke(['--phase','score','--graph','nobridge','--model','T1'],False)
    assert 'Completed scores changed' in result.stderr
