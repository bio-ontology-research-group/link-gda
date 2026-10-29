"""Prepared-annotation strata and exact represented-annotation fallback counts."""
import json
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rank_strata import annotation_bins
from revision_metrics import sha,summarize

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-root',type=Path,required=True)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
root=args.input_root.resolve()
workspace=args.workspace.resolve()
data=root/'corrected-inputs';pool=root/'expanded-phenotype-candidates.txt'
genes=pool.read_text().splitlines();lookup={g:i for i,g in enumerate(genes)}
frames={name:pd.read_csv(data/file,dtype=str).drop_duplicates() for name,file in
        [('phenotype','gene_phenotypes.csv'),('function','gene_functions.csv'),('expression','gene_site.csv')]}
bins={};cuts={}
for name,frame in frames.items():
    counts=frame.groupby('Gene').size()
    bins[name],cuts[name]=annotation_bins([counts.get(g,0) for g in genes])
represented={}
for name,ontology,column in [('function','go','Function'),('expression','uberon','Tissue')]:
    edges=pd.read_csv(root/'independent-graphs'/f'{ontology}_edges.tsv',sep='\t',header=None,dtype=str)
    terms=set(edges.iloc[:,0])|set(edges.iloc[:,-1])
    frame=frames[name]
    represented[name]=set(frame.loc[frame[column].isin(terms),'Gene'])
truth=pd.read_csv(data/'gene_diseases.csv',dtype=str)
fallback={}
for name,have in [('f',represented['function']),('s',represented['expression']),('fs',represented['function']|represented['expression'])]:
    missing=set(genes)-have
    fallback[name]={'candidate_count':len(genes),'fallback_candidates':len(missing),
                    'test_pairs':len(truth),'fallback_test_pairs':int(truth.Gene.isin(missing).sum())}
result={}
groups=['main-linkgda-pfs-calsel','main-convkb-pfs-calsel','main-indigena-pfs-calsel','main-ultra-pfs',
        'resnik_resnik_bma','resnik_lin_bma','resnik_simgic','phive']
for group in groups:
    result[group]=[]
    for fold in range(10):
        path=workspace/'revised-metrics'/group/f'fold_{fold}.json';obj=json.loads(path.read_text())
        assert obj['provenance']['candidates']['sha256']==sha(pool)
        keys=[(k['gene'],k['disease']) for k in obj['keys']]
        item={'fold':fold,'metrics_sha256':sha(path),'settings':{}}
        for setting,values in obj['settings'].items():
            ranks=np.asarray(values['ranks']);item['settings'][setting]={}
            for dimension,labels in bins.items():
                query_labels=[labels[lookup[g]] for g,_ in keys]
                item['settings'][setting][dimension]={}
                for label in sorted(set(query_labels)):
                    idx=[i for i,x in enumerate(query_labels) if x==label]
                    item['settings'][setting][dimension][label]=summarize(ranks[idx],[keys[i] for i in idx],len(genes))
        result[group].append(item)
out={'candidate_sha256':sha(pool),'cutpoints':cuts,'fallback':fallback,'strata':result,
     'input_sha256':{str(p.relative_to(root)):sha(p) for p in [*data.glob('gene_*.csv'),* (root/'independent-graphs').glob('*_edges.tsv')]},
     'interpretation':'Descriptive strata of full-pool ranks by prepared annotation counts; fallback uses represented graph annotations.'}
(workspace/'revised-metrics/annotation-audit.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(fallback))
