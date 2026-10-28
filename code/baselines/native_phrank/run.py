"""Native HPO Phrank 1.6, training-GDA annotations, fixed full gene universe.

Uses the author implementation for IC, closures and parity checks. Cached
closures eliminate repeated ontology traversals; deterministic math.fsum over
sorted common terms stabilizes mathematically equal overlaps without changing
the native marginal-information-content score formula. No cross-species projection or normalization is added.
"""
import argparse,csv,hashlib,importlib.util,json,math,os,sys,time
from collections import defaultdict
from pathlib import Path
from graphlib import TopologicalSorter
import xml.etree.ElementTree as ET
import numpy as np

HPO='http://purl.obolibrary.org/obo/HP_'
OWL='{http://www.w3.org/2002/07/owl#}Class'
RDF='{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
SUB='{http://www.w3.org/2000/01/rdf-schema#}subClassOf'


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,p)
def rows(p):
 with Path(p).open() as f:return [(r['Gene'],r['Disease']) for r in csv.DictReader(f,delimiter='\t')]
def extract_hpo(source,destination):
 destination=Path(destination)
 if destination.exists():raise FileExistsError(destination)
 destination.mkdir(parents=True)
 nodes=set();edges=set()
 for _,element in ET.iterparse(source,events=('end',)):
  if element.tag==OWL:
   child=element.get(RDF+'about','')
   if child.startswith(HPO):
    nodes.add(child)
    for parent in element.findall(SUB):
     target=parent.get(RDF+'resource','')
     if target.startswith(HPO):nodes.add(target);edges.add((child,target))
   element.clear()
 parents={n:[] for n in nodes}
 for c,p in edges:parents[c].append(p)
 list(TopologicalSorter(parents).static_order())
 (destination/'hpo-dag.tsv').write_text(''.join(c+'\t'+p+'\n' for c,p in sorted(edges)))
 (destination/'hpo-nodes.txt').write_text('\n'.join(sorted(nodes))+'\n')
 info={'ontology_source':str(source),'ontology_sha256':sha(source),'node_count':len(nodes),'edge_count':len(edges),
       'extraction':'Only explicit named HPO-to-HPO rdfs:subClassOf axioms in retained upheno.owl; no inferred edges, restrictions, other predicates or cross-species mapping',
       'dag_acyclic':True,'dag_sha256':sha(destination/'hpo-dag.tsv'),'nodes_sha256':sha(destination/'hpo-nodes.txt')}
 write(destination/'ontology.json',info)
 return info


def training_annotations(training,validation,test,profiles,candidates):
 sets=[{d for _,d in x} for x in [training,validation,test]]
 if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):raise ValueError('Held-out disease overlap')
 annotations=defaultdict(set);sources=defaultdict(set)
 for gene,disease in training:
  if gene not in candidates:raise ValueError('Training gene outside candidate universe')
  for p in profiles[disease]:
   if not p.startswith(HPO):raise ValueError('Non-HPO query requires a separately declared adaptation')
   annotations[gene].add(p);sources[gene,p].add(disease)
 assert all(ds<=sets[0] and not ds&(sets[1]|sets[2]) for ds in sources.values())
 return annotations,sources


def cached_scores(author,profiles,diseases,genes):
 from phrank.utils import closure
 gene_closures={g:closure(author._gene_pheno_map.get(g,[]),author._child_to_parent) for g in genes}
 supported=[(i,gene_closures[g]) for i,g in enumerate(genes) if gene_closures[g]]
 result=np.zeros((len(diseases),len(genes)),dtype=np.float64)
 for i,disease in enumerate(diseases):
  query=closure(profiles[disease],author._child_to_parent)
  common_scores={}
  for j,annotation in supported:
   overlap=frozenset(query&annotation)
   if overlap not in common_scores:
    common_scores[overlap]=math.fsum(author._marginal_IC.get(term,0) for term in sorted(overlap))
   result[i,j]=common_scores[overlap]
 return result


def run_fold(data_root,ontology_root,author_root,output_root,fold,summary_path):
 started=time.time();sys.path.insert(0,str(author_root));from phrank import Phrank
 spec=importlib.util.spec_from_file_location('campaign_summarize',summary_path);summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
 data=Path(data_root)/f'fold{fold}';out=Path(output_root)/f'fold{fold}'
 if out.exists():raise FileExistsError('Refusing existing baseline output '+str(out))
 out.mkdir(parents=True)
 original=json.loads((data/'preparation.json').read_text());ontology_root=Path(ontology_root)
 for name,label in [('training.tsv','training'),('validation.tsv','validation'),('test.tsv','test'),('profiles.json','profiles'),('candidates.txt','candidates')]:
  if sha(data/name)!=original[label+'_sha256']:raise ValueError('Input changed '+name)
 genes=(data/'candidates.txt').read_text().splitlines();profiles=json.loads((data/'profiles.json').read_text())
 if len(genes)!=len(set(genes)):raise ValueError('Duplicate candidates')
 training,validation,test=[rows(data/(s+'.tsv')) for s in ['training','validation','test']]
 annotation,sources=training_annotations(training,validation,test,profiles,set(genes))
 nodes=set((ontology_root/'hpo-nodes.txt').read_text().splitlines())
 query_terms={p for _,d in training+validation+test for p in profiles[d]}
 unknown=query_terms-nodes
 # Phrank naturally treats terms absent from DAG as singleton nodes. Retain
 # those identifiers, explicitly recording them instead of losing query terms.
 annotations_path=out/'training-gene-hpo.tsv'
 annotations_path.write_text(''.join(p+'\t'+g+'\n' for g in sorted(annotation) for p in sorted(annotation[g])))
 provenance_path=out/'annotation-sources.tsv'
 provenance_path.write_text('Gene\tPhenotype\tTrainingDiseases\n'+''.join(g+'\t'+p+'\t'+'|'.join(sorted(ds))+'\n' for (g,p),ds in sorted(sources.items())))
 author=Phrank(str(ontology_root/'hpo-dag.tsv'),geneannotationsfile=str(annotations_path))
 diseases=sorted({d for _,d in test});scores=cached_scores(author,profiles,diseases,genes)
 if not np.isfinite(scores).all():raise ValueError('Nonfinite native Phrank score')
 unsupported=[i for i,g in enumerate(genes) if g not in annotation]
 assert np.all(scores[:,unsupported]==0)
 # Independently invoke the author implementation at diverse supported and
 # unsupported genes for actual held-out queries. No labels select parameters.
 rng=np.random.default_rng(0);query_indices=rng.choice(len(diseases),min(10,len(diseases)),replace=False)
 supported=[i for i,g in enumerate(genes) if g in annotation]
 gi=list(rng.choice(supported,min(20,len(supported)),replace=False))+unsupported[:5]
 parity=[]
 for qi in query_indices:
  for j in gi:
   expected=author.compute_phenotype_match(profiles[diseases[qi]],author._gene_pheno_map.get(genes[j],[]))
   if not np.isclose(scores[qi,j],expected,rtol=1e-12,atol=1e-12):raise AssertionError(('Native author score mismatch',fold,diseases[qi],genes[j],scores[qi,j],expected))
   parity.append((int(qi),int(j)))
 np.savez_compressed(out/'scores.npz',scores=scores,diseases=np.array(diseases),genes=np.array(genes))
 (out/'pairs.tsv').write_bytes((data/'test.tsv').read_bytes());(out/'candidates.txt').write_bytes((data/'candidates.txt').read_bytes())
 report={mode:summary.metrics(summary.rank_pairs(x,diseases,genes,test)) for mode,x in [('raw',scores),('calibrated',summary.calibrated(scores))]}
 write(out/'metrics.json',report)
 evidence={'method':'Phrank 1.6 native direct gene scoring','primary':'raw','calibrated':'descriptive posthoc unique-disease LOO sensitivity',
           'normalization':False,'adaptation':'None to score formula; whole-candidate task and fold-specific training-GDA annotations replace published exome setting; HPO-only, no mouse/cross-species arm',
           'fold':fold,'training_diseases':len({d for _,d in training}),'annotation_source_diseases':len(set().union(*sources.values())),
           'training_pairs':len(training),'candidate_count':len(genes),'annotated_candidates':len(annotation),'unsupported_candidates':len(unsupported),
           'unsupported_test_pairs':sum(g not in annotation for g,_ in test),'no_evidence_score':0,'tie_policy':'average ranks across every candidate',
           'ontology':json.loads((ontology_root/'ontology.json').read_text()),'unknown_hpo_singletons':sorted(unknown),
           'non_hpo_query_terms':sorted(p for p in query_terms if not p.startswith(HPO)),
           'all_test_queries_retained':True,'baseline_protocol_sha256':sha(Path(__file__).parent/'PROTOCOL.md'),
           'identifier_policy':'Frozen prepared OMIM/gene URIs only; no added alias database; original alias completeness unverified',
           'author_source_sha256':{str(p.relative_to(author_root)):sha(p) for p in sorted((Path(author_root)/'phrank').rglob('*.py'))},
           'runner_sha256':sha(__file__),'evaluator_sha256':sha(summary_path),'prepared_identity':original,
           'annotation_sha256':sha(annotations_path),'annotation_sources_sha256':sha(provenance_path),'author_score_parity_checks':len(parity),'parity_tolerance':{'rtol':1e-12,'atol':1e-12},'summation':'math.fsum on sorted common terms; stabilizes equal-overlap ties',
           'held_out_annotation_sources':0,'test_diseases':len(diseases),'test_pairs':len(test),'wall_seconds':time.time()-started}
 write(out/'provenance.json',evidence)
 write(out/'checksums.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
 summary.load_scores(out)  # Independent file reread, ties and metrics contract.
 write(out/'completion.json',{'phase':'phrank','fold':fold,'status':'complete','primary':'raw','checksums_sha256':sha(out/'checksums.json'),
                              'provenance_sha256':sha(out/'provenance.json'),'preparation_sha256':sha(data/'preparation.json'),
                              'runner_sha256':sha(__file__),'protocol_sha256':sha(Path(__file__).parent/'PROTOCOL.md'),'metrics':report,'wall_seconds':time.time()-started})
 print(json.dumps({'fold':fold,'metrics':report,'unsupported':len(unsupported),'parity_checks':len(parity)}),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 e=sub.add_parser('extract');e.add_argument('--ontology',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
 r=sub.add_parser('run');r.add_argument('--data-root',type=Path,required=True);r.add_argument('--ontology-root',type=Path,required=True);r.add_argument('--author-root',type=Path,required=True);r.add_argument('--output-root',type=Path,required=True);r.add_argument('--summary-path',type=Path,default=Path(__file__).resolve().parents[2]/'analysis/summarize_ultra_campaign.py');r.add_argument('--fold',type=int,required=True)
 a=p.parse_args()
 if a.command=='extract':print(json.dumps(extract_hpo(a.ontology,a.output)))
 else:run_fold(a.data_root,a.ontology_root,a.author_root,a.output_root,a.fold,a.summary_path)
