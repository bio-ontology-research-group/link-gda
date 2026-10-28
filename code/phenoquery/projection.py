"""Streaming, provenance-retaining OWL expression projection; no lexical inference."""
import argparse,collections,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
O='{http://www.w3.org/2002/07/owl#}';R='{http://www.w3.org/1999/02/22-rdf-syntax-ns#}';S='{http://www.w3.org/2000/01/rdf-schema#}'
PH=('HP_','MP_','UPHENO_');CORE=PH+('GO_','UBERON_','PATO_','CL_','CHEBI_','BFO_','CARO_','MPATH_','BSPO_','OBA_')
def short(u):return u.rsplit('/',1)[-1]
def phenotype(u):return short(u).startswith(PH)
def allowed(u):return short(u).startswith(CORE)
def canon(x):return json.dumps(x,separators=(',',':'),ensure_ascii=True)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def expression(x):
 if x.tag in [O+'unionOf',O+'complementOf',O+'oneOf',O+'allValuesFrom']:raise ValueError('unsupported non-existential connective')
 u=x.get(R+'resource') or x.get(R+'about')
 if u:return ('atom',u)
 if x.tag==O+'Restriction':
  r=x.find(O+'onProperty');v=x.find(O+'someValuesFrom')
  if r is None or v is None or not r.get(R+'resource'):raise ValueError('unsupported restriction')
  return ('some',r.get(R+'resource'),expression(v))
 inter=x.find(O+'intersectionOf')
 if inter is not None:
  if inter.get(R+'parseType')!='Collection':raise ValueError('unsupported RDF list encoding')
  return ('and',tuple(sorted(set(expression(c) for c in inter),key=canon)))
 if len(x)==1:return expression(x[0])
 raise ValueError('unsupported class expression')
def atoms(e,path=()):
 if e[0]=='atom':yield e[1],path
 elif e[0]=='some':yield from atoms(e[2],path+(e[1],))
 else:
  for c in e[1]:yield from atoms(c,path)
def flat_entails(h,e):
 if e[0]=='atom':yield h,'http://subclassof',e[1]
 elif e[0]=='and':
  for c in e[1]:yield from flat_entails(h,c)
 elif e[2][0]=='atom':yield h,'urn:owl-exists:'+e[1],e[2][1]
def syntax(e,edges):
 if e[0]=='atom':return e[1]
 node='urn:eq-expression:'+hashlib.sha256(canon(e).encode()).hexdigest()
 if e[0]=='some':edges.add((node,'urn:eq-syntax:exists:'+e[1],syntax(e[2],edges)))
 else:
  for c in e[1]:edges.add((node,'urn:eq-syntax:intersection_member',syntax(c,edges)))
 return node

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'projection';out.mkdir(parents=True,exist_ok=True)
 edges={k:set() for k in ['backbone','entity','typed','structure']};counts=collections.Counter();coverage={k:set() for k in edges};sources={}
 with (out/'axioms.jsonl').open('w') as provenance:
  for file in sorted((a.root/'ontologies').glob('*.owl')):
   sources[file.name]=digest(file);depth=0
   for event,x in ET.iterparse(file,events=['start','end']):
    if event=='start':depth+=1;continue
    if depth==2:
     h=x.get(R+'about','')
     if x.tag==O+'Class' and allowed(h) and x.find(O+'deprecated') is None:
      for ax in x:
       if ax.tag not in [S+'subClassOf',O+'equivalentClass']:continue
       try:e=expression(ax)
       except ValueError as error:counts[str(error)]+=1;continue
       if any(not allowed(t) for t,path in atoms(e)):counts['outside_allowed_class_namespaces']+=1;continue
       aid=hashlib.sha256((h+ax.tag+canon(e)).encode()).hexdigest();counts['supported_axioms']+=1
       provenance.write(json.dumps({'id':aid,'source':file.name,'subject':h,'kind':ax.tag.split('}')[-1],'expression':e},separators=(',',':'))+'\n')
       if not phenotype(h):
        for edge in flat_entails(h,e):
         if not phenotype(edge[2]):edges['backbone'].add(edge)
        continue
       if e[0]=='atom':
        if phenotype(e[1]):edges['backbone'].add((h,'http://subclassof',e[1]))
        continue
       for t,path in atoms(e):
        if short(t).startswith(('GO_','UBERON_')) and path:
         edges['entity'].add((h,'urn:eq:entity',t));coverage['entity'].add(h)
         edges['typed'].add((h,'urn:eq:entity-via:'+short(path[-1]),t));coverage['typed'].add(h)
        elif short(t).startswith('PATO_'):
         kind='qualifier' if any(short(r) in ['RO_0002573'] for r in path) else 'quality'
         edges['typed'].add((h,'urn:eq:'+kind,t));coverage['typed'].add(h)
       node=syntax(e,edges['structure']);edges['structure'].add((h,'urn:eq-syntax:'+ax.tag.split('}')[-1],node));coverage['structure'].add(h)
     x.clear()
    depth-=1
   print('PARSED',file.name,dict(counts),flush=True)
 for k,es in edges.items():
  with (out/(k+'.tsv')).open('w') as f:
   for e in sorted(es):f.write('\t'.join(e)+'\n')
 report={'sources_sha256':sources,'counts':dict(counts),'edges':{k:len(v) for k,v in edges.items()},'coverage':{k:{prefix:sum(short(h).startswith(prefix) for h in hs) for prefix in PH} for k,hs in coverage.items()},'files':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}}
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
