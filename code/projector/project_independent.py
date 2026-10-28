"""Project exactly one ontology in a fresh JVM process; retain unique sorted edges."""
import argparse
import json
import sys
from pathlib import Path
from importlib.metadata import version
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from link_gda.go_projection import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ontology',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import mowl
    mowl.init_jvm('10g')
    from mowl.datasets import PathDataset
    from mowl.projection import OWL2VecStarProjector
    edges=OWL2VecStarProjector(bidirectional_taxonomy=True).project(PathDataset(str(a.ontology)).ontology)
    triples=sorted({(e.src,e.rel,e.dst) for e in edges})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:
        for triple in triples:f.write('\t'.join(triple)+'\n')
    metadata={'policy':'independent ontology per fresh process; sorted unique triples',
              'ontology':str(a.ontology),'source_sha256':sha256(a.ontology),
              'output_sha256':sha256(a.output),'projector':'OWL2VecStarProjector',
              'bidirectional_taxonomy':True,'include_literals':False,'only_taxonomy':False,
              'mowl_version':version('mowl-borg'),'projected_rows':len(edges),'unique_edges':len(triples)}
    Path(str(a.output)+'.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata))
if __name__=='__main__':main()
