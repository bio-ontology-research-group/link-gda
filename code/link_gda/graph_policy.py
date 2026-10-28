"""Validate independently projected ontology files before revised training."""
import json
from pathlib import Path
from .go_projection import sha256

def independent_edges(data_dir,ontology):
    root=Path(data_dir);path=root/f'{ontology}_edges.tsv'
    metadata=json.loads(Path(str(path)+'.metadata.json').read_text())
    expected={'policy':'independent ontology per fresh process; sorted unique triples',
              'source_sha256':sha256(root/f'{ontology}.owl'),'output_sha256':sha256(path),
              'projector':'OWL2VecStarProjector','bidirectional_taxonomy':True,
              'include_literals':False,'only_taxonomy':False}
    for key,value in expected.items():
        if metadata.get(key)!=value:raise ValueError(f'{ontology} projection metadata mismatch: {key}')
    return str(path)

def unique_triples(triples):
    return sorted(set(triples))
