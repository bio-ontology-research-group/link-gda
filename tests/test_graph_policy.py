import json
import pytest
from link_gda.graph_policy import independent_edges, unique_triples
from link_gda.go_projection import sha256

def test_projection_hashes_prevent_stale_cache(tmp_path):
    owl=tmp_path/'go.owl';edge=tmp_path/'go_edges.tsv'
    owl.write_text('ontology');edge.write_text('a\tr\tb\n')
    metadata={'policy':'independent ontology per fresh process; sorted unique triples',
        'source_sha256':sha256(owl),'output_sha256':sha256(edge),'projector':'OWL2VecStarProjector',
        'bidirectional_taxonomy':True,'include_literals':False,'only_taxonomy':False}
    (tmp_path/'go_edges.tsv.metadata.json').write_text(json.dumps(metadata))
    assert independent_edges(tmp_path,'go')==str(edge)
    edge.write_text('a\tr\tc\n')
    with pytest.raises(ValueError,match='output_sha256'):independent_edges(tmp_path,'go')
    assert unique_triples([('b','r','c'),('a','r','b'),('a','r','b')])==[('a','r','b'),('b','r','c')]
