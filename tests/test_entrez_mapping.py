import pytest
from link_gda.identifiers import entrez_ids,gene_iris

def test_multi_mapping_expands_individual_gene_identifiers():
    assert entrez_ids('100008586; 2579; 26748')==['2579','26748','100008586']
    assert gene_iris('http://mowl.borg/3303; 3304')==['http://mowl.borg/3303','http://mowl.borg/3304']

def test_single_identifier_and_duplicate_mapping():
    assert entrez_ids('123')==['123']
    assert entrez_ids('123; 123')==['123']

@pytest.mark.parametrize('value',['123;','nan','1.0','0','gene123','http://other/123'])
def test_invalid_mapping_rejected(value):
    with pytest.raises(ValueError):entrez_ids(value)
