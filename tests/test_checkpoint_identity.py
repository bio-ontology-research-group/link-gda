import pytest
from link_gda.checkpoint_identity import check_or_write

def test_same_size_different_candidate_order_rejected(tmp_path):
    p=tmp_path/'model.pt'
    identity={'graph_sha256':'graph-a','candidates':['a','b'],'calibration_unit':'disease'}
    check_or_write(p,identity)
    check_or_write(p,identity,require_existing=True)
    with pytest.raises(ValueError,match='identity mismatch'):
        check_or_write(p,{**identity,'candidates':['b','a']},require_existing=True)
    with pytest.raises(ValueError,match='identity mismatch'):
        check_or_write(p,{**identity,'graph_sha256':'graph-b'},require_existing=True)
    with pytest.raises(FileNotFoundError):check_or_write(tmp_path/'other.pt',identity,require_existing=True)
