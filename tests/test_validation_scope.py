import pandas as pd
import pytest
from link_gda.validation import eligible_validation

def test_unseen_selection_uses_training_gene_status_and_keeps_original_frames():
    train=pd.DataFrame({'Gene':['g1'],'Disease':['t']})
    val=pd.DataFrame({'Gene':['g1','g2','g3','g4'],'Disease':['v1','v2','v3','v4']})
    out=eligible_validation(val,train,['g1','g2','g3'],'unseen-gene')
    assert out.Gene.tolist()==['g2','g3']
    assert len(val)==4 and len(train)==1
    with pytest.raises(ValueError,match='Fewer than two'):
        eligible_validation(val,train,['g1','g2'],'unseen-gene')
