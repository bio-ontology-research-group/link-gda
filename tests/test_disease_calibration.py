import numpy as np
import pytest
from link_gda.calibration import standardize, reference_statistics, standardize_fixed

S=np.array([[1.,5.,7.],[3.,3.,7.],[5.,1.,7.]])
D=['a','b','c']

def test_disease_loo_hand_computed_and_constant_gene():
    np.testing.assert_allclose(standardize(S,D),[[-3,3,0],[0,0,0],[3,-3,0]],atol=1e-10)

def test_adding_associations_for_same_disease_cannot_change_calibration():
    order=[2,0,0,1,0]
    result=standardize(S[order],[D[i] for i in order])
    np.testing.assert_allclose(result,standardize(S,D)[order])
    assert not np.allclose(result,standardize(S[order],[D[i] for i in order],unit='row'))

def test_fixed_reference_is_unique_disease_weighted_and_supports_single_query():
    ref=S[[0,0,1,2]]
    mean,sd=reference_statistics(ref,['a','a','b','c'])
    np.testing.assert_allclose(mean,[3,3,7])
    np.testing.assert_allclose(sd,[np.sqrt(8/3),np.sqrt(8/3),0])
    np.testing.assert_allclose(standardize_fixed([[3,3,7]],ref,['a','a','b','c']),0)

def test_inconsistent_repeated_disease_fails_loudly():
    with pytest.raises(ValueError,match='inconsistent'):
        standardize(S,['a','a','c'])

def test_single_unique_disease_is_not_silently_uncalibrated():
    with pytest.raises(ValueError,match='at least two'):
        standardize(S[[0,0]],['a','a'])

def test_legacy_row_mode_matches_archived_formula():
    from analysis.rq1_table import calibrate
    rows=S[[0,0,1,2]]
    np.testing.assert_allclose(standardize(rows,['a','a','b','c'],unit='row'),calibrate(rows))
def test_constant_and_large_offset_scores_are_stable():
    import numpy as np
    from link_gda.calibration import standardize
    scores=np.array([[1e8,1e8+1],[1e8,1e8+2],[1e8,1e8+3]],dtype=float)
    diseases=['a','b','c']
    actual=standardize(scores,diseases)
    np.testing.assert_array_equal(actual[:,0],0)
    np.testing.assert_allclose(actual[:,1],[-3,0,3],rtol=1e-11)

