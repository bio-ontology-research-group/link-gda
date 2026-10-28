import pytest
import torch
from link_gda.filtered_negative_sampling import GenePoolNegativeSampler


def sampler(n=32):
    # Relation 2 is forward; 3 is inverse. Both candidates 0,1 are true for symptom4.
    triples = torch.tensor([[0,2,4],[1,2,4],[4,3,0],[4,3,1],[2,0,5]])
    return GenePoolNegativeSampler(mapped_triples=triples, num_entities=6, num_relations=4,
                                   target_relations=[2], gene_pool=[0,1,2,3], num_negs_per_pos=n)


@pytest.mark.parametrize('n', [1,32])
def test_rejects_all_known_forward_inverse_positives(n):
    torch.manual_seed(5)
    positives = torch.tensor([[0,2,4],[4,3,1]]).repeat(100,1)
    negatives = sampler(n).corrupt_batch(positives)
    assert negatives.shape == (200,n,3)
    assert torch.isin(negatives[::2,:,0], torch.tensor([2,3])).all()
    assert torch.isin(negatives[1::2,:,2], torch.tensor([2,3])).all()
    assert torch.equal(negatives[:,:,1], positives[:,1,None].expand(-1,n))
    assert (negatives[::2,:,2] == 4).all()
    assert (negatives[1::2,:,0] == 4).all()


def test_original_rejected_even_when_not_in_known_training_set():
    negatives = sampler().corrupt_batch(torch.tensor([[2,2,4],[4,3,2]]))
    assert (negatives[0,:,0] == 3).all()
    assert (negatives[1,:,2] == 3).all()


def test_inverse_only_known_positive_is_filtered_in_forward_direction():
    s = GenePoolNegativeSampler(mapped_triples=torch.tensor([[4,3,0],[4,3,1]]),
        num_entities=6,num_relations=4,target_relations=[2],gene_pool=[0,1,2],num_negs_per_pos=32)
    assert (s.corrupt_batch(torch.tensor([[0,2,4]]))[:,:,0] == 2).all()


def test_non_target_matches_basic_sampler_exactly():
    from pykeen.sampling import BasicNegativeSampler
    s = sampler()
    p = torch.tensor([[2,0,5],[0,1,4],[3,0,1]])
    b = BasicNegativeSampler(mapped_triples=p,num_entities=6,num_relations=4,num_negs_per_pos=32)
    torch.manual_seed(10); expected = b.corrupt_batch(p)
    torch.manual_seed(10); observed = s.corrupt_batch(p)
    assert torch.equal(observed,expected)


def test_saturated_context_fails_before_training():
    with pytest.raises(ValueError,match='no eligible'):
        GenePoolNegativeSampler(mapped_triples=torch.tensor([[0,2,4],[1,2,4]]),
            num_entities=5,num_relations=4,target_relations=[2],gene_pool=[0,1])
