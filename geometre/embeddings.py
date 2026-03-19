import torch as th
from geometre.box import Box
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


empty_tensor = th.tensor([])

def get_box_data(box_data, index_tensor):
    center_embed, offset_embed = box_data
    center = center_embed(index_tensor)
    offset = th.abs(offset_embed(index_tensor))
    return center, offset

def get_role_data(role_data, transitive_ids, inverse_ids, transitive, index_tensor):
    # transitive_id_to_dimension = {t_id.item(): i for i, t_id in enumerate(transitive_ids)}

    transf_cen_mul, transf_cen_add, transf_off_mul, transf_off_add = role_data

    transitive_mask = th.isin(index_tensor, transitive_ids)
    projection_dims = index_tensor[transitive_mask]
    inverse_ids_mask = th.isin(projection_dims, inverse_ids)
    projection_dims[inverse_ids_mask] = projection_dims[inverse_ids_mask] - 1

    inverse_mask = th.isin(index_tensor, inverse_ids)
    trans_inv = transitive_mask & inverse_mask
    trans_not_inv = transitive_mask & ~inverse_mask

    cen_mul = transf_cen_mul(index_tensor)
    cen_add = transf_cen_add(index_tensor)
    off_mul = transf_off_mul(index_tensor)
    off_add = transf_off_add(index_tensor)

    if transitive:
        bs_ids = th.nonzero(transitive_mask).squeeze()
        cen_mul[bs_ids, projection_dims] = 1.0  # identity: no scaling in transitive dim
        cen_add[bs_ids, projection_dims] = 0.0  # identity: no shift in transitive dim
        off_mul[bs_ids, projection_dims] = 1.0
        off_add[bs_ids, projection_dims] = 0.0

    return (cen_mul, cen_add, off_mul, off_add), (trans_inv, trans_not_inv, projection_dims)

def embedding_sub(data, box_data):
    assert data.shape[1] == 1, "Sub queries should have 1 component"
    c, c_offset = get_box_data(box_data, data[:, 0])

    false_tensor = th.zeros(c.shape[0]).bool().to(c.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c.device)
    negative_box = None
    
    return Box(c, c_offset), *transitive_data, None

def embedding_test_query(symptom_ids, has_symptom_id, associated_with_id, box_data, role_data, transitive_ids, inverse_ids, transitive):
    c, c_offset = get_box_data(box_data, symptom_ids)
    has_symptom_id = th.tensor([has_symptom_id], device=symptom_ids.device)
    associated_with_id = th.tensor([associated_with_id], device=symptom_ids.device)
    
    role_data = role_data[0]
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, has_symptom_id)
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, associated_with_id)
    
    projected_boxes = Box(c, c_offset).transform(*transf_data_1)
    assert len(projected_boxes) == len(symptom_ids), f"Projected boxes should have the same batch size as symptom_ids. Got {len(projected_boxes)} and {len(symptom_ids)}"
    # boxes = []
    # for i in range(len(projected_boxes)):
        # box = projected_boxes.slice(i)
        # boxes.append(box)
    # intersection = Box.intersection(*boxes)
    # projection = intersection.transform(*transf_data_2)
    projection = projected_boxes.transform(*transf_data_2)
    # assert len(projection) == len(symptom_ids), f"Projection should have the same batch size as symptom_ids. Got {len(projection)} and {len(symptom_ids)}"
    return projection
    
def embedding_1p(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    "P(Anchor)"
    "r,e"
    role_data = role_data[0]
    transf_data, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 0])
    c, c_offset = get_box_data(box_data, data[:, 1])
    role_data = role_data[0]
    
    negative_box = None
    return Box(c, c_offset).transform(*transf_data), *transitive_data, negative_box



def embedding_2p(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    "P(P(Anchor))"
    "r2,r1,e  ->  Box(e).transform(r1).transform(r2)"
    role_data = role_data[0]
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    transf_data_2, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 0])
    c, c_offset = get_box_data(box_data, data[:, 2])

    negative_box = None
    return Box(c, c_offset).transform(*transf_data_1).transform(*transf_data_2), *transitive_data, negative_box


def embedding_pi(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    "I(Anchor,P(Anchor))"
    "e,r,e"
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    role_data = role_data[0]
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    c_2, c_2_offset = get_box_data(box_data, data[:, 2])
    
    box_c_1 = Box(c_1, c_1_offset)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_1)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    negative_box = None
    return Box.intersection(box_c_1, box_c_2), *transitive_data, negative_box

def embedding_ppi(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    "I(Anchor,P(Anchor),P(Anchor))"
    "e,r,e,r,e"

    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    role_data = role_data[0]

    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    c_2, c_2_offset = get_box_data(box_data, data[:, 2])

    transf_data_3, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 3])
    c_3, c_3_offset = get_box_data(box_data, data[:, 4])

    box_c_1 = Box(c_1, c_1_offset)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2)
    box_c_3 = Box(c_3, c_3_offset).transform(*transf_data_3)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)

    negative_box = None
    return Box.intersection(box_c_1, box_c_2, box_c_3), *transitive_data, negative_box

def embedding_pip(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    "P(I(Anchor,P(Anchor)))"
    "r,e,r,e"

    role_data = role_data[0]
    transf_data_0, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 0])
    c_1, c_1_offset = get_box_data(box_data, data[:, 1])
    
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 2])
    c_2, c_2_offset = get_box_data(box_data, data[:, 3])

    
    box_c_1 = Box(c_1, c_1_offset)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2)
    
    intersection_box = Box.intersection(box_c_1, box_c_2)
    projection_box = intersection_box.transform(*transf_data_0)
    
    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)

    negative_box = None

    
    
    return projection_box, *transitive_data, negative_box


def embedding_ppip(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    "P(I(Anchor,P(Anchor),P(Anchor)))"
    "r,e,r,e,r,e"

    role_data = role_data[0]
    transf_data_0, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 0])
    c_1, c_1_offset = get_box_data(box_data, data[:, 1])
    
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 2])
    c_2, c_2_offset = get_box_data(box_data, data[:, 3])

    transf_data_3, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 4])
    c_3, c_3_offset = get_box_data(box_data, data[:, 5])

    
    
    box_c_1 = Box(c_1, c_1_offset)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2)
    box_c_3 = Box(c_3, c_3_offset).transform(*transf_data_3)

    intersection_box = Box.intersection(box_c_1, box_c_2, box_c_3)
    projection_box = intersection_box.transform(*transf_data_0)
    
    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)

    negative_box = None

    
    
    return projection_box, *transitive_data, negative_box



def embedding_ki(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    """I(P(Anchor_1), ..., P(Anchor_k)) — k-way intersection of 1p queries.
    data layout: [r, e_1, r, e_2, ..., r, e_k]  (shape: batch_size x 2k)
    """
    k = data.shape[1] // 2
    role_data = role_data[0]

    boxes = []
    for i in range(k):
        transf_data, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 2 * i])
        c, c_offset = get_box_data(box_data, data[:, 2 * i + 1])
        boxes.append(Box(c, c_offset).transform(*transf_data))

    result = Box.intersection(*boxes)

    false_tensor = th.zeros(data.shape[0]).bool().to(data.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(data.device)
    return result, *transitive_data, None


def embedding_kip(data, box_data, role_data, transitive_ids, inverse_ids, transitive, intersection_net=None):
    """P(I_neural(Anchor_1, ..., Anchor_k), r) — intersect k anchors (padded), then project.

    Padded data layout: [k_actual, e_1, ..., e_maxk, r]  (shape: batch x (max_k + 2))
      - data[:, 0]         : k_actual — real number of anchors per sample
      - data[:, 1:1+max_k] : anchor entity IDs, padded with 0 for positions >= k_actual
      - data[:, -1]        : relation ID

    The padding mask (True = ignore) is derived from k_actual and passed to the Set
    Transformer so padded slots are invisible to attention and to min/mean offset.
    """
    k_actual = data[:, 0]               # (batch,)
    max_k    = data.shape[1] - 2        # 1 slot for k_actual, max_k slots, 1 for r
    role_data = role_data[0]

    # Batch-lookup all anchor embeddings at once — (batch, max_k, dim)
    c, c_offset = get_box_data(box_data, data[:, 1:1+max_k])
    boxes = [Box(c[:, i, :], c_offset[:, i, :]) for i in range(max_k)]

    # True = padding slot (should be ignored by attention)
    pad_mask = th.arange(max_k, device=data.device).unsqueeze(0) >= k_actual.unsqueeze(1)

    if intersection_net is not None:
        result = Box.neural_intersection(boxes, intersection_net, padding_mask=pad_mask)
    else:
        result = Box.intersection(*boxes)

    transf_r, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, -1])
    result = result.transform(*transf_r)

    false_tensor = th.zeros(data.shape[0]).bool().to(data.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(data.device)
    return result, *transitive_data, None


def embedding_ki2p(data, box_data, role_data, transitive_ids, inverse_ids, transitive, intersection_net=None):
    """I_neural(Anchor_1,...,Anchor_k) → r1 → r2 with padded fixed-size input.

    Padded data layout: [k_actual, e_1, ..., e_maxk, r1, r2]  (shape: batch x (max_k + 3))
      - data[:, 0]         : k_actual — real number of anchors
      - data[:, 1:1+max_k] : anchor entity IDs, padded with 0 for positions >= k_actual
      - data[:, -2]        : r1 (has_phenotype)
      - data[:, -1]        : r2 (associated_with)

    The r1 intermediate is detached before r2 so ki2p training only updates r2
    (associated_with). The intersection net + r1 are trained exclusively by kipd,
    avoiding gradient conflicts. During eval (torch.no_grad()), detach is a no-op.
    """
    k_actual = data[:, 0]               # (batch,)
    max_k    = data.shape[1] - 3        # 1 for k_actual, max_k slots, 2 for r1/r2
    role_data = role_data[0]

    c, c_offset = get_box_data(box_data, data[:, 1:1+max_k])  # (batch, max_k, dim)
    boxes = [Box(c[:, i, :], c_offset[:, i, :]) for i in range(max_k)]

    pad_mask = th.arange(max_k, device=data.device).unsqueeze(0) >= k_actual.unsqueeze(1)

    if intersection_net is not None:
        result = Box.neural_intersection(boxes, intersection_net, padding_mask=pad_mask)
    else:
        raise NotImplementedError("ki2p requires a SetTransformerIntersection net")

    transf_r1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, -2])
    transf_r2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, -1])
    after_r1 = result.transform(*transf_r1)
    result = after_r1.transform(*transf_r2)

    false_tensor = th.zeros(data.shape[0]).bool().to(data.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(data.device)
    return result, *transitive_data, None


def embedding_humanoid(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    "P(I(Anchor,P(I(Anchor,P(Anchor))),P(Anchor)))"
    "r,e,r,e,r,e,r,e"

    role_data = role_data[0]
    transf_data_0, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 0])
    c_1, c_1_offset = get_box_data(box_data, data[:, 1])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 2])
    c_2, c_2_offset = get_box_data(box_data, data[:, 3])
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 4])
    c_3, c_3_offset = get_box_data(box_data, data[:, 5])
    transf_data_3, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 6])
    c_4, c_4_offset = get_box_data(box_data, data[:, 7])

    box_c_1 = Box(c_1, c_1_offset)
    box_c_2 = Box(c_2, c_2_offset)
    box_c_3 = Box(c_3, c_3_offset).transform(*transf_data_2)
    box_c_4 = Box(c_4, c_4_offset).transform(*transf_data_3)

    intersection_1 = Box.intersection(box_c_2, box_c_3).transform(*transf_data_1)
    intersection_2 = Box.intersection(box_c_1, intersection_1, box_c_4).transform(*transf_data_0)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    negative_box = None
    return intersection_2, *transitive_data, negative_box
    

    
    




