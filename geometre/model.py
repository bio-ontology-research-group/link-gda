import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from tqdm import tqdm
from geometre import embeddings as E
from geometre.box import Box
from scipy.stats import rankdata

# argsort = torch.argsort(negative_logit, dim=1, descending=True)
def compute_ranks_with_min_tie_breaking(negative_logit):
    assert len(negative_logit.shape) == 2  # (batch_size, num_entities)
    ranks = []
    for i in range(negative_logit.shape[0]):
        # Convert to numpy, compute ranks, convert back
        batch_ranks = rankdata(negative_logit[i].cpu().numpy(), method='min') - 1  # -1 to make 0-indexed
        ranks.append(torch.tensor(batch_ranks, device=negative_logit.device))
    return torch.stack(ranks)

def Identity(x):
    return x

class MAB(nn.Module):
    """Multihead Attention Block (Set Transformer, Lee et al. 2019)."""
    def __init__(self, dim, num_heads, ff_mult=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, X, Y, key_padding_mask=None):
        H = self.norm1(X + self.attn(X, Y, Y, key_padding_mask=key_padding_mask)[0])
        return self.norm2(H + self.ff(H))


class SAB(nn.Module):
    """Self-Attention Block — MAB(X, X)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mab = MAB(dim, num_heads)

    def forward(self, X, padding_mask=None):
        return self.mab(X, X, key_padding_mask=padding_mask)


class PMA(nn.Module):
    """Pooling by Multihead Attention: aggregates k elements into num_seeds via learned seed queries."""
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, num_heads)

    def forward(self, X, padding_mask=None):
        S = self.S.expand(X.shape[0], -1, -1)       # (batch, num_seeds, dim)
        return self.mab(S, X, key_padding_mask=padding_mask)  # (batch, num_seeds, dim)


class SetTransformerIntersection(nn.Module):
    """Set Transformer intersection operator with optional padding-mask support.

    Handles variable k via a fixed padded tensor + boolean mask:
    - SAB: each real box attends to all other real boxes (padding masked out).
    - PMA: learned seed aggregates real boxes → 1 vector (padding masked out).
    - min/mean offset computed only over non-padded positions.

    Input:  centers (batch, k, dim), offsets (batch, k, dim)
            padding_mask (batch, k) bool — True = padding slot, ignored in attention
    Output: Box with center (batch, dim) and offset (batch, dim)
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        in_dim = dim * 2  # concat(center, offset)
        self.input_proj = nn.Linear(in_dim, in_dim)
        self.sab = SAB(in_dim, num_heads)
        self.pma = PMA(in_dim, num_heads, num_seeds=1)
        self.center_head = nn.Linear(in_dim, dim)
        self.offset_gate = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, centers, offsets, padding_mask=None):
        # centers, offsets: (batch, k, dim)
        x = torch.cat([centers, offsets], dim=-1)              # (batch, k, 2*dim)
        x = self.input_proj(x)
        x = self.sab(x, padding_mask=padding_mask)             # (batch, k, 2*dim)
        z = self.pma(x, padding_mask=padding_mask).squeeze(1)  # (batch, 2*dim)

        new_center = self.center_head(z)   # (batch, dim)
        gate = self.offset_gate(z)         # (batch, dim) in (0, 1)

        if padding_mask is not None:
            # Mask padding slots: use inf for min, 0 for mean
            m = padding_mask.unsqueeze(-1).expand_as(offsets)  # (batch, k, dim)
            counts = (~padding_mask).float().sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
            min_offset  = offsets.masked_fill(m, float('inf')).min(dim=1).values
            mean_offset = offsets.masked_fill(m, 0.0).sum(dim=1) / counts
        else:
            min_offset  = offsets.min(dim=1).values
            mean_offset = offsets.mean(dim=1)

        # new_offset = gate * min_offset + (1 - gate) * mean_offset
        new_offset = gate * mean_offset
        return new_center, new_offset


class GeometrE(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, alpha,
                 test_batch_size=1, query_name_dict=None, transitive_ids=None,
                 inverse_ids=None, with_answer_embedding=False, device="cpu",
                 gene_entity_ids=None, dispersion_weight=0.0, dispersion_sample_size=512,
                 max_k=2):
        super(GeometrE, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.device = device
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).to(device) # used in test_step
        self.query_name_dict = query_name_dict
        # max_k is the padded width used for all kip/ki2p intersection queries
        self.register_buffer('max_k', torch.tensor(max_k, dtype=torch.long))

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        if transitive_ids is None:
            transitive_ids = torch.LongTensor([])
            
        self.transitive_ids = nn.Parameter(torch.LongTensor(transitive_ids), requires_grad=False)

        if inverse_ids is None:
            inverse_ids = torch.LongTensor([])
        self.inverse_ids = nn.Parameter(torch.LongTensor(inverse_ids), requires_grad=False) if inverse_ids is not None else None
        
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.alpha = nn.Parameter(
            torch.Tensor([alpha]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )

        
        
        self.center_embedding = self.init_embedding(nentity, self.entity_dim)
        self.offset_embedding = self.init_embedding(nentity, self.entity_dim)

        self.with_answer_embedding = with_answer_embedding

        if self.with_answer_embedding:
            self.answer_embedding = self.init_embedding(nentity, self.entity_dim)

        # Affine transform: new_center = center * cen_mul + cen_add
        self.relation_cen_mul = self.init_embedding(nrelation, self.relation_dim)
        self.relation_cen_add = self.init_embedding(nrelation, self.relation_dim)
        self.offset_mul = self.init_embedding(nrelation, self.relation_dim)
        self.offset_add = self.init_embedding(nrelation, self.relation_dim)

        self.relation_neg_cen_mul = self.init_embedding(nrelation, self.relation_dim)
        self.relation_neg_cen_add = self.init_embedding(nrelation, self.relation_dim)
        self.offset_neg_mul = self.init_embedding(nrelation, self.relation_dim)
        self.offset_neg_add = self.init_embedding(nrelation, self.relation_dim)

        # Gene dispersion regularisation
        self.dispersion_weight = dispersion_weight
        self.dispersion_sample_size = dispersion_sample_size
        if gene_entity_ids is not None:
            self.register_buffer('gene_entity_ids', torch.tensor(gene_entity_ids, dtype=torch.long))
        else:
            self.gene_entity_ids = None

        # Set Transformer intersection operator (robust to variable k)
        self.intersection_net = SetTransformerIntersection(self.entity_dim)


    def init_embedding(self, num_embeddings, dimension):
        embedding = nn.Embedding(num_embeddings, dimension)
        nn.init.uniform_(
            tensor=embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        return embedding
    
    def get_box_data(self):
        return self.center_embedding, self.offset_embedding

    def get_role_data(self):
        positive_data = self.relation_cen_mul, self.relation_cen_add, self.offset_mul, self.offset_add
        negative_data = self.relation_neg_cen_mul, self.relation_neg_cen_add, self.offset_neg_mul, self.offset_neg_add
        return positive_data, negative_data

    def get_intersection_net(self):
        return self.intersection_net


    def embedding_sub(self, data, transitive):
        return E.embedding_sub(data, self.get_box_data())

    def embedding_ppi(self, data, transitive):
        return E.embedding_ppi(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_pip(self, data, transitive):
        return E.embedding_pip(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_ppip(self, data, transitive):
        return E.embedding_ppip(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    
    def embedding_ki(self, data, transitive):
        return E.embedding_ki(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_kip(self, data, transitive):
        return E.embedding_kip(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, intersection_net=self.get_intersection_net())

    def embedding_ki2p(self, data, transitive):
        return E.embedding_ki2p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, intersection_net=self.get_intersection_net())

    def embedding_humanoid(self, data, transitive):
        return E.embedding_humanoid(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)
    
    def embedding_1p(self, data, transitive):
        return E.embedding_1p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_2p(self, data, transitive):
        return E.embedding_2p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_pi(self, data, transitive):
        return E.embedding_pi(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def construct_test_query(self, symptom_ids, has_symptom_id, associated_with_id):
        return E.embedding_test_query(symptom_ids, has_symptom_id, associated_with_id, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, False)
    
    def get_embedding_fn(self, task_name):
        """
        This chooses the corresponding embedding fuction given the name of the task.
        """

        if task_name == "ki":
            return self.embedding_ki
        if task_name == "kip":
            return self.embedding_kip
        if task_name == "ki2p":
            return self.embedding_ki2p
        return {
            "sub": self.embedding_sub,
            "1p": self.embedding_1p,
            "2p": self.embedding_2p,
            "pi": self.embedding_pi,
            "ppi": self.embedding_ppi,
            "pip": self.embedding_pip,
            "ppip": self.embedding_ppip,
            "humanoid": self.embedding_humanoid,
        }[task_name]
    
    
    def embed_query_box(self, queries, query_type, transitive):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        
        embedding_fn = self.get_embedding_fn(query_type)
        return embedding_fn(queries, transitive)

    def cal_membership_logit(self, entity_embedding, box_embedding):
        return Box.box_inclusion_score(box_embedding, entity_embedding, self.alpha)

    def cal_transitive_relation_logit(self, transitive_ids, inverse_ids):
        # For transitive relations the identity transform means cen_mul=1, cen_add=0,
        # off_mul near 1, off_add near 0.
        cen_mul = self.relation_cen_mul(transitive_ids)
        cen_add = self.relation_cen_add(transitive_ids)
        off_mul = self.offset_mul(transitive_ids)
        off_add = self.offset_add(transitive_ids)

        cen_mul_loss = torch.linalg.norm(cen_mul - 1, ord=1)
        cen_add_loss = torch.linalg.norm(cen_add, ord=1)
        off_mul_loss = torch.linalg.norm(off_mul - 1, ord=1) + torch.linalg.norm(off_mul - 1, dim=-1, ord=1)
        off_add_loss = torch.linalg.norm(off_add, ord=1)

        loss = cen_mul_loss + cen_add_loss + off_mul_loss + off_add_loss
        return loss


    def cal_logit_box(self, entity_embedding, box_embedding, trans_inv, trans_not_inv, projection_dims, transitive=False, negative=False, negative_box=None, negation_indices=None, verbose=False):
        if transitive:
            logit = Box.box_composed_score_with_projection(box_embedding, entity_embedding, self.alpha, trans_inv, trans_not_inv, projection_dims, negative=negative, transitive=transitive)
        else:
            logit = Box.box_inclusion_score(box_embedding, entity_embedding, self.alpha, negative=negative, verbose=verbose)

        # Add exclusion score for queries with negative boxes
        if negative_box is not None and negation_indices is not None:
            # Extract entity embeddings and boxes for negation queries only
            negation_entity_embedding = Box(entity_embedding.center[negation_indices], entity_embedding.offset[negation_indices])
            exclusion_logit = Box.box_exclusion_score(negative_box, negation_entity_embedding, self.alpha, negative=negative)
            # Add exclusion scores back to the full logit tensor at the correct indices
            logit[negation_indices] = logit[negation_indices] + exclusion_logit

        return self.gamma - logit

    def score_genes(self, query_embedding, eval_genes):
        """
        Query Embeddings (num_symptoms)
        """
        if self.with_answer_embedding:
            gene_center_embedding = self.answer_embedding(eval_genes).unsqueeze(1)
            gene_boxes = Box(gene_center_embedding, as_point=True) # (num_genes)
        else:
            gene_center_embedding = self.center_embedding(eval_genes).unsqueeze(1)
            gene_offset_embedding = self.offset_embedding(eval_genes).unsqueeze(1)
            gene_boxes = Box(gene_center_embedding, offset=gene_offset_embedding)
        
        scores = self.cal_logit_box(gene_boxes, query_embedding, None, None, None, transitive=False, negative=False, negative_box=None, negation_indices=None, verbose=False)

        
        # assert scores.shape == (len(eval_genes), len(query_embedding)), f"Error in score shape: expected {(len(eval_genes), len(query_embedding))}, got {scores.shape}"

        # scores = torch.min(scores, dim=1).values
        scores = torch.mean(scores, dim=1)

        
        return scores
        
    
    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=False):
        all_boxes, all_idxs, all_trans_masks, all_inv_masks, all_projection_dims, all_negative_boxes = [], [], [], [], [], []

        # Track the cumulative count of queries to map negative boxes to correct indices
        query_count = 0
        negation_indices = []  # Indices of queries with negation
        negation_boxes_list = []  # Corresponding negative boxes

        for query_structure in batch_queries_dict:
            query_type = self.query_name_dict[query_structure]
            batch_size = len(batch_queries_dict[query_structure])

            boxes, inv_mask, trans_mask, projection_dims, negative_box = self.embed_query_box(batch_queries_dict[query_structure], query_type, transitive)
            all_boxes.append(boxes)
            all_idxs.extend(batch_idxs_dict[query_structure])
            all_trans_masks.append(trans_mask)
            all_inv_masks.append(inv_mask)
            all_projection_dims.append(projection_dims)

            # Track negation queries
            if negative_box is not None:
                for i in range(len(boxes)):
                    negation_indices.append(query_count + i)
                negation_boxes_list.append(negative_box)

            query_count += len(boxes)

        if len(all_boxes) > 0:
            all_boxes = Box.cat(all_boxes, dim=0)
            all_boxes.center = all_boxes.center.unsqueeze(1)
            all_boxes.offset = all_boxes.offset.unsqueeze(1)
            all_trans_masks = torch.cat(all_trans_masks, dim=0)
            all_inv_masks = torch.cat(all_inv_masks, dim=0)
            all_projection_dims = torch.cat(all_projection_dims, dim=0).long()

            # Concatenate negative boxes
            if len(negation_boxes_list) > 0:
                all_negative_boxes = Box.cat(negation_boxes_list, dim=0)
                all_negative_boxes.center = all_negative_boxes.center.unsqueeze(1)
                all_negative_boxes.offset = all_negative_boxes.offset.unsqueeze(1)
                negation_indices = torch.tensor(negation_indices, device=all_boxes.center.device)
            else:
                all_negative_boxes = None
                negation_indices = None
        
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs]

        if type(positive_sample) != type(None):
            if len(all_boxes) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                if self.with_answer_embedding:
                    positive_center_embedding = self.answer_embedding(positive_sample_regular).unsqueeze(1)
                    positive_box = Box(positive_center_embedding, as_point=True)
                else:
                    positive_center_embedding = self.center_embedding(positive_sample_regular).unsqueeze(1)
                    positive_offset_embedding = torch.abs(self.offset_embedding(positive_sample_regular).unsqueeze(1))
                    positive_box = Box(positive_center_embedding, offset=positive_offset_embedding)
                positive_logit = self.cal_logit_box(positive_box, all_boxes, all_inv_masks, all_trans_masks, all_projection_dims, transitive=transitive, negative_box=all_negative_boxes, negation_indices=negation_indices)
            else:
                positive_logit = torch.Tensor([]).to(self.center_embedding.weight.device)
                
        else:
            positive_logit = None
            
        if type(negative_sample) != type(None):
            if len(all_boxes) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                if self.with_answer_embedding:
                    negative_center_embedding = self.answer_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                    negative_box = Box(negative_center_embedding, as_point=True)
                else:
                    negative_center_embedding = self.center_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                    negative_offset_embedding = torch.abs(self.offset_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))
                    negative_box = Box(negative_center_embedding, offset=negative_offset_embedding)
                negative_logit = self.cal_logit_box(negative_box, all_boxes, all_inv_masks, all_trans_masks, all_projection_dims, negative=True, transitive=transitive, negative_box=all_negative_boxes, negation_indices=negation_indices)
            else:
                negative_logit = torch.Tensor([]).to(self.center_embedding.weight.device)

        else:
            negative_logit = None

        all_query_boxes = Box(self.center_embedding.weight, self.offset_embedding.weight)
        if self.with_answer_embedding:
            all_answer_boxes = Box(self.answer_embedding.weight, as_point=True)
        else:
            all_answer_boxes = Box(self.center_embedding.weight, as_point=True)
        membership_logit = self.cal_membership_logit(all_answer_boxes, all_query_boxes)
        transitive_relation_logit = self.cal_transitive_relation_logit(self.transitive_ids, self.inverse_ids)

        return positive_logit, negative_logit, membership_logit, transitive_relation_logit, subsampling_weight, all_idxs
    
    def cal_dispersion_loss(self):
        """Encourage gene embeddings to spread out.

        Samples a random subset of gene entity IDs, retrieves their center
        embeddings, and returns the negative mean pairwise L1 distance.
        Minimising this loss maximises the average distance between gene
        embeddings, directly countering cluster collapse.
        """
        if self.gene_entity_ids is None or self.dispersion_weight == 0.0:
            return torch.tensor(0.0, device=self.center_embedding.weight.device)

        n = min(self.dispersion_sample_size, len(self.gene_entity_ids))
        idx = torch.randperm(len(self.gene_entity_ids), device=self.gene_entity_ids.device)[:n]
        gene_ids = self.gene_entity_ids[idx]

        emb = self.center_embedding(gene_ids)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)  # project onto unit sphere → distances bounded in [0, 2]
        pdist = torch.cdist(emb, emb, p=2)             # (n, n)  L2 pairwise distances, max = 2
        # Mask diagonal (self-distances = 0) and take mean of off-diagonal entries
        mask = ~torch.eye(n, dtype=torch.bool, device=emb.device)
        return pdist[mask].mean()                      # in (0, 2] — higher = more spread

    def cal_disjointness_loss(self, pairs):
        """Penalise overlapping box embeddings for disjoint class pairs.

        Uses box_disjointness_score (overlap amount): 0 when boxes are already
        disjoint, positive when they overlap. Minimising directly is sufficient —
        no margin or sign flip needed.

        Args:
            pairs: LongTensor (batch, 2) — entity ID pairs that must be disjoint.
        Returns:
            Scalar >= 0.
        """
        id_a, id_b = pairs[:, 0], pairs[:, 1]
        box_a = Box(self.center_embedding(id_a), torch.abs(self.offset_embedding(id_a)))
        box_b = Box(self.center_embedding(id_b), torch.abs(self.offset_embedding(id_b)))
        return Box.box_disjointness_score(box_a, box_b).mean()

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step, disjoint_iterator=None):
        transitive=False # Hardcoded to False
        
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries): # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
            
        # Build per-query-type slice map before forward() — iteration order is preserved.
        offset = 0
        query_type_slices = {}
        for query_structure in batch_queries_dict:
            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).to(model.device)
            n = len(batch_queries_dict[query_structure])
            query_type_slices[query_structure] = slice(offset, offset + n)
            offset += n

        positive_sample = positive_sample.to(model.device)
        negative_sample = negative_sample.to(model.device)
        subsampling_weight = subsampling_weight.to(model.device)

        positive_logit, negative_logit, membership_logit, transitive_relation_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=transitive)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)

        # Group scores by short name (e.g. kip_2, kip_3 → kip) for logging only.
        type_pos = collections.defaultdict(list)
        type_neg = collections.defaultdict(list)
        for query_structure, slc in query_type_slices.items():
            short = model.query_name_dict.get(query_structure, query_structure)
            type_pos[short].append(positive_score[slc])
            type_neg[short].append(negative_score[slc])

        # Re-weight intersection queries by 1/k to reduce variance from large phenotype sets
        k_weights = torch.ones(len(positive_sample), device=model.device)
        intersection_types = {'kip', 'kipd', 'ki2p'}
        for query_structure, slc in query_type_slices.items():
            short = model.query_name_dict.get(query_structure, query_structure)
            if short in intersection_types:
                k_actual = batch_queries_dict[query_structure][:, 0].float()
                k_weights[slc] = 1.0 / k_actual.clamp(min=1)

        effective_weight = subsampling_weight * k_weights
        positive_sample_loss = -(effective_weight * positive_score).sum() / effective_weight.sum()
        negative_sample_loss = -(effective_weight * negative_score).sum() / effective_weight.sum()

        membership_loss = -F.logsigmoid(membership_logit).mean()

        if transitive:
            relation_loss = -F.logsigmoid(transitive_relation_logit).mean()
        else:
            relation_loss = torch.tensor(0.0).to(positive_sample_loss.device)

        dispersion_loss = model.cal_dispersion_loss()

        if disjoint_iterator is not None:
            disjoint_pairs = next(disjoint_iterator).to(model.device)
            disjointness_loss = model.cal_disjointness_loss(disjoint_pairs)
        else:
            disjointness_loss = torch.tensor(0.0, device=positive_sample_loss.device)

        loss = (positive_sample_loss + negative_sample_loss)/2 - model.dispersion_weight * dispersion_loss + disjointness_loss
        loss.backward()
        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'membership_loss': membership_loss.item(),
            'transitive_rel_loss': relation_loss.item(),
            'dispersion_loss': dispersion_loss.item(),
            'disjointness_loss': disjointness_loss.item(),
            'loss': loss.item()
        }
        for short in type_pos:
            log[f'pos_{short}'] = -torch.cat(type_pos[short]).mean().item()
            log[f'neg_{short}'] = -torch.cat(type_neg[short]).mean().item()

        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, transitive_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                                            
                negative_sample = negative_sample.cuda()

                _, negative_logit, _, _, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict, transitive=transitive)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
 
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)

                    assert len(hard_answer.intersection(easy_answer)) == 0
                    
                    cur_ranking = ranking[idx, list(easy_answer) + list(transitive_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy + num_transitive
                    
                    answer_list = torch.arange(num_hard + num_easy + num_transitive).to(torch.float).cuda()
                                            
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics


