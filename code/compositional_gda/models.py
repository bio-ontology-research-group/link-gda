"""Path query-embedding models: GQE/TransE path, Query2Box, BetaE and GeometrE.

Each model exposes the same interface:

- ``encode(anchors, relations)`` embeds the path query that starts at the anchor
  entities and applies the relation projections in column order. No entity is
  looked up between hops.
- ``score(query, entities, negative=False)`` returns gamma minus the method's
  distance for every (query, entity) pair. ``entities`` has shape ``[B, K]`` or
  ``[1, K]``; the result has shape ``[B, K]``.
- ``auxiliary_losses()`` returns method-specific loss terms (GeometrE
  membership only).

Sources, pinned in ``SOURCES``:

- GQE/TransE path, Query2Box and BetaE follow KGReasoning ``models.py``
  (``embed_query_vec``/``cal_logit_vec``, ``embed_query_box``/``cal_logit_box``,
  ``embed_query_beta``/``cal_logit_beta``, ``BetaProjection``, ``Regularizer``,
  parameter initialisation in ``KGReasoning.__init__``).
- GeometrE follows ``src/models.py`` (parameters, ``cal_logit_box``,
  ``forward_box`` membership term, ``train_step`` loss), ``src/embeddings.py``
  (``get_box_data``, ``embedding_1p``, ``embedding_2p``) and ``src/box.py``
  (``transform``, ``box_inclusion_score``, ``corner_loss``) with transitivity off.
"""
import torch
import torch.nn.functional as F
from torch import nn

SOURCES = {
    'kgreasoning': {'commit': 'ec728497f083973edbe8b9e3c4066f416fc74250', 'files': ['models.py', 'dataloader.py', 'main.py']},
    'geometre': {'commit': '9c4551585fbbbc9eaaced64c86077a9a42b13a96',
                 'files': ['src/models.py', 'src/embeddings.py', 'src/box.py', 'src/main.py']},
}

METHODS = ('transe', 'q2b', 'betae', 'geometre')
MEMBERSHIP_MODES = ('original', 'sign_corrected')

METHOD_DEFAULTS = {
    'transe': {'gamma': 24.0},
    'q2b': {'gamma': 24.0, 'cen': 0.02, 'offset_activation': 'none'},
    'betae': {'gamma': 60.0, 'beta_hidden': 1600, 'beta_layers': 2},
    'geometre': {'gamma': 20.0, 'alpha': 0.2, 'answer_embedding': False, 'membership_mode': None},
}

METHOD_LABELS = {
    'transe': 'GQE/TransE path (KGReasoning geo=vec)',
    'q2b': 'Query2Box (KGReasoning geo=box)',
    'betae': 'BetaE (KGReasoning geo=beta)',
    'geometre': 'GeometrE (released code, transitivity off)',
}


def embedding_range(gamma, dim, epsilon=2.0):
    """Initialisation half-width computed in float32 as both sources do."""
    stored_gamma = torch.Tensor([gamma]).item()
    return torch.Tensor([(stored_gamma + epsilon) / dim]).item()


def uniform_parameter(rows, columns, low, high):
    parameter = nn.Parameter(torch.zeros(rows, columns))
    nn.init.uniform_(parameter, a=low, b=high)
    return parameter


def uniform_embedding(rows, columns, half_width):
    embedding = nn.Embedding(rows, columns)
    nn.init.uniform_(embedding.weight, a=-half_width, b=half_width)
    return embedding


def query_loss(positive_logit, negative_logit, weight):
    """KGReasoning ``train_step`` loss; GeometrE uses the same query term.

    ``positive_logit`` is ``[B]``, ``negative_logit`` is ``[B, K]`` and ``weight``
    is the subsampling weight ``[B]``. Returns the loss and its two halves.
    """
    negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
    positive_score = F.logsigmoid(positive_logit)
    positive_loss = -(weight * positive_score).sum() / weight.sum()
    negative_loss = -(weight * negative_score).sum() / weight.sum()
    return (positive_loss + negative_loss) / 2, positive_loss, negative_loss


class PathModel(nn.Module):
    """Shared buffers and helpers."""

    def __init__(self, gamma):
        super().__init__()
        self.register_buffer('gamma', torch.Tensor([gamma]))

    def auxiliary_losses(self):
        return {}

    def diagnostics(self):
        return {}


class VecPath(PathModel):
    """GQE path operator: the query vector is the anchor plus each relation vector."""

    def __init__(self, num_entities, num_relations, dim, gamma):
        super().__init__(gamma)
        half_width = embedding_range(gamma, dim)
        self.entity_embedding = uniform_parameter(num_entities, dim, -half_width, half_width)
        self.relation_embedding = uniform_parameter(num_relations, dim, -half_width, half_width)

    def encode(self, anchors, relations):
        center = torch.index_select(self.entity_embedding, 0, anchors)
        for hop in range(relations.shape[1]):
            center = center + torch.index_select(self.relation_embedding, 0, relations[:, hop])
        return (center,)

    def score(self, query, entities, negative=False):
        (center,) = query
        entity = self.entity_embedding[entities]
        return self.gamma - torch.norm(entity - center.unsqueeze(1), p=1, dim=-1)


class BoxPath(PathModel):
    """Query2Box path operator; entities are points and relation offsets are not clamped by default."""

    def __init__(self, num_entities, num_relations, dim, gamma, cen=0.02, offset_activation='none'):
        super().__init__(gamma)
        half_width = embedding_range(gamma, dim)
        self.cen = float(cen)
        self.offset_activation = offset_activation
        self.func = {'none': lambda x: x, 'relu': F.relu, 'softplus': F.softplus}[offset_activation]
        self.entity_embedding = uniform_parameter(num_entities, dim, -half_width, half_width)
        self.relation_embedding = uniform_parameter(num_relations, dim, -half_width, half_width)
        self.offset_embedding = uniform_parameter(num_relations, dim, 0.0, half_width)

    def encode(self, anchors, relations):
        center = torch.index_select(self.entity_embedding, 0, anchors)
        offset = torch.zeros_like(center)
        for hop in range(relations.shape[1]):
            center = center + torch.index_select(self.relation_embedding, 0, relations[:, hop])
            offset = offset + self.func(torch.index_select(self.offset_embedding, 0, relations[:, hop]))
        return center, offset

    def score(self, query, entities, negative=False):
        center, offset = (part.unsqueeze(1) for part in query)
        delta = (self.entity_embedding[entities] - center).abs()
        outside = F.relu(delta - offset)
        inside = torch.min(delta, offset)
        return self.gamma - torch.norm(outside, p=1, dim=-1) - self.cen * torch.norm(inside, p=1, dim=-1)

    def diagnostics(self):
        offsets = self.offset_embedding.detach()
        return {'relation_offset_negative_fraction': float((offsets < 0).float().mean())}


class Regularizer:
    """KGReasoning ``Regularizer``: clamp(x + base_add, min_val, max_val)."""

    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, value):
        return torch.clamp(value + self.base_add, self.min_val, self.max_val)


class BetaIntersection(nn.Module):
    """KGReasoning ``BetaIntersection``.

    It is constructed because the source constructs it before the projection
    network, so the projection network receives the source's initial weights for
    a given seed. Path queries never call it and it receives no gradient.
    """

    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(2 * dim, 2 * dim)
        self.layer2 = nn.Linear(2 * dim, dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)


class BetaProjection(nn.Module):
    """KGReasoning ``BetaProjection`` with identical layer construction and initialisation order."""

    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layer1 = nn.Linear(entity_dim + relation_dim, hidden_dim)
        self.layer0 = nn.Linear(hidden_dim, entity_dim)
        for index in range(2, num_layers + 1):
            setattr(self, f'layer{index}', nn.Linear(hidden_dim, hidden_dim))
        for index in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f'layer{index}').weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, entity, relation):
        hidden = torch.cat([entity, relation], dim=-1)
        for index in range(1, self.num_layers + 1):
            hidden = F.relu(getattr(self, f'layer{index}')(hidden))
        return self.projection_regularizer(self.layer0(hidden))


class BetaPath(PathModel):
    """BetaE path operator: repeated MLP projection of Beta parameters, KL-based score."""

    def __init__(self, num_entities, num_relations, dim, gamma, beta_hidden=1600, beta_layers=2):
        super().__init__(gamma)
        half_width = embedding_range(gamma, dim)
        self.entity_embedding = uniform_parameter(num_entities, 2 * dim, -half_width, half_width)
        self.entity_regularizer = Regularizer(1, 0.05, 1e9)
        self.projection_regularizer = Regularizer(1, 0.05, 1e9)
        self.relation_embedding = uniform_parameter(num_relations, dim, -half_width, half_width)
        self.center_net = BetaIntersection(dim)
        self.projection_net = BetaProjection(2 * dim, dim, beta_hidden, self.projection_regularizer, beta_layers)

    def encode(self, anchors, relations):
        embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, 0, anchors))
        for hop in range(relations.shape[1]):
            embedding = self.projection_net(embedding, torch.index_select(self.relation_embedding, 0, relations[:, hop]))
        return tuple(torch.chunk(embedding, 2, dim=-1))

    def score(self, query, entities, negative=False):
        alpha, beta = (part.unsqueeze(1) for part in query)
        query_distribution = torch.distributions.beta.Beta(alpha, beta)
        entity_alpha, entity_beta = torch.chunk(self.entity_regularizer(self.entity_embedding[entities]), 2, dim=-1)
        entity_distribution = torch.distributions.beta.Beta(entity_alpha, entity_beta)
        divergence = torch.distributions.kl.kl_divergence(entity_distribution, query_distribution)
        return self.gamma - torch.norm(divergence, p=1, dim=-1)


class GeometrEPath(PathModel):
    """GeometrE box path with transitivity off.

    ``membership_mode='original'`` minimises softplus(-d) as released
    (``models.py`` ``train_step``: ``-logsigmoid(membership_logit)``), where d is
    the box inclusion distance between each entity's raw query box and its own
    answer point. ``'sign_corrected'`` minimises softplus(d) and changes nothing
    else; it is an experimental correction, not a reconstruction of a paper formula.
    The ``*_neg_*`` relation tables are omitted: the source creates them after all
    used tables and only negation queries read them.
    """

    def __init__(self, num_entities, num_relations, dim, gamma, alpha=0.2, answer_embedding=False,
                 membership_mode='original'):
        super().__init__(gamma)
        if membership_mode not in MEMBERSHIP_MODES:
            raise ValueError(f'membership_mode must be one of {MEMBERSHIP_MODES}')
        half_width = embedding_range(gamma, dim)
        self.register_buffer('alpha', torch.Tensor([alpha]))
        self.membership_mode = membership_mode
        self.with_answer_embedding = bool(answer_embedding)
        self.center_embedding = uniform_embedding(num_entities, dim, half_width)
        self.offset_embedding = uniform_embedding(num_entities, dim, half_width)
        if self.with_answer_embedding:
            self.answer_embedding = uniform_embedding(num_entities, dim, half_width)
        self.center_mul = uniform_embedding(num_relations, dim, half_width)
        self.center_add = uniform_embedding(num_relations, dim, half_width)
        self.offset_mul = uniform_embedding(num_relations, dim, half_width)
        self.offset_add = uniform_embedding(num_relations, dim, half_width)

    def answer_points(self, entities):
        table = self.answer_embedding if self.with_answer_embedding else self.center_embedding
        return table(entities)

    def encode(self, anchors, relations):
        center = self.center_embedding(anchors)
        offset = torch.abs(self.offset_embedding(anchors))
        for hop in range(relations.shape[1]):
            index = relations[:, hop]
            center = center * self.center_mul(index) + self.center_add(index)
            offset = torch.abs(offset * self.offset_mul(index) + self.offset_add(index))
        return center, offset

    @staticmethod
    def inclusion_distance(center, offset, point, alpha, corner):
        lower = center - offset
        upper = center + offset
        outside = torch.linalg.norm(torch.relu(point - upper) + torch.relu(lower - point), dim=-1, ord=1)
        inside = torch.linalg.norm(center - torch.min(upper, torch.max(lower, point)), dim=-1, ord=1)
        distance = outside + alpha * inside
        if corner:
            distance = distance + torch.linalg.norm(torch.relu(lower - upper), dim=-1, ord=1)
        return distance

    def score(self, query, entities, negative=False):
        center, offset = (part.unsqueeze(1) for part in query)
        point = self.answer_points(entities)
        return self.gamma - self.inclusion_distance(center, offset, point, self.alpha, corner=not negative)

    def membership_distance(self):
        point = self.answer_embedding.weight if self.with_answer_embedding else self.center_embedding.weight
        return self.inclusion_distance(self.center_embedding.weight, self.offset_embedding.weight, point,
                                       self.alpha, corner=True)

    def auxiliary_losses(self):
        distance = self.membership_distance()
        sign = 1.0 if self.membership_mode == 'original' else -1.0
        return {'membership': -F.logsigmoid(sign * distance).mean()}

    def diagnostics(self):
        with torch.no_grad():
            distance = self.membership_distance()
            raw = self.offset_embedding.weight
            return {'membership_distance_mean': float(distance.mean()),
                    'raw_offset_negative_fraction': float((raw < 0).float().mean())}


def build_model(config, num_entities, num_relations):
    """Construct the configured model; ``config`` must already carry resolved method defaults."""
    method = config['method']
    dim = config['dim']
    gamma = config['gamma']
    if method == 'transe':
        return VecPath(num_entities, num_relations, dim, gamma)
    if method == 'q2b':
        return BoxPath(num_entities, num_relations, dim, gamma, config['cen'], config['offset_activation'])
    if method == 'betae':
        return BetaPath(num_entities, num_relations, dim, gamma, config['beta_hidden'], config['beta_layers'])
    if method == 'geometre':
        return GeometrEPath(num_entities, num_relations, dim, gamma, config['alpha'], config['answer_embedding'],
                            config['membership_mode'])
    raise ValueError(f'Unknown method {method}')
