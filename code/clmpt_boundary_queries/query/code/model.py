"""CLMPT packed lookup execution variant; original parameterization and initialization."""
from original_model import QueryModel as OriginalQueryModel
from packed_encode import install


class QueryModel(OriginalQueryModel):
    def __init__(self, pretrained, allowed_hp_ids, device='cuda', temperature=.05):
        super().__init__(pretrained, allowed_hp_ids, device=device, temperature=temperature)
        install(self)
