"""Reject checkpoints with incompatible graph mappings or evaluation protocols."""
import hashlib
import json
from pathlib import Path

def graph_identity(factory):
    payload={'entity_to_id':factory.entity_to_id,'relation_to_id':factory.relation_to_id,
             'create_inverse_triples':factory.create_inverse_triples}
    h=hashlib.sha256(json.dumps(payload,sort_keys=True).encode())
    h.update(factory.mapped_triples.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def check_or_write(path,identity,require_existing=False):
    path=Path(str(path)+'.identity.json')
    if path.exists():
        if json.loads(path.read_text())!=identity:raise ValueError(f'Checkpoint identity mismatch: {path}')
    elif require_existing:raise FileNotFoundError(f'Missing checkpoint identity: {path}')
    else:path.write_text(json.dumps(identity,sort_keys=True,indent=2)+'\n')
