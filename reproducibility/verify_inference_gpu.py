"""Check native GPU execution using synthetic graphs; no benchmark/test queries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['matched', 'query'], required=True)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--ultra', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError('Choose a new verification output')
sys.path.insert(0, str(args.source / 'code'))
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable in the reconstructed environment')
torch.set_num_threads(1)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
checks = {}
rtol, atol = 1e-4, 1e-5

if args.mode == 'matched':
    from pykeen.models import ConvKB, TransD
    from pykeen.triples import TriplesFactory
    factory = TriplesFactory.from_labeled_triples(
        np.array([['g0', 'r', 'p0'], ['g1', 'r', 'p1']]), create_inverse_triples=True)
    for cls, options in [(TransD, {'relation_dim': 8}),
                         (ConvKB, {'num_filters': 2, 'hidden_dropout_rate': 0.})]:
        cpu = cls(triples_factory=factory, embedding_dim=8, random_seed=0, **options).eval()
        gpu = copy.deepcopy(cpu).cuda().eval()
        expected = cpu.score_hrt(factory.mapped_triples)
        actual = gpu.score_hrt(factory.mapped_triples.cuda())
        assert actual.is_cuda and actual.shape == (2, 1) and torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        (-actual.mean()).backward()
        grads = [p.grad for p in gpu.parameters() if p.grad is not None]
        assert grads and all(g.is_cuda and torch.isfinite(g).all() for g in grads)
        checks[cls.__name__] = dict(cpu_gpu_max_abs_error=float((actual.cpu()-expected).abs().max()),
                                   finite_gpu_scores=True, finite_gpu_gradients=True)
else:
    sys.path.insert(0, str(args.ultra))
    from torch_geometric.data import Data
    from ultra.tasks import build_relation_graph
    from phenoquery.model import QueryBackbone
    from phenoquery.matched_support import load_initial
    from phenoquery.ultra_io import load_model_config
    triples = torch.tensor([[0,1,0],[1,2,0],[2,3,1],[3,4,1],
                            [4,5,0],[5,0,1],[0,2,1],[1,4,0]])
    edge = triples[:, :2].T
    graph = build_relation_graph(Data(edge_index=torch.cat([edge, edge.flip(0)], dim=1),
        edge_type=torch.cat([triples[:,2], triples[:,2]+2]), num_nodes=6, num_relations=4))
    batch = torch.stack([torch.zeros(6, dtype=torch.long), torch.arange(6),
                         torch.full((6,), 2, dtype=torch.long)], dim=-1)[None]
    for initialization, kind in [('ultra_4g', 'general'), ('ultraquery', 'query')]:
        checkpoint = args.ultra / 'ckpts' / (initialization+'.pth')
        cpu = QueryBackbone(load_model_config(str(args.ultra))).eval()
        cpu.ultra.load_state_dict(load_initial(checkpoint, kind))
        gpu = copy.deepcopy(cpu).cuda().eval()
        with torch.no_grad():
            expected = cpu(graph, batch)
            actual = gpu(copy.deepcopy(graph).cuda(), batch.cuda())
        assert actual.is_cuda and actual.shape == (1, 6) and torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        checks[initialization] = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            cpu_gpu_max_abs_error=float((actual.cpu()-expected).abs().max()), finite_gpu_scores=True)
torch.cuda.synchronize()
result = dict(status='passed', mode=args.mode, device=torch.cuda.get_device_name(0),
    capability=list(torch.cuda.get_device_capability(0)), torch=str(torch.__version__),
    cuda=torch.version.cuda, rtol=rtol, atol=atol, checks=checks,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Synthetic-graph GPU runtime readiness and CPU/GPU numerical checks. No production checkpoint inference, benchmark evaluation or training parity claim.')
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
