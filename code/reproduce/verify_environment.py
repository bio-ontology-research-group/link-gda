"""Verify a reconstructed ULTRA environment against its captured package identities."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys


def canonical(name):
    return re.sub(r'[-_.]+','-',name).lower()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    original=json.loads(a.capture.read_text())
    if platform.python_version()!=original['python']:
        raise ValueError('Python version differs from captured environment')
    expected={canonical(d['name']):d['version'] for d in original['distributions']}
    installed={canonical(d.metadata['Name']):d.version for d in importlib.metadata.distributions()}
    if expected!=installed:
        raise ValueError('Python distributions differ: '+str({n:(expected.get(n),installed.get(n))
            for n in sorted(expected.keys()|installed.keys()) if expected.get(n)!=installed.get(n)}))
    records={r['name']:r for path in (Path(sys.prefix)/'conda-meta').glob('*.json')
             for r in [json.loads(path.read_text())]}
    if set(records)!={r['name'] for r in original['conda']}:
        raise ValueError('Conda package set differs')
    for record in original['conda']:
        for field in ['version','build','md5']:
            if record[field]!=records[record['name']].get(field):
                raise ValueError('Conda package identity differs: '+record['name']+'/'+field)
    import numpy as np
    import torch
    import torch_geometric
    import torch_scatter
    x=torch.tensor([1.,2.,3.]);index=torch.tensor([0,1,0])
    assert torch.equal(torch_scatter.scatter_add(x,index,dim=0),torch.tensor([4.,2.]))
    assert np.array_equal(torch.tensor(np.array([1.,2.])).numpy(),np.array([1.,2.]))
    result={'capture_sha256':hashlib.sha256(a.capture.read_bytes()).hexdigest(),
        'python':platform.python_version(),'conda_packages':len(records),
        'python_distributions':len(installed),'all_package_identities_equal':True,
        'numpy':np.__version__,'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,
        'torch_scatter':torch_scatter.__version__,'cpu_extension_and_numpy_interchange':True,
        'scope':'Exact captured package versions and Conda builds/MD5; imports and CPU extension; not GPU trajectory parity'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
