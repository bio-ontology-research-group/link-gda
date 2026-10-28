"""Commit selected artifacts with a resumable validation state."""
from pathlib import Path
import shutil
from .common import sha, eligible, key


def publish_selection(directory, artifacts, expected_names=None):
    directory = Path(directory)
    expected_names = {'best.pt','validation.npz'} if expected_names is None else set(expected_names)
    if not expected_names or set(artifacts) != expected_names:
        raise ValueError('Resume state lacks complete selected artifacts')
    for name, record in artifacts.items():
        if Path(name).name != name or Path(record['file']).name != record['file']:
            raise ValueError('Invalid selection artifact path')
        source = directory/record['file']
        if sha(source) != record['sha256']:
            raise ValueError('Committed selection artifact changed')
        temporary = directory/(name+'.publish.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(directory/name)


def pending_validation(curve, last, total, frequency=100):
    if frequency <= 0:
        raise ValueError('Validation frequency must be positive')
    if not 0 <= last <= total:
        raise ValueError('Resume step outside training budget')
    expected = sorted({0, *range(frequency,last+1,frequency), *([total] if last==total else [])})
    actual = [item['step'] for item in curve]
    if actual == expected:
        return False
    if expected[-1] == last and actual == expected[:-1]:
        return True
    raise ValueError('Resume state has missing or repeated past validation steps')


def require_complete_validation(progress, total, best, frequency=100):
    if progress['step'] != total or pending_validation(progress['curve'],total,total,frequency):
        raise ValueError('Training lacks its final scheduled validation')
    if progress['best'] != best:
        raise ValueError('Completion and validation choice differ')


def verify_boundary_state(before, after):
    """Validation alone must preserve the exact interrupted training state."""
    import torch
    def equal(left, right, path):
        if isinstance(left, torch.Tensor):
            valid = isinstance(right,torch.Tensor) and left.dtype==right.dtype and torch.equal(left,right)
            if not valid: raise ValueError('Boundary restoration changed '+path)
        elif isinstance(left,dict):
            if left.keys()!=right.keys():raise ValueError('Boundary state keys changed '+path)
            for name in left:equal(left[name],right[name],path+'/'+str(name))
        elif isinstance(left,(tuple,list)):
            if len(left)!=len(right):raise ValueError('Boundary state length changed '+path)
            for j,(x,y) in enumerate(zip(left,right)):equal(x,y,path+'/'+str(j))
        elif left!=right:raise ValueError('Boundary restoration changed '+path)
    for name in ['model','pool','optimizer','rng','torch_rng','cuda_rng','step','baseline']:
        equal(before[name],after[name],name)
    total=before['step']
    if not pending_validation(before['curve'],total,total):
        raise ValueError('Boundary input does not have pending validation')
    if pending_validation(after['curve'],total,total) or after['curve'][:-1]!=before['curve']:
        raise ValueError('Pending boundary validation was not completed exactly once')
    final=after['curve'][-1];current=final['both_modes']['calibrated']
    chosen=eligible(current,before['baseline']) and key(current)>key(before['best']['metrics'])
    best=({'step':total,'metrics':current,'both_modes':final['both_modes']} if chosen else before['best'])
    if final['eligible']!=eligible(current,before['baseline']) or final['selected']!=chosen or after['best']!=best:
        raise ValueError('Boundary validation choice violates the fixed rule')
    return {'steps':total,'selected_step':after['best']['step'],
            'restored_state_bitwise_equal':True,'pending_validation_completed':True,
            'validation_steps':[item['step'] for item in after['curve']]}
