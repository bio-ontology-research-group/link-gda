"""Commit selected artifacts with a resumable validation state."""
from pathlib import Path
import shutil
from .common import sha


def publish_selection(directory, artifacts):
    directory = Path(directory)
    if set(artifacts) != {'best.pt','validation.npz'}:
        raise ValueError('Resume state lacks complete selected artifacts')
    for name, record in artifacts.items():
        if Path(record['file']).name != record['file']:
            raise ValueError('Invalid selection artifact path')
        source = directory/record['file']
        if sha(source) != record['sha256']:
            raise ValueError('Committed selection artifact changed')
        temporary = directory/(name+'.publish.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(directory/name)


def pending_validation(curve, last, total):
    if not 0 <= last <= total:
        raise ValueError('Resume step outside training budget')
    expected = sorted({0, *range(100,last+1,100), *([total] if last==total else [])})
    actual = [item['step'] for item in curve]
    if actual == expected:
        return False
    if expected[-1] == last and actual == expected[:-1]:
        return True
    raise ValueError('Resume state has missing or repeated past validation steps')


def require_complete_validation(progress, total, best):
    if progress['step'] != total or pending_validation(progress['curve'],total,total):
        raise ValueError('Training lacks its final scheduled validation')
    if progress['best'] != best:
        raise ValueError('Completion and validation choice differ')
