"""Independent recursive comparison of saved qualification states."""
import hashlib
import numpy as np
import torch


def compare(left, right, atol=1e-6, rtol=1e-5):
    result = {'exact': True, 'numerical': True, 'tensor_arrays': 0,
              'maximum_absolute_error': 0., 'differences': [], 'absolute_tolerance': atol,
              'relative_tolerance': rtol}
    def difference(path, detail, numerical=False):
        result['exact'] = False
        if not numerical:
            result['numerical'] = False
        if len(result['differences']) < 100:
            result['differences'].append({'path': path, **detail})
    def visit(a, b, path):
        if torch.is_tensor(a) or isinstance(a, np.ndarray):
            if torch.is_tensor(a):
                if not torch.is_tensor(b):
                    difference(path, {'reason': 'tensor type mismatch'})
                    return
                a, b = a.detach().cpu().contiguous().numpy(), b.detach().cpu().contiguous().numpy()
            elif not isinstance(b, np.ndarray):
                difference(path, {'reason': 'array type mismatch'})
                return
            result['tensor_arrays'] += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                difference(path, {'reason': 'shape/dtype mismatch'})
                return
            a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
            ah = hashlib.sha256(memoryview(a).cast('B')).hexdigest()
            bh = hashlib.sha256(memoryview(b).cast('B')).hexdigest()
            if np.issubdtype(a.dtype, np.number):
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    difference(path, {'reason': 'nonfinite state'})
                    return
            if ah != bh:
                floating = np.issubdtype(a.dtype, np.floating)
                error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.
                result['maximum_absolute_error'] = max(result['maximum_absolute_error'], error)
                passed = floating and np.allclose(a, b, atol=atol, rtol=rtol)
                difference(path, {'left_sha256': ah, 'right_sha256': bh, 'maximum_absolute_error': error}, numerical=passed)
            return
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                difference(path, {'reason': 'mapping keys mismatch'})
                return
            for key in a:
                visit(a[key], b[key], path + '/' + str(key))
            return
        if isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                difference(path, {'reason': 'sequence type/length mismatch'})
                return
            for i, (av, bv) in enumerate(zip(a, b)):
                visit(av, bv, path + '/' + str(i))
            return
        if type(a) is not type(b) or a != b:
            floating = isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating))
            passed = floating and np.isfinite(a) and np.isfinite(b) and np.isclose(a, b, atol=atol, rtol=rtol)
            difference(path, {'reason': 'scalar mismatch', 'left': repr(a), 'right': repr(b)}, numerical=passed)
    visit(left, right, 'state')
    return result
