import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("rq2_stats_under_test", ROOT / "code" / "analysis" / "rq2_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def folds(values):
    return {str(fold): value for fold, value in enumerate(values)}


def document(left=None, right=None):
    module = load_module()
    left = left or list(range(10))
    right = right or [value + (1 if value % 2 else 2) for value in range(10)]
    return {
        "metadata": {"source": "test fixture", "paired_cases_verified_by": "fixture construction"},
        "comparisons": {
            name: {"left": folds(left), "right": folds(right)} for name in module.COMPARISONS
        },
    }


def test_two_sided_test_is_invariant_to_sign_reversal_and_differs_from_directional_p():
    module = load_module()
    left = list(range(10))
    right = [value + (1 if value % 2 else 2) for value in range(10)]
    forward = module.corrected_test(left, right)
    reversed_result = module.corrected_test(right, left)
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    expected_t = float(difference.mean() / (difference.std(ddof=1) * math.sqrt(1 / 10 + 1 / 9)))
    expected_directional = float(stats.t.sf(abs(expected_t), 9))
    expected_two_sided = 2 * expected_directional
    assert forward["t"] == pytest.approx(expected_t)
    assert forward["p_two_sided"] == pytest.approx(reversed_result["p_two_sided"])
    assert forward["p_two_sided"] == pytest.approx(expected_two_sided)
    assert forward["p_bonferroni"] == pytest.approx(min(1.0, 6 * expected_two_sided))


def test_bonferroni_is_applied_before_rounding_and_capped_at_one():
    module = load_module()
    result = module.corrected_test(range(10), [value + (-1) ** value for value in range(10)])
    assert result["p_bonferroni"] == 1.0


def test_rejects_zero_variance_as_an_undefined_corrected_test():
    module = load_module()
    with pytest.raises(ValueError, match="zero variance"):
        module.corrected_test([1] * 10, [2] * 10)


@pytest.mark.parametrize(
    "left, right, message",
    [
        ([1] * 9, [2] * 9, "ten folds"),
        ([[1] * 10], [[2] * 10], "one-dimensional"),
        ([1] * 9 + [float("inf")], [2] * 10, "finite values"),
    ],
)
def test_corrected_test_validates_public_inputs(left, right, message):
    module = load_module()
    with pytest.raises(ValueError, match=message):
        module.corrected_test(left, right)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data["comparisons"]["f_vs_p"]["left"].pop("9"), "exactly folds 0..9"),
        (lambda data: data["comparisons"]["f_vs_p"]["right"].__setitem__("4", float("nan")), "finite number"),
    ],
)
def test_rejects_missing_or_nonfinite_folds(mutation, message):
    module = load_module()
    data = document()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        module.calculate(data, "fixture-sha")


def test_cli_records_exact_input_digest_metadata_and_all_five_comparisons(tmp_path):
    path = tmp_path / "rq2.json"
    raw = json.dumps(document(), allow_nan=False).encode()
    path.write_bytes(raw)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "code" / "analysis" / "rq2_stats.py"), "--input", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["input"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert output["input"]["metadata"] == document()["metadata"]
    assert set(output["comparisons"]) == set(load_module().COMPARISONS)
    assert "cannot prove" in output["protocol"]["alignment_limit"]
