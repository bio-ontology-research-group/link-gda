import json
from pathlib import Path

import pytest

from link_gda import go_projection


class FakeEdge:
    def __init__(self, src, rel, dst):
        self.src = src
        self.rel = rel
        self.dst = dst


class FakeDataset:
    def __init__(self, path):
        self.ontology = Path(path).name


class StatefulProjector:
    def __init__(self, calls):
        self.calls = calls

    def project(self, ontology):
        self.calls.append(ontology)
        return [FakeEdge(ontology, "state", str(len(self.calls)))]


def sources(tmp_path):
    (tmp_path / "go.owl").write_text("go", encoding="utf-8")
    (tmp_path / "upheno.owl").write_text("upheno", encoding="utf-8")


def test_upheno_first_uses_dedicated_stateful_projector_despite_cached_upheno_edges(tmp_path):
    sources(tmp_path)
    (tmp_path / "upheno_edges.tsv").write_text("cached\tedge\there\n", encoding="utf-8")
    calls = []
    path = go_projection.ensure_go_edges(
        tmp_path, "upheno-first", lambda: StatefulProjector(calls), FakeDataset,
        mowl_version="test"
    )
    assert calls == ["upheno.owl", "go.owl"]
    assert path.read_text(encoding="utf-8") == "go.owl\tstate\t2\n"
    metadata = json.loads((tmp_path / "go_edges.tsv.metadata.json").read_text())
    assert metadata["source_sha256"].keys() == {"go.owl", "upheno.owl"}


def test_independent_uses_fresh_go_only_projection(tmp_path):
    (tmp_path / "go.owl").write_text("go", encoding="utf-8")
    calls = []
    path = go_projection.ensure_go_edges(
        tmp_path, "independent", lambda: StatefulProjector(calls), FakeDataset,
        mowl_version="test"
    )
    assert calls == ["go.owl"]
    assert path.name == "go_edges_independent.tsv"
    assert path.read_text(encoding="utf-8") == "go.owl\tstate\t1\n"
    metadata = json.loads((tmp_path / "go_edges_independent.tsv.metadata.json").read_text())
    assert metadata["source_sha256"].keys() == {"go.owl"}


def test_valid_cache_does_not_invoke_projector(tmp_path):
    sources(tmp_path)
    path = go_projection.ensure_go_edges(
        tmp_path, "upheno-first", lambda: StatefulProjector([]), FakeDataset,
        mowl_version="test"
    )

    def fail_factory():
        raise AssertionError("cache hit must not construct a projector")

    assert go_projection.ensure_go_edges(
        tmp_path, "upheno-first", fail_factory, FakeDataset
    ) == path


@pytest.mark.parametrize("field,value", [
    ("mode", "independent"),
    ("output_sha256", "wrong"),
    ("projector", "another-projector"),
])
def test_cache_metadata_mismatch_is_rejected(tmp_path, field, value):
    sources(tmp_path)
    go_projection.ensure_go_edges(
        tmp_path, "upheno-first", lambda: StatefulProjector([]), FakeDataset,
        mowl_version="test"
    )
    metadata_path = tmp_path / "go_edges.tsv.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Stale or mismatched"):
        go_projection.ensure_go_edges(
            tmp_path, "upheno-first", lambda: StatefulProjector([]), FakeDataset
        )


def test_source_change_is_rejected(tmp_path):
    sources(tmp_path)
    go_projection.ensure_go_edges(
        tmp_path, "upheno-first", lambda: StatefulProjector([]), FakeDataset,
        mowl_version="test"
    )
    (tmp_path / "go.owl").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source_sha256"):
        go_projection.ensure_go_edges(
            tmp_path, "upheno-first", lambda: StatefulProjector([]), FakeDataset
        )


def test_unknown_unmarked_cache_is_rejected(tmp_path):
    path = tmp_path / "go_edges.tsv"
    path.write_text("a\tr\tb\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing unmarked"):
        go_projection.ensure_go_edges(
            tmp_path, "upheno-first", lambda: StatefulProjector([]), FakeDataset
        )


def test_known_legacy_cache_is_accepted_without_ontology_sources(tmp_path, monkeypatch):
    path = tmp_path / "go_edges.tsv"
    path.write_text("a\tr\tb\na\tr\tb\n", encoding="utf-8")
    monkeypatch.setattr(go_projection, "LEGACY_DEFAULT_SHA256", go_projection.sha256(path))
    monkeypatch.setattr(go_projection, "LEGACY_DEFAULT_ROWS", 2)
    monkeypatch.setattr(go_projection, "LEGACY_DEFAULT_UNIQUE", 1)
    result = go_projection.ensure_go_edges(
        tmp_path, "upheno-first", lambda: StatefulProjector([]), FakeDataset
    )
    assert result == path


def test_mode_suffix_preserves_default_names():
    assert go_projection.go_projection_suffix("upheno-first") == ""
    assert go_projection.go_projection_suffix("independent") == "_go_independent"


def test_convkb_warm_start_identifier_places_mode_before_arm():
    identifier = go_projection.transd_checkpoint_identifier(
        2, 3, 200, 8192, "0.001", "pheno_func", "owl2vecstar", True,
        15, "_calsel", "independent"
    )
    assert identifier.endswith("_tol_15_go_independent_calsel")
