"""Prepare and validate cached GO OWL2Vec* projections."""

import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


MODES = ("upheno-first", "independent")
LEGACY_DEFAULT_SHA256 = "ee7f380c83418924c755e64505c55e4b45fd6e390f1c35a52346292e38edb0c5"
LEGACY_DEFAULT_ROWS = 219802
LEGACY_DEFAULT_UNIQUE = 187797


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def go_projection_paths(data_dir, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown GO projection mode: {mode}")
    name = "go_edges.tsv" if mode == "upheno-first" else "go_edges_independent.tsv"
    edges = Path(data_dir) / name
    return edges, edges.with_name(edges.name + ".metadata.json")


def go_projection_suffix(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown GO projection mode: {mode}")
    return "" if mode == "upheno-first" else "_go_independent"


def transd_checkpoint_identifier(fold, seed, dimension, batch_size, learning_rate,
                                 sources, projector_name, use_graph, tolerance,
                                 arm, go_projection_mode):
    tolerance_suffix = "" if tolerance == 5 else f"_tol_{tolerance}"
    return (
        f"transd_fold_{fold}_seed_{seed}_dim_{dimension}_bs_{batch_size}_lr_{learning_rate}"
        f"_{sources}_proj_{projector_name}_use_graph_{use_graph}"
        f"{tolerance_suffix}{go_projection_suffix(go_projection_mode)}{arm}"
    )


def _package_version():
    for package in ("mowl-borg", "mowl"):
        try:
            return version(package)
        except PackageNotFoundError:
            pass
    return None


def _source_hashes(data_dir, mode):
    sources = {"go.owl": sha256(Path(data_dir) / "go.owl")}
    if mode == "upheno-first":
        sources["upheno.owl"] = sha256(Path(data_dir) / "upheno.owl")
    return sources


def _edge_counts(path):
    counts = Counter()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            counts[line.rstrip("\r\n")] += 1
    return sum(counts.values()), len(counts)


def _validate_metadata(edges_path, metadata_path, data_dir, mode):
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid GO projection metadata {metadata_path}: {error}") from error
    expected = {
        "mode": mode,
        "projector": "OWL2VecStarProjector",
        "projector_settings": {
            "bidirectional_taxonomy": True,
            "only_taxonomy": False,
            "include_literals": False,
        },
        "output_sha256": sha256(edges_path),
        "source_sha256": _source_hashes(data_dir, mode),
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    rows, unique = _edge_counts(edges_path)
    if metadata.get("edge_rows") != rows:
        mismatches.append("edge_rows")
    if metadata.get("edge_unique") != unique:
        mismatches.append("edge_unique")
    if mismatches:
        joined = ", ".join(sorted(set(mismatches)))
        raise RuntimeError(
            f"Stale or mismatched GO projection cache {edges_path} ({joined}). "
            "Run code/projector/prepare_go_edges.py with --rebuild after checking the inputs."
        )
    return edges_path


def _atomic_write_edges(path, edges):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            for edge in edges:
                stream.write(f"{edge.src}\t{edge.rel}\t{edge.dst}\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def ensure_go_edges(data_dir, mode, projector_factory, dataset_factory, rebuild=False,
                    mowl_version=None, logger=None):
    data_dir = Path(data_dir)
    edges_path, metadata_path = go_projection_paths(data_dir, mode)
    logger = logger or logging.getLogger(__name__)
    if edges_path.exists() and not rebuild:
        if metadata_path.exists():
            return _validate_metadata(edges_path, metadata_path, data_dir, mode)
        digest = sha256(edges_path)
        rows, unique = _edge_counts(edges_path)
        if (mode == "upheno-first" and digest == LEGACY_DEFAULT_SHA256
                and rows == LEGACY_DEFAULT_ROWS and unique == LEGACY_DEFAULT_UNIQUE):
            logger.warning(
                "Using the verified published legacy GO cache %s without metadata "
                "(sha256 %s, %d rows, %d unique edges).",
                edges_path, digest, rows, unique
            )
            return edges_path
        raise RuntimeError(
            f"Refusing unmarked GO projection cache {edges_path} (sha256 {digest}). "
            "Run code/projector/prepare_go_edges.py with --rebuild after checking the inputs."
        )
    for source in ([data_dir / "upheno.owl"] if mode == "upheno-first" else []) + [data_dir / "go.owl"]:
        if not source.is_file():
            raise FileNotFoundError(source)
    projector = projector_factory()
    if mode == "upheno-first":
        projector.project(dataset_factory(str(data_dir / "upheno.owl")).ontology)
    projected = projector.project(dataset_factory(str(data_dir / "go.owl")).ontology)
    _atomic_write_edges(edges_path, projected)
    rows, unique = _edge_counts(edges_path)
    metadata = {
        "mode": mode,
        "projector": "OWL2VecStarProjector",
        "projector_settings": {
            "bidirectional_taxonomy": True,
            "only_taxonomy": False,
            "include_literals": False,
        },
        "mowl_version": _package_version() if mowl_version is None else mowl_version,
        "source_sha256": _source_hashes(data_dir, mode),
        "output_sha256": sha256(edges_path),
        "edge_rows": rows,
        "edge_unique": unique,
    }
    _atomic_write_json(metadata_path, metadata)
    logger.info("Wrote %s GO projection to %s (%d rows, %d unique edges).",
                mode, edges_path, rows, unique)
    return edges_path
