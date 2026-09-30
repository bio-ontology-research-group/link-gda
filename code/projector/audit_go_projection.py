"""Project two GO snapshots and audit their edge sets against an existing TSV."""

import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", required=True, type=Path, help="Path to go.owl")
    parser.add_argument("--go-plus", required=True, type=Path, help="Path to go-plus.owl")
    parser.add_argument("--existing", required=True, type=Path, help="Existing edge TSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=Path, help="Ontology to project before GO")
    parser.add_argument("--fresh-per-ontology", action="store_true",
                        help="Create a fresh projector for each GO ontology")
    parser.add_argument("--memory", default="10g", help="JVM maximum heap (default: 10g)")
    return parser.parse_args(argv)


def announce(message):
    print(message, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_metadata(path):
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256(path)}


def optional_iri(value):
    try:
        return str(value.get()) if value.isPresent() else None
    except Exception:
        return None


def ontology_metadata(ontology):
    try:
        identifier = ontology.getOntologyID()
        return {
            "ontology_iri": optional_iri(identifier.getOntologyIRI()),
            "version_iri": optional_iri(identifier.getVersionIRI()),
        }
    except Exception as error:
        return {"ontology_iri": None, "version_iri": None,
                "metadata_error": f"{type(error).__name__}: {error}"}


def read_existing(path):
    triples = Counter()
    lines = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            lines += 1
            triples[line.rstrip("\r\n")] += 1
    return triples, lines


def project(name, source, output, PathDataset, projector):
    announce(f"Loading {name}: {source}")
    dataset = PathDataset(str(source))
    metadata = ontology_metadata(dataset.ontology)
    announce(f"Projecting {name}")
    edges = projector.project(dataset.ontology)
    triples = []
    relations = Counter()
    with output.open("w", encoding="utf-8", newline="") as stream:
        for edge in edges:
            triple = f"{edge.src}\t{edge.rel}\t{edge.dst}"
            triples.append(triple)
            relations[str(edge.rel)] += 1
            stream.write(triple + "\n")
    counts = Counter(triples)
    announce(f"Wrote {len(triples)} {name} edges ({len(counts)} unique) to {output}")
    return counts, {
        "source": file_metadata(source),
        "ontology": metadata,
        "edges_total": len(triples),
        "edges_unique": len(counts),
        "duplicate_edges": len(triples) - len(counts),
        "relation_counts": dict(sorted(relations.items())),
        "output": file_metadata(output),
    }


def comparison(existing, projected, existing_lines):
    existing_counts = Counter(existing)
    projected_counts = Counter(projected)
    existing_set = set(existing_counts)
    projected_set = set(projected_counts)
    return {
        "exact_set_equality": existing_set == projected_set,
        "intersection": len(existing_set & projected_set),
        "only_in_existing": len(existing_set - projected_set),
        "only_in_new": len(projected_set - existing_set),
        "exact_multiset_equality": existing_counts == projected_counts,
        "row_count_intersection": sum((existing_counts & projected_counts).values()),
        "row_count_only_in_existing": sum((existing_counts - projected_counts).values()),
        "row_count_only_in_new": sum((projected_counts - existing_counts).values()),
        "existing_file_line_count": existing_lines,
        "existing_unique_triples": len(existing_set),
    }


def main(argv=None):
    args = parse_args(argv)
    for path in (args.go, args.go_plus, args.existing, args.warmup):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.output_dir}")

    import mowl

    announce(f"Starting mOWL JVM with {args.memory} heap")
    mowl.init_jvm(args.memory)
    from mowl.datasets import PathDataset
    from mowl.projection import OWL2VecStarProjector

    args.output_dir.mkdir(parents=True)

    def new_projector():
        return OWL2VecStarProjector(bidirectional_taxonomy=True)

    projector = new_projector()
    existing, existing_lines = read_existing(args.existing)
    projection_order = (["warmup"] if args.warmup else []) + ["go", "go_plus"]
    report = {
        "settings": {
            "projector": "OWL2VecStarProjector",
            "bidirectional_taxonomy": True,
            "only_taxonomy": False,
            "include_literals": False,
            "jvm_memory": args.memory,
            "projection_order": projection_order,
            "fresh_per_ontology": args.fresh_per_ontology,
            "projector_reset_after_warmup": bool(
                args.warmup and args.fresh_per_ontology
            ),
        },
        "existing": {
            **file_metadata(args.existing),
            "line_count": existing_lines,
            "unique_triples": len(existing),
            "expected_line_count": 219802,
            "matches_expected_line_count": existing_lines == 219802,
        },
        "projections": {},
    }
    try:
        report["settings"]["mowl_version"] = version("mowl-borg")
    except PackageNotFoundError:
        try:
            report["settings"]["mowl_version"] = version("mowl")
        except PackageNotFoundError:
            report["settings"]["mowl_version"] = getattr(mowl, "__version__", None)

    if args.warmup:
        triples, details = project(
            "warmup", args.warmup, args.output_dir / "warmup_edges.tsv",
            PathDataset, projector
        )
        report["projections"]["warmup"] = details

    for name, source, filename in (
        ("go", args.go, "go_edges.tsv"),
        ("go_plus", args.go_plus, "go_plus_edges.tsv"),
    ):
        if args.fresh_per_ontology:
            projector = new_projector()
        triples, details = project(
            name, source, args.output_dir / filename, PathDataset, projector
        )
        details["comparison_with_existing"] = comparison(
            existing, triples, existing_lines
        )
        report["projections"][name] = details

    report_path = args.output_dir / "audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    announce(f"Wrote audit report to {report_path}")


if __name__ == "__main__":
    main()
