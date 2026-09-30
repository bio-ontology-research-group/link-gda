"""Project two GO snapshots and audit their edge sets against an existing TSV."""

import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", required=True, type=Path, help="Path to go.owl")
    parser.add_argument("--go-plus", required=True, type=Path, help="Path to go-plus.owl")
    parser.add_argument("--existing", required=True, type=Path, help="Existing edge TSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--memory", default="10g", help="JVM maximum heap (default: 10g)")
    return parser.parse_args()


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
    triples = set()
    lines = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            lines += 1
            triples.add(line.rstrip("\r\n"))
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
    unique = set(triples)
    announce(f"Wrote {len(triples)} {name} edges ({len(unique)} unique) to {output}")
    return unique, {
        "source": file_metadata(source),
        "ontology": metadata,
        "edges_total": len(triples),
        "edges_unique": len(unique),
        "duplicate_edges": len(triples) - len(unique),
        "relation_counts": dict(sorted(relations.items())),
        "output": file_metadata(output),
    }


def comparison(existing, projected, existing_lines):
    return {
        "exact_set_equality": existing == projected,
        "intersection": len(existing & projected),
        "only_in_existing": len(existing - projected),
        "only_in_new": len(projected - existing),
        "existing_file_line_count": existing_lines,
        "existing_unique_triples": len(existing),
    }


def main():
    args = parse_args()
    for path in (args.go, args.go_plus, args.existing):
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
    projector = OWL2VecStarProjector(bidirectional_taxonomy=True)
    existing, existing_lines = read_existing(args.existing)
    report = {
        "settings": {
            "projector": "OWL2VecStarProjector",
            "bidirectional_taxonomy": True,
            "only_taxonomy": False,
            "include_literals": False,
            "jvm_memory": args.memory,
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

    for name, source, filename in (
        ("go", args.go, "go_edges.tsv"),
        ("go_plus", args.go_plus, "go_plus_edges.tsv"),
    ):
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
