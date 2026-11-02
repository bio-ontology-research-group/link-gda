"""Prepare Track A one-hop training graphs from the frozen benchmark folds.

Each prepared fold directory holds graph.npz, vocabulary.json, profiles.json,
candidates.txt, training.tsv, validation.tsv, test.tsv and identity.json.
"""

import argparse
import collections
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from array import array
from pathlib import Path

import numpy as np

SCHEMA = 1
INVERSE_PREFIX = "inverse:"
ASSOCIATED = "associated_with"
SYMPTOM = "has_symptom"
SHORTCUT = "causes_phenotype"
GENE_PREFIX = "http://mowl.borg/"
HP_PREFIX = "http://purl.obolibrary.org/obo/HP_"
COPIED = ("profiles.json", "candidates.txt", "training.tsv", "validation.tsv", "test.tsv")
USED = ("raw/train.txt", "preparation.json") + COPIED
PREPARATION_KEYS = {
    "raw/train.txt": "graph_sha256",
    "profiles.json": "profiles_sha256",
    "candidates.txt": "candidates_sha256",
    "training.tsv": "training_sha256",
    "validation.tsv": "validation_sha256",
    "test.tsv": "test_sha256",
}
OUTPUTS = ("graph.npz", "vocabulary.json") + COPIED
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class PreparationError(RuntimeError):
    """Raised when inputs are missing, altered or ambiguous."""


def sha256(path):
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_gene(name):
    """True for an Entrez gene IRI of the benchmark namespace."""
    return name.startswith(GENE_PREFIX) and name[len(GENE_PREFIX):].isdigit()


def is_disease(name):
    """True for a non-gene IRI of the benchmark namespace."""
    return name.startswith(GENE_PREFIX) and not name[len(GENE_PREFIX):].isdigit()


def audit_inputs(inputs, fold):
    """Verify every used fold file against MANIFESTS/benchmark-foldN.json."""
    inputs = Path(inputs)
    manifest_path = inputs / "MANIFESTS" / f"benchmark-fold{fold}.json"
    if not manifest_path.is_file():
        raise PreparationError(f"missing manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("component") != f"benchmark-fold{fold}":
        raise PreparationError(f"manifest component mismatch in {manifest_path}")
    files = manifest.get("files", {})
    audited = {}
    for name in USED:
        key = f"benchmark/data/fold{fold}/{name}"
        if key not in files:
            raise PreparationError(f"{key} absent from manifest")
        path = inputs / key
        if not path.is_file():
            raise PreparationError(f"missing input {path}")
        size = path.stat().st_size
        if size != files[key]["bytes"]:
            raise PreparationError(f"byte count mismatch for {key}")
        digest = sha256(path)
        if digest != files[key]["sha256"]:
            raise PreparationError(f"SHA-256 mismatch for {key}")
        audited[name] = {"path": key, "bytes": size, "sha256": digest}
    preparation = json.loads((inputs / f"benchmark/data/fold{fold}/preparation.json").read_text())
    if preparation.get("fold") != fold:
        raise PreparationError("preparation.json fold mismatch")
    for name, field in PREPARATION_KEYS.items():
        if preparation.get(field) != audited[name]["sha256"]:
            raise PreparationError(f"preparation.json {field} disagrees with manifest")
    return {
        "manifest": {"path": f"MANIFESTS/benchmark-fold{fold}.json", "sha256": sha256(manifest_path)},
        "files": audited,
    }


def read_pairs(path):
    """Read a Gene/Disease TSV, refusing malformed or duplicate rows."""
    rows = []
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header != ["Gene", "Disease"]:
            raise PreparationError(f"unexpected header in {path}: {header}")
        for number, line in enumerate(handle, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or not is_gene(fields[0]) or not is_disease(fields[1]):
                raise PreparationError(f"malformed row {number} in {path}")
            rows.append((fields[0], fields[1]))
    if len(set(rows)) != len(rows):
        raise PreparationError(f"duplicate rows in {path}")
    return rows


def write_npz(path, triples):
    """Write triples to a byte-deterministic uncompressed npz archive."""
    info = zipfile.ZipInfo("triples.npy", date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as archive:
        with archive.open(info, "w", force_zip64=True) as handle:
            np.lib.format.write_array(handle, np.ascontiguousarray(triples, dtype=np.int64), allow_pickle=False)


def mirrored_relations(forward, relation_names):
    """List forward relation pairs whose edge sets are exact reverses of each other."""
    if len(forward) == 0:
        return []
    order = np.argsort(forward[:, 1], kind="stable")
    ordered = forward[order]
    bounds = np.flatnonzero(np.diff(ordered[:, 1])) + 1
    groups = np.split(ordered, bounds)
    signatures = {}
    reversed_signatures = {}
    for group in groups:
        relation = int(group[0, 1])
        pairs = group[:, [0, 2]]
        straight = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        flipped = pairs[:, ::-1]
        flipped = flipped[np.lexsort((flipped[:, 1], flipped[:, 0]))]
        signatures[relation] = hashlib.sha256(np.ascontiguousarray(straight).tobytes()).hexdigest()
        reversed_signatures[relation] = hashlib.sha256(np.ascontiguousarray(flipped).tobytes()).hexdigest()
    by_signature = collections.defaultdict(list)
    for relation, signature in signatures.items():
        by_signature[signature].append(relation)
    found = set()
    for relation, signature in reversed_signatures.items():
        for other in by_signature.get(signature, []):
            found.add(tuple(sorted((relation_names[relation], relation_names[other]))))
    return sorted([list(pair) for pair in found])


def build(inputs, fold):
    """Build the fold graph in memory and return arrays, vocabulary and audit counts."""
    root = Path(inputs) / "benchmark" / "data" / f"fold{fold}"
    training = read_pairs(root / "training.tsv")
    validation = read_pairs(root / "validation.tsv")
    test = read_pairs(root / "test.tsv")
    training_diseases = {d for _, d in training}
    validation_diseases = {d for _, d in validation}
    test_diseases = {d for _, d in test}
    if training_diseases & validation_diseases or training_diseases & test_diseases or validation_diseases & test_diseases:
        raise PreparationError("split diseases are not disjoint")
    held_out = validation_diseases | test_diseases
    profiles = json.loads((root / "profiles.json").read_text())
    if not isinstance(profiles, dict) or not all(isinstance(v, list) and all(isinstance(p, str) for p in v) for v in profiles.values()):
        raise PreparationError("profiles.json is not a disease to phenotype-list mapping")
    missing_profiles = (training_diseases | held_out) - profiles.keys()
    if missing_profiles:
        raise PreparationError(f"{len(missing_profiles)} split diseases lack profiles")
    empty_training = sorted(d for d in training_diseases if not profiles[d])
    if empty_training:
        raise PreparationError(f"{len(empty_training)} training diseases have empty profiles")
    candidates = (root / "candidates.txt").read_text().splitlines()
    if len(set(candidates)) != len(candidates) or not all(is_gene(g) for g in candidates):
        raise PreparationError("candidates.txt has duplicates or non-gene entries")
    candidate_set = set(candidates)
    absent = {g for g, _ in training + validation + test} - candidate_set
    if absent:
        raise PreparationError(f"{len(absent)} split genes are not candidates")

    entity_index = {}
    relation_index = {}
    heads, relations, tails = array("q"), array("q"), array("q")
    raw_counts = collections.Counter()
    raw_shortcuts = set()
    raw_associated = set()
    raw_symptoms = collections.defaultdict(collections.Counter)
    dropped_symptom_diseases = set()
    dropped_symptom_triples = 0
    raw_lines = 0
    with open(root / "raw" / "train.txt") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3 or not all(fields):
                raise PreparationError(f"malformed raw triple at line {number}")
            head, relation, tail = fields
            raw_lines += 1
            raw_counts[relation] += 1
            if relation.startswith(INVERSE_PREFIX):
                raise PreparationError(f"raw relation already uses the {INVERSE_PREFIX} prefix: {relation}")
            if head in held_out or tail in held_out:
                raise PreparationError(f"held-out disease in raw graph at line {number}")
            if relation == SHORTCUT:
                if not is_gene(head) or not tail.startswith(HP_PREFIX):
                    raise PreparationError(f"unexpected {SHORTCUT} triple at line {number}")
                raw_shortcuts.add((head, tail))
                continue
            if relation == ASSOCIATED:
                if not is_gene(head) or head not in candidate_set or tail not in training_diseases:
                    raise PreparationError(f"unexpected {ASSOCIATED} triple at line {number}")
                raw_associated.add((head, tail))
                continue
            if relation == SYMPTOM:
                if not is_disease(head) or not tail.startswith(HP_PREFIX):
                    raise PreparationError(f"unexpected {SYMPTOM} triple at line {number}")
                if head in training_diseases:
                    raw_symptoms[head][tail] += 1
                else:
                    dropped_symptom_diseases.add(head)
                    dropped_symptom_triples += 1
                continue
            if is_disease(head) or is_disease(tail):
                raise PreparationError(f"disease entity under native relation {relation} at line {number}")
            if (is_gene(head) and tail.startswith(HP_PREFIX)) or (is_gene(tail) and head.startswith(HP_PREFIX)):
                raise PreparationError(f"gene to HP edge under native relation {relation} at line {number}")
            heads.append(entity_index.setdefault(head, len(entity_index)))
            relations.append(relation_index.setdefault(relation, len(relation_index)))
            tails.append(entity_index.setdefault(tail, len(entity_index)))

    expected_shortcuts = {(g, p) for g, d in training for p in profiles[d]}
    if raw_shortcuts != expected_shortcuts:
        raise PreparationError(f"{SHORTCUT} triples differ from the training GDA and profile join")
    if raw_associated != set(training):
        raise PreparationError(f"raw {ASSOCIATED} triples differ from training.tsv")
    symptom_mismatch = sorted(d for d in training_diseases if raw_symptoms.get(d, collections.Counter()) != collections.Counter(profiles[d]))
    if symptom_mismatch:
        raise PreparationError(f"raw {SYMPTOM} multisets differ from profiles for {len(symptom_mismatch)} training diseases")

    native_names = sorted(relation_index)
    for name in (ASSOCIATED, SYMPTOM):
        if name in relation_index:
            raise PreparationError(f"{name} already present as a native relation")
    for gene, disease in sorted(set(training)):
        heads.append(entity_index.setdefault(gene, len(entity_index)))
        relations.append(relation_index.setdefault(ASSOCIATED, len(relation_index)))
        tails.append(entity_index.setdefault(disease, len(entity_index)))
    symptom_entries = 0
    for disease in sorted(training_diseases):
        for phenotype in profiles[disease]:
            symptom_entries += 1
            heads.append(entity_index.setdefault(disease, len(entity_index)))
            relations.append(relation_index.setdefault(SYMPTOM, len(relation_index)))
            tails.append(entity_index.setdefault(phenotype, len(entity_index)))

    graph_entities = set(entity_index)
    profile_phenotypes = {p for values in profiles.values() for p in values}
    isolated_phenotypes = sorted(profile_phenotypes - graph_entities)
    if any(not p.startswith(HP_PREFIX) for p in isolated_phenotypes):
        raise PreparationError("profile phenotype outside HP and absent from the graph")
    isolated_candidates = sorted(candidate_set - graph_entities)
    names = sorted(graph_entities | profile_phenotypes | candidate_set)
    leaked = held_out & set(names)
    if leaked:
        raise PreparationError(f"{len(leaked)} held-out diseases in vocabulary")
    entities = {name: i for i, name in enumerate(names)}
    forward_names = sorted(relation_index)
    inverse_names = [INVERSE_PREFIX + name for name in forward_names]
    if set(inverse_names) & set(forward_names):
        raise PreparationError("inverse relation names collide with forward names")
    relation_ids = {name: i for i, name in enumerate(forward_names)}
    relation_ids.update({name: i + len(forward_names) for i, name in enumerate(inverse_names)})

    entity_map = np.empty(len(entity_index), dtype=np.int64)
    for name, raw in entity_index.items():
        entity_map[raw] = entities[name]
    relation_map = np.empty(len(relation_index), dtype=np.int64)
    for name, raw in relation_index.items():
        relation_map[raw] = relation_ids[name]
    loaded = np.stack([
        entity_map[np.frombuffer(heads, dtype=np.int64)],
        relation_map[np.frombuffer(relations, dtype=np.int64)],
        entity_map[np.frombuffer(tails, dtype=np.int64)],
    ], axis=1)
    del heads, relations, tails
    forward = np.unique(loaded, axis=0)
    loaded_count = len(loaded)
    del loaded
    inverse = np.stack([forward[:, 2], forward[:, 1] + len(forward_names), forward[:, 0]], axis=1)
    triples = np.unique(np.concatenate([forward, inverse]), axis=0)
    if len(triples) != 2 * len(forward):
        raise PreparationError("inverse triples collide with forward triples")

    if SHORTCUT in relation_ids:
        raise PreparationError(f"{SHORTCUT} survived preparation")

    relation_counts = collections.Counter(int(r) for r in forward[:, 1])
    retained = {forward_names[r]: c for r, c in sorted(relation_counts.items())}
    audit = {
        "raw_lines": raw_lines,
        "raw_relation_counts": dict(sorted(raw_counts.items())),
        "removed": {
            SHORTCUT: len(raw_shortcuts),
            "native_associated_with_replaced_by_training_tsv": len(raw_associated),
            "native_has_symptom_training_replaced_by_profiles": sum(sum(c.values()) for c in raw_symptoms.values()),
            "native_has_symptom_non_split_diseases_dropped": dropped_symptom_triples,
            "non_split_diseases_dropped": len(dropped_symptom_diseases),
        },
        "reconstructed": {
            ASSOCIATED: {"source": "training.tsv", "rows": len(training), "unique_triples": retained.get(ASSOCIATED, 0)},
            SYMPTOM: {"source": "profiles.json restricted to training.tsv diseases", "diseases": len(training_diseases), "profile_entries": symptom_entries, "unique_triples": retained.get(SYMPTOM, 0)},
        },
        "checks": {
            "causes_phenotype_equals_training_profile_join": True,
            "native_associated_with_equals_training_tsv": True,
            "native_has_symptom_equals_training_profiles_as_multisets": True,
            "held_out_diseases_absent_from_graph_and_vocabulary": True,
            "no_gene_to_hp_native_edges": True,
            "validation_diseases": len(validation_diseases),
            "test_diseases": len(test_diseases),
        },
        "loaded_triples_before_deduplication": loaded_count,
        "forward_triples": int(len(forward)),
        "inverse_triples": int(len(inverse)),
        "graph_triples": int(len(triples)),
        "retained_forward_relation_counts": retained,
        "native_relations": len(native_names),
        "forward_relations": len(forward_names),
        "inverse_relation_offset": len(forward_names),
        "entities": len(entities),
        "graph_entities": len(graph_entities),
        "isolated_profile_phenotypes": len(isolated_phenotypes),
        "isolated_candidates": len(isolated_candidates),
        "candidates": len(candidates),
        "profile_diseases": len(profiles),
        "profile_duplicate_entries": sum(len(v) - len(set(v)) for v in profiles.values()),
        "mirrored_native_relation_pairs": mirrored_relations(forward, forward_names),
        "triple_order": "unique rows sorted lexicographically by head, relation, tail",
    }
    return triples, {"entities": entities, "relations": relation_ids}, audit


def identity_key(inputs, fold):
    """Return the fields that determine a fold's prepared bytes."""
    return {
        "schema": SCHEMA,
        "fold": fold,
        "source": {"path": "code/compositional_gda/prepare.py", "sha256": sha256(Path(__file__).resolve())},
        "inputs": audit_inputs(inputs, fold),
    }


def verify_existing(target, key):
    """Confirm an existing prepared fold matches the identity key and its recorded outputs."""
    identity_path = target / "identity.json"
    if not identity_path.is_file():
        raise PreparationError(f"{target} exists without identity.json; refusing to overwrite")
    identity = json.loads(identity_path.read_text())
    if {k: identity.get(k) for k in key} != key:
        raise PreparationError(f"{target} was prepared from different inputs or source; refusing to overwrite")
    for name, record in identity["outputs"].items():
        path = target / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise PreparationError(f"{path} does not match its identity record")
    if set(identity["outputs"]) != set(OUTPUTS):
        raise PreparationError(f"{identity_path} lists unexpected outputs")
    return identity


def prepare_fold(inputs, output, fold):
    """Prepare one fold under OUTPUT/foldN, or verify an identical existing preparation."""
    inputs = Path(inputs)
    output = Path(output)
    key = identity_key(inputs, fold)
    target = output / f"fold{fold}"
    if target.exists():
        verify_existing(target, key)
        return "unchanged", target
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".fold{fold}.partial-", dir=output))
    try:
        triples, vocabulary, audit = build(inputs, fold)
        if identity_key(inputs, fold) != key:
            raise PreparationError("inputs changed during preparation")
        write_npz(staging / "graph.npz", triples)
        (staging / "vocabulary.json").write_text(json.dumps(vocabulary, separators=(",", ":")) + "\n")
        source = inputs / "benchmark" / "data" / f"fold{fold}"
        for name in COPIED:
            shutil.copyfile(source / name, staging / name)
            if sha256(staging / name) != key["inputs"]["files"][name]["sha256"]:
                raise PreparationError(f"copied {name} differs from audited input")
        identity = dict(key)
        identity["outputs"] = {name: {"bytes": (staging / name).stat().st_size, "sha256": sha256(staging / name)} for name in OUTPUTS}
        identity["graph"] = {"file": "graph.npz", "array": "triples", "dtype": "int64", "shape": [int(triples.shape[0]), 3], "columns": ["head", "relation", "tail"], "includes_inverses": True, "inverse_prefix": INVERSE_PREFIX}
        identity["audit"] = audit
        identity["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
        (staging / "identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        os.chmod(staging, 0o755)
        try:
            os.rename(staging, target)
        except OSError:
            if target.exists():
                verify_existing(target, key)
                shutil.rmtree(staging, ignore_errors=True)
                return "unchanged", target
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return "prepared", target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    if len(set(args.folds)) != len(args.folds) or min(args.folds) < 0 or max(args.folds) > 9:
        parser.error("choose distinct fold IDs in 0..9")
    for fold in args.folds:
        try:
            status, target = prepare_fold(args.inputs, args.output, fold)
        except PreparationError as error:
            print(json.dumps({"fold": fold, "status": "refused", "reason": str(error)}), flush=True)
            return 2
        identity = json.loads((target / "identity.json").read_text())
        print(json.dumps({"fold": fold, "status": status, "path": str(target), "graph_sha256": identity["outputs"]["graph.npz"]["sha256"], "graph_triples": identity["audit"]["graph_triples"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
