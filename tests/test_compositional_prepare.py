import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

PREPARE = Path(__file__).resolve().parents[1] / "code" / "compositional_gda" / "prepare.py"
spec = importlib.util.spec_from_file_location("compositional_prepare_under_test", PREPARE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

B = "http://mowl.borg/"
O = "http://purl.obolibrary.org/obo/"
G1, G2, G3, G4, G5 = (B + str(i) for i in (11, 22, 33, 44, 55))
D_TRAIN, D_TRAIN2, D_VAL, D_TEST, D_OTHER = (B + n for n in ("OMIM_1", "OMIM_2", "OMIM_3", "OMIM_4", "OMIM_9"))
HP = {i: O + f"HP_{i:07d}" for i in range(1, 8)}
MP1, GO1, GO2, UB1 = O + "MP_0000001", O + "GO_0000001", O + "GO_0000002", O + "UBERON_0000001"

CANDIDATES = [G3, G1, G5, G2, G4]
TRAINING = [(G1, D_TRAIN), (G2, D_TRAIN), (G2, D_TRAIN2)]
VALIDATION = [(G3, D_VAL)]
TEST = [(G4, D_TEST)]
PROFILES = {
    D_TRAIN: [HP[1], HP[2], HP[2]],
    D_TRAIN2: [HP[3]],
    D_VAL: [HP[4], HP[1]],
    D_TEST: [HP[5], HP[7], HP[7]],
}


def native_triples():
    rows = [
        (G1, "has_phenotype", MP1),
        (G2, "has_function", GO1),
        (G5, "expressed_in", UB1),
        (GO1, "http://subclassof", GO2),
        (GO2, "http://superclassof", GO1),
        (HP[2], "http://subclassof", HP[1]),
        (HP[1], "http://superclassof", HP[2]),
        (HP[4], "http://subclassof", HP[1]),
        (HP[1], "http://superclassof", HP[4]),
        (HP[5], "http://subclassof", HP[3]),
        (HP[3], "http://superclassof", HP[5]),
        (D_OTHER, "has_symptom", HP[6]),
        (G1, "has_function", GO1),
        (G1, "has_function", GO1),
    ]
    for gene, disease in TRAINING:
        rows.append((gene, "associated_with", disease))
    for disease in (D_TRAIN, D_TRAIN2):
        for phenotype in PROFILES[disease]:
            rows.append((disease, "has_symptom", phenotype))
    for gene, disease in TRAINING:
        for phenotype in PROFILES[disease]:
            rows.append((gene, "causes_phenotype", phenotype))
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_collection(root, fold=0, raw=None, profiles=None):
    data = root / "benchmark" / "data" / f"fold{fold}"
    (data / "raw").mkdir(parents=True)
    raw = native_triples() if raw is None else raw
    (data / "raw" / "train.txt").write_text("".join("\t".join(t) + "\n" for t in raw))
    (data / "profiles.json").write_text(json.dumps(PROFILES if profiles is None else profiles))
    (data / "candidates.txt").write_text("\n".join(CANDIDATES) + "\n")
    for name, rows in (("training.tsv", TRAINING), ("validation.tsv", VALIDATION), ("test.tsv", TEST)):
        (data / name).write_text("Gene\tDisease\n" + "".join(f"{g}\t{d}\n" for g, d in rows))
    write_preparation_and_manifest(root, fold)
    return data


def write_preparation_and_manifest(root, fold=0):
    data = root / "benchmark" / "data" / f"fold{fold}"
    fields = {"raw/train.txt": "graph_sha256", "profiles.json": "profiles_sha256", "candidates.txt": "candidates_sha256", "training.tsv": "training_sha256", "validation.tsv": "validation_sha256", "test.tsv": "test_sha256"}
    preparation = {"fold": fold}
    preparation.update({field: digest(data / name) for name, field in fields.items()})
    (data / "preparation.json").write_text(json.dumps(preparation))
    files = {}
    for path in sorted(data.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "mode": 420, "sha256": digest(path)}
    (root / "MANIFESTS").mkdir(exist_ok=True)
    (root / "MANIFESTS" / f"benchmark-fold{fold}.json").write_text(json.dumps({"component": f"benchmark-fold{fold}", "files": files, "schema": 1}))


def load_graph(target):
    with np.load(target / "graph.npz", allow_pickle=False) as archive:
        return archive["triples"]


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.inputs = self.tmp / "inputs"
        self.inputs.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def prepared(self, output="out"):
        status, target = prepare.prepare_fold(self.inputs, self.tmp / output, 0)
        return status, target

    def named(self, target):
        vocabulary = json.loads((target / "vocabulary.json").read_text())
        entities = {i: n for n, i in vocabulary["entities"].items()}
        relations = {i: n for n, i in vocabulary["relations"].items()}
        return vocabulary, {(entities[h], relations[r], entities[t]) for h, r, t in load_graph(target).tolist()}

    def test_graph_content_and_interface(self):
        write_collection(self.inputs)
        status, target = self.prepared()
        self.assertEqual(status, "prepared")
        triples = load_graph(target)
        self.assertEqual(triples.dtype, np.int64)
        self.assertEqual(triples.ndim, 2)
        self.assertEqual(triples.shape[1], 3)
        vocabulary, named = self.named(target)
        self.assertEqual(set(vocabulary), {"entities", "relations"})
        relation_names = set(vocabulary["relations"])
        self.assertNotIn("causes_phenotype", relation_names)
        self.assertNotIn("inverse:causes_phenotype", relation_names)
        for name in ("associated_with", "has_symptom", "inverse:associated_with", "inverse:has_symptom", "has_phenotype", "expressed_in", "has_function"):
            self.assertIn(name, relation_names)
        forward = {t for t in named if not t[1].startswith("inverse:")}
        inverse = {t for t in named if t[1].startswith("inverse:")}
        self.assertEqual(inverse, {(t, "inverse:" + r, h) for h, r, t in forward})
        self.assertEqual({(h, t) for h, r, t in forward if r == "associated_with"}, set(TRAINING))
        self.assertEqual({(h, t) for h, r, t in forward if r == "has_symptom"}, {(d, p) for d in (D_TRAIN, D_TRAIN2) for p in PROFILES[d]})
        self.assertFalse(any(r == "has_symptom" and h == D_OTHER for h, r, t in forward))
        self.assertFalse(any(prepare.is_gene(h) and t.startswith(prepare.HP_PREFIX) for h, r, t in forward))
        for disease in (D_VAL, D_TEST):
            self.assertNotIn(disease, vocabulary["entities"])
        self.assertTrue(set(CANDIDATES) <= set(vocabulary["entities"]))
        self.assertTrue({p for v in PROFILES.values() for p in v} <= set(vocabulary["entities"]))
        self.assertEqual(sorted(vocabulary["entities"].values()), list(range(len(vocabulary["entities"]))))
        self.assertEqual(sorted(vocabulary["relations"].values()), list(range(len(vocabulary["relations"]))))
        offset = len(relation_names) // 2
        for name, index in vocabulary["relations"].items():
            if not name.startswith("inverse:"):
                self.assertEqual(vocabulary["relations"]["inverse:" + name], index + offset)

    def test_split_files_copied_byte_for_byte(self):
        source = write_collection(self.inputs)
        _, target = self.prepared()
        for name in ("candidates.txt", "training.tsv", "validation.tsv", "test.tsv", "profiles.json"):
            self.assertEqual((target / name).read_bytes(), (source / name).read_bytes())
        self.assertEqual((target / "candidates.txt").read_text().splitlines(), CANDIDATES)
        self.assertEqual(json.loads((target / "profiles.json").read_text())[D_TRAIN], [HP[1], HP[2], HP[2]])

    def test_identity_records_outputs_inputs_and_counts(self):
        write_collection(self.inputs)
        _, target = self.prepared()
        identity = json.loads((target / "identity.json").read_text())
        for name in prepare.OUTPUTS:
            self.assertEqual(identity["outputs"][name]["sha256"], digest(target / name))
        self.assertEqual(set(identity["inputs"]["files"]), set(prepare.USED))
        self.assertEqual(identity["source"]["sha256"], digest(PREPARE))
        audit = identity["audit"]
        self.assertEqual(audit["removed"]["causes_phenotype"], len({(g, p) for g, d in TRAINING for p in PROFILES[d]}))
        self.assertEqual(audit["removed"]["native_has_symptom_non_split_diseases_dropped"], 1)
        self.assertEqual(audit["removed"]["non_split_diseases_dropped"], 1)
        self.assertEqual(audit["inverse_triples"], audit["forward_triples"])
        self.assertEqual(audit["graph_triples"], 2 * audit["forward_triples"])
        self.assertEqual(audit["isolated_profile_phenotypes"], 1)
        self.assertEqual(audit["profile_duplicate_entries"], 2)
        self.assertIn(["http://subclassof", "http://superclassof"], audit["mirrored_native_relation_pairs"])

    def test_rerun_is_idempotent_and_bytes_are_deterministic(self):
        write_collection(self.inputs)
        _, first = self.prepared("a")
        before = {p.name: digest(p) for p in first.iterdir()}
        status, again = self.prepared("a")
        self.assertEqual(status, "unchanged")
        self.assertEqual({p.name: digest(p) for p in again.iterdir()}, before)
        _, second = self.prepared("b")
        self.assertEqual({p.name: digest(p) for p in second.iterdir()}, before)
        self.assertEqual([p.name for p in (self.tmp / "a").iterdir()], ["fold0"])

    def test_refuses_existing_output_from_other_inputs(self):
        data = write_collection(self.inputs)
        self.prepared()
        with (data / "raw" / "train.txt").open("a") as handle:
            handle.write(f"{GO2}\thttp://subclassof\t{O}GO_0000003\n")
        write_preparation_and_manifest(self.inputs)
        with self.assertRaisesRegex(prepare.PreparationError, "different inputs"):
            self.prepared()

    def test_refuses_tampered_existing_output(self):
        write_collection(self.inputs)
        _, target = self.prepared()
        (target / "candidates.txt").write_text("\n".join(reversed(CANDIDATES)) + "\n")
        with self.assertRaisesRegex(prepare.PreparationError, "does not match"):
            self.prepared()

    def test_refuses_checksum_mismatch(self):
        data = write_collection(self.inputs)
        (data / "candidates.txt").write_text("\n".join(reversed(CANDIDATES)) + "\n")
        with self.assertRaisesRegex(prepare.PreparationError, "SHA-256 mismatch"):
            self.prepared()
        self.assertFalse((self.tmp / "out" / "fold0").exists())

    def test_refuses_held_out_disease_in_raw_graph(self):
        for disease in (D_VAL, D_TEST):
            with self.subTest(disease=disease):
                shutil.rmtree(self.inputs)
                self.inputs.mkdir()
                write_collection(self.inputs, raw=native_triples() + [(disease, "has_symptom", HP[1])])
                with self.assertRaisesRegex(prepare.PreparationError, "held-out disease"):
                    self.prepared()
                self.assertFalse(any((self.tmp / "out").iterdir()))

    def test_refuses_shortcut_not_matching_training_join(self):
        write_collection(self.inputs, raw=native_triples() + [(G3, "causes_phenotype", HP[4])])
        with self.assertRaisesRegex(prepare.PreparationError, "causes_phenotype"):
            self.prepared()

    def test_refuses_native_gene_to_hp_edge(self):
        write_collection(self.inputs, raw=native_triples() + [(G3, "has_phenotype", HP[4])])
        with self.assertRaisesRegex(prepare.PreparationError, "gene to HP"):
            self.prepared()

    def test_refuses_profile_disagreeing_with_native_symptoms(self):
        profiles = dict(PROFILES)
        profiles[D_TRAIN] = [HP[1], HP[2]]
        rows = [t for t in native_triples() if not (t[1] == "causes_phenotype")]
        rows += [(g, "causes_phenotype", p) for g, d in TRAINING for p in profiles[d]]
        write_collection(self.inputs, raw=rows, profiles=profiles)
        with self.assertRaisesRegex(prepare.PreparationError, "multisets differ"):
            self.prepared()

    def test_refuses_preexisting_inverse_relation_name(self):
        write_collection(self.inputs, raw=native_triples() + [(GO1, "inverse:part_of", GO2)])
        with self.assertRaisesRegex(prepare.PreparationError, "inverse:"):
            self.prepared()

    def test_cli(self):
        write_collection(self.inputs)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = prepare.main(["--inputs", str(self.inputs), "--output", str(self.tmp / "cli"), "--folds", "0"])
        self.assertEqual(code, 0)
        record = json.loads(stdout.getvalue().splitlines()[0])
        self.assertEqual(record["status"], "prepared")
        self.assertEqual(record["graph_sha256"], digest(self.tmp / "cli" / "fold0" / "graph.npz"))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = prepare.main(["--inputs", str(self.inputs), "--output", str(self.tmp / "cli"), "--folds", "1"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "refused")


if __name__ == "__main__":
    unittest.main()
