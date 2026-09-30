"""Turn a --dump_triples export into an ULTRA transductive dataset directory.

ULTRA reads a dataset as <root>/<name>/raw/{train,valid,test}.txt with
head<sep>relation<sep>tail per line, and caches a processed tensor next to it.

Three properties of ULTRA's loader drive what this script writes:

* Inverse edges are added by the loader itself, so the exported triples go in as
  they are. Pre-adding inverses would double every relation a second time.
* The relation count for the whole dataset is taken from the *test* file
  (`num_relations = test_results["num_relation"]`, datasets.py), and the entity
  count from the test file too. Anything absent there is absent from the model's
  relation graph, so valid.txt and test.txt are not empty placeholders: they carry
  one triple of every relation, drawn from train.txt, which fixes the relation
  count at the true value and introduces no node the training graph lacks.
* The processed cache is keyed only by directory, so a stale processed/ would be
  reused for a regenerated graph. It is deleted here whenever raw files are written.

Zero-shot scoring never reads valid.txt or test.txt as evaluation targets; the
scoring driver ranks candidates against the train graph only. The two files exist
purely to make the loader's vocabulary correct.
"""
import os
import shutil

import click as ck


def first_triple_per_relation(path):
    """One triple per relation, in order of first appearance in the dump."""
    seen = {}
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"malformed line in {path}: {line!r}")
            if parts[1] not in seen:
                seen[parts[1]] = line
    return list(seen.values())


@ck.command()
@ck.option("--dump", required=True, help="Triple file written by kge_transd.py --dump_triples")
@ck.option("--root", required=True, help="ULTRA dataset root directory")
@ck.option("--name", required=True, help="Dataset name; becomes <root>/<name>")
def main(dump, root, name):
    raw_dir = os.path.join(root, name, "raw")
    processed_dir = os.path.join(root, name, "processed")
    os.makedirs(raw_dir, exist_ok=True)

    train_path = os.path.join(raw_dir, "train.txt")
    shutil.copyfile(dump, train_path)

    probes = first_triple_per_relation(train_path)
    for split in ("valid.txt", "test.txt"):
        with open(os.path.join(raw_dir, split), "w") as handle:
            handle.writelines(probes)

    if os.path.isdir(processed_dir):
        shutil.rmtree(processed_dir)

    num_train = sum(1 for _ in open(train_path))
    print(f"{name}: {num_train} train triples, {len(probes)} relations, "
          f"raw at {raw_dir}, processed cache cleared")


if __name__ == "__main__":
    main()
