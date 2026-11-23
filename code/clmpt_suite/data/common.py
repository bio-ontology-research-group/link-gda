"""Shared helpers and pinned identities for CLMPT suite data preparation."""
import hashlib
import json
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]

FOLDS = tuple(range(10))
MODALITIES = ('P', 'F', 'S', 'PF', 'PS', 'FS', 'PFS')
NATIVE_RELATION = {'P': 'has_phenotype', 'F': 'has_function', 'S': 'expressed_in'}
CLINICAL_RELATIONS = ('associated_with', 'has_symptom')
LEGACY_SHORTCUT = 'causes_phenotype'
GENE_PREFIX = 'http://mowl.borg/'
DISEASE_PREFIX = 'http://mowl.borg/OMIM_'
HP_PREFIX = 'http://purl.obolibrary.org/obo/HP_'
MP_PREFIX = 'http://purl.obolibrary.org/obo/MP_'

CANDIDATES_SHA256 = '9960b9e08880114fe17293fd7ab3d5a73ecc7631c0ea8b36c4e210f3bb652ff1'
CANDIDATE_COUNT = 13625
REPRESENTED_CLASSES = 175906
ENTITY_COUNT = 239552
NON_DISEASE_ENTITIES = 234644
RELATION_COUNT = 624
FOLD0_TRIPLES = 4222290
PATTERN_COUNT = 13

FOLD_BUILDER = 'code/clmpt_suite/data/fold_builder.py'
FOLD_BUILDER_SHA256 = 'befaaea690b0ee91c7531b34e8cd1ce6a9c245f7474e3874d5018a799f38c13e'
PREPARE_SOURCE = 'code/compositional_gda/prepare.py'
PREPARE_SHA256 = '0a77cf2f819d830c5ad46d86127f1ff38a3759dfcc910d6ec3bf9cf0035329a3'
BENCHMARK_INPUTS = 'data/reproduction-20261102/collections/inputs-20261028-v1'
IRI_BANK = 'data/query-bank-20261112/upheno-shared-existing-folds-v1'
IRI_BANK_COMPLETE = '4ebbee88742472a6ffa866a71903305d4434fdbd3e480c709fa5cb2451d0cc43'
CLASS_POOL = 'data/query-training-20261112/represented-class-pool.json'
CLASS_POOL_SHA256 = 'd6813b738c6baaa77e9595783668f8cf5e4c01f70d833f5d0fb01dbb5ed168f0'

EXISTING_BASES = {
    0: {'graph': ('data/query-training-20261112/fold0-vocab-v1', 'af68a9a3230fc1f05a105e1d57f02200129879c0bea5097c9094cf0feb0201ed'),
        'bank': ('data/clmpt-next-study-20261113/packet-v1/inputs/bank-original', '8f34c9d29efa42d07353d6b51ee190b24901da9f7ba738368c5d75543c5521ec'),
        'test': ('data/clmpt-threefold-pilot-20261116/inputs/fold0/test-fold0', '3ee0c0824899fe425daff96254b8514dbe2b67be7f6d4bf3e6aff5a8d8a3d3ca')},
    1: {'graph': ('data/clmpt-threefold-pilot-20261116/inputs/fold1/graph-fold1', 'f47334a5f447cf5a5e5412782ff7df2f25d600ce8b6ff808ee1e5af454ae304a'),
        'bank': ('data/clmpt-threefold-pilot-20261116/inputs/fold1/bank-fold1', '3e1081279c1f7fe901467575489591c5e72803fadbab582b6aaf435d2e081c6c'),
        'test': ('data/clmpt-threefold-pilot-20261116/inputs/fold1/test-fold1', '4ff896de0b31373580235654b2d8e0b3e30e67526344085702c28fec5e25c6d9')},
    2: {'graph': ('data/clmpt-threefold-pilot-20261116/inputs/fold2/graph-fold2', '83ec8a7ae17897777cb18e4169c4d4ff840ec907784b0bfe81ca56a38cec467b'),
        'bank': ('data/clmpt-threefold-pilot-20261116/inputs/fold2/bank-fold2', '998e2d35bd59aeccff1db054c5206e7d6093074a0f0211e96cbc1870b508056f'),
        'test': ('data/clmpt-threefold-pilot-20261116/inputs/fold2/test-fold2', '315f21fd732a8ad2b77ef2e2d3df16a5cd8545ef94252a4ee835c8656a46d88a')},
}

EXPECTED_FALLBACK_GENES = {'F': 418, 'S': 1175, 'FS': 257}
EXPECTED_FALLBACK_TEST_PAIRS = {'F': 29, 'S': 426, 'FS': 15}
EXPECTED_OUTER_TEST_PAIRS = 6571


class Refusal(RuntimeError):
    """Raised when an input identity, count or invariant does not hold."""


def check(condition, message):
    """Raise Refusal with MESSAGE unless CONDITION holds."""
    if not condition:
        raise Refusal(message)


def sha(path):
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data):
    """Hex SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def read(path):
    """Load a JSON file."""
    return json.loads(pathlib.Path(path).read_text())


def encode(value, compact=False):
    """Canonical JSON bytes with a trailing newline."""
    if compact:
        text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    else:
        text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
    return (text + '\n').encode()


def write_bytes(path, data):
    """Atomically create PATH with DATA; refuse to replace an existing file."""
    path = pathlib.Path(path)
    check(not path.exists(), 'refusing to overwrite %s' % path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path, value, compact=False):
    """Atomically create a canonical JSON file."""
    write_bytes(path, encode(value, compact))


def verify_complete(directory, expected=None):
    """Check COMPLETE.json (optionally its own hash) and every file digest it lists."""
    directory = pathlib.Path(directory)
    check((directory / 'COMPLETE.json').is_file(), 'missing COMPLETE.json in %s' % directory)
    if expected is not None:
        check(sha(directory / 'COMPLETE.json') == expected, 'COMPLETE.json hash mismatch in %s' % directory)
    value = read(directory / 'COMPLETE.json')
    for name, record in value['files'].items():
        digest = record['sha256'] if isinstance(record, dict) else record
        check(sha(directory / name) == digest, 'file hash mismatch %s/%s' % (directory, name))
    return value


def link_or_copy(source, target):
    """Hardlink SOURCE to TARGET on the same filesystem, otherwise copy; returns the method used.

    The source is never modified, and permissions are not changed because a hardlink shares the inode.
    """
    import shutil
    source, target = pathlib.Path(source), pathlib.Path(target)
    check(not target.exists(), 'refusing to overwrite %s' % target)
    try:
        os.link(source, target)
        return 'hardlink'
    except OSError:
        shutil.copyfile(source, target)
        check(sha(source) == sha(target), 'copy differs from source %s' % source)
        return 'copy'


def repo_path(relative):
    """Absolute path of a repository-relative path."""
    return REPO / relative


def pair_rows(path):
    """Gene/Disease TSV rows as tuples, refusing duplicates."""
    lines = pathlib.Path(path).read_text().splitlines()
    check(lines and lines[0].split('\t') == ['Gene', 'Disease'], 'unexpected header in %s' % path)
    rows = [tuple(line.split('\t')) for line in lines[1:] if line]
    check(all(len(row) == 2 for row in rows), 'malformed row in %s' % path)
    check(len(rows) == len(set(rows)), 'duplicate rows in %s' % path)
    return rows
