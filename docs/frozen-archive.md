# Frozen archive preparation and restoration

The authors will upload the final frozen data to Zenodo themselves. The
upload-ready bundle is verified at
`/datawaha/borg/multihop-gda-zenodo-20261030-v1` on `dm.kaust.edu.sa`.
That path is on the data mover, not mounted on IBEX. `ARCHIVE_READY.json`
confirms that all 153 payload files (322,013,622,516 bytes)
passed destination checksums and exact file-set checks. All ten original collection
views passed restoration. See the [handoff proof](../reproducibility/verification/frozen-archive-datawaha-handoff.json)
and [manual upload instructions](zenodo-upload-handoff.md). Public Zenodo record
identifiers are not available yet.

The curation plan retains all 10,662 entries from the ten frozen collections:
9,141 distinct logical paths and 6,308 unique file contents. It removes duplicate
bytes without changing arrays, model states, validation selections, cohorts or
statistical rows. Every unique object is checked against its original SHA256.
Parts are approximately 4 GB of uncompressed payload, grouped into meaningful
records below the default Zenodo quota. Original manifests remain in the catalog.

The code and small metadata stay on GitHub. Large payloads stay on institutional
storage until the authors' Zenodo upload. No upload credentials are required to
prepare or verify the local bundle.

## Restore an original collection

The restore command recreates original filenames, permissions, embedded component
manifests and the empty artifact lock. Existing reproduction commands can use
that view. Cached objects are shared by hard links when their permissions and
filesystem permit; executable aliases are copied to preserve their modes.
Different existing files and symlinks are rejected.

Use Python 3.9 or later for restoration; the scientific analyses still require
their separately pinned environments. From the code repository root, set
`BUNDLE` to an accessible local copy of the bundle and `WORK` to a project-local
working directory, then run:

```bash
mkdir -p "$WORK"
python3 code/reproduce/curated_archive.py restore \
  --catalog "$BUNDLE/catalog.json" \
  --bundle-root "$BUNDLE" \
  --collections controlled-query-scores-20261029-v1 \
  --root "$WORK/query-scores" \
  --cache "$WORK/archive-cache" \
  --receipt "$WORK/query-restore.json"
```

Replace the collection name for the other original workflows. Keep each original
workflow's workspace separate and share the cache outside those workspaces.
The original collection names and manifests are retained in `catalog.json`.
The [portable reproduction guide](reproduction.md) supplies the subsequent
analysis and inference commands.

## Use restored views with the existing recipes

Restore replaces the initial `artifacts.py fetch` command in each recipe. Set
that recipe's workspace/input-root variable to the restored directory, then
continue with its numerical replay, inference or training commands. Retain the
source revision and pinned environment specified by the scientific recipe.
The restore tool itself comes from the current repository; older scientific
checkouts may predate it. Do not run the old GitHub downloader after restoration.

| Original collection | Recipe | Top-level paths in its restored view |
|---|---|---|
| `inputs-20261028-v1` | [Frozen inputs](frozen-inputs.md) | `benchmark`, `environment`, `eq`, `matched`, `sources`, `ultra` |
| `matched-inputs-20261029-v1` | [Matched annotation supplement](frozen-inputs.md#matched-kge-annotation-supplement) | `matched-kge` |
| `controlled-query-scores-20261029-v1` | [Controlled-query score replay](controlled-query-score-replay.md) | `controlled-query`, `controlled-query-reference` |
| `controlled-query-checkpoints-20261029-v1` | [Controlled-query inference](controlled-query-checkpoints.md) | `controlled-query` |
| `matched-kge-scores-20261029-v1` | [Matched score replay](matched-kge-score-replay.md) | `matched-kge`, `matched-benchmark`, `matched-kge-reference` |
| `matched-kge-checkpoints-20261029-v1` | [Matched checkpoint inference](matched-kge-checkpoints.md) | `matched-kge` |
| `baseline-outputs-20261029-v1` | [Native baseline replay](final-baseline-results.md) | `baseline-outputs` |
| `embedding-outputs-20261029-v1` | [Supporting embedding replay](embedding-archive-replay.md) | `legacy-embedding`, `legacy-embedding-reference` |
| `comparator-outputs-20261029-v1` | [Historical graph/query controls](historical-comparator-replay.md) | `benchmark`, `historical-comparators` |
| `query-outputs-20261028-v1` | [Supporting query replay](reproduction.md) | `eq`, `matched` |

An inference recipe can require several collections in the same workspace.
Pass all required names to `--collections` in one restore command. For example,
the matched checkpoint workflow uses the frozen inputs, annotation supplement
and matched checkpoint collection. Their shared paths must have identical
content and permissions; restoration checks this before installing them.
Keep different scientific workflows in separate workspaces.

The catalog preserves the original top-level release manifests byte for byte.
If a recipe needs a `*_RELEASE_MANIFEST` argument, extract those small JSON files
outside the restored workspace:

```bash
python3 - "$BUNDLE/catalog.json" "$WORK/original-manifests" <<'PY'
import hashlib, json, sys
from pathlib import Path
catalog = json.loads(Path(sys.argv[1]).read_text())
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
for tag, original in catalog['original_manifests'].items():
    payload = original['text'].encode()
    assert hashlib.sha256(payload).hexdigest() == original['sha256']
    target = destination / (tag + '.json')
    if target.exists():
        assert target.read_bytes() == payload
    else:
        with target.open('xb') as stream:
            stream.write(payload)
PY
```

These original manifests retain withdrawn GitHub URLs as provenance. Use the
curated catalog to fetch bytes, and the original manifests for scientific
identity checks. The new catalog's object locations reconstruct the same file
views without those old downloads.

## Verification and publication boundary

Packing checks the source archive identities, selected member bytes and complete
unique-object inventory. Final verification checked every delivered part,
restored all ten original views, and independently checked every delivered byte
on DataWaha. The numerical inference audit is
separate: source or archive hashes alone do not prove score/rank/table agreement.

After the authors upload the records, the catalog's part URLs and manuscript
data-availability statement must be updated to the actual versioned Zenodo
identifiers. Until then, verified local institutional restoration is available;
complete public data access remains pending.
