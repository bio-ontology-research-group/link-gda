# Frozen archive preparation and restoration

The authors will upload the final frozen data to Zenodo themselves. Preparation
is in progress for an upload-ready bundle at
`/datawaha/borg/multihop-gda-zenodo-20261030-v1` on `dm.kaust.edu.sa`.
That path is on the data mover, not mounted on IBEX. The bundle is ready only
when its final verification receipt is present; a running pack or copy is not
a verified archive. Public Zenodo record identifiers are not available yet.

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

## Verification and publication boundary

Packing checks the source archive identities, selected member bytes and complete
unique-object inventory. Final verification must check every delivered part and
restore all ten original views before handoff. The numerical inference audit is
separate: source or archive hashes alone do not prove score/rank/table agreement.

After the authors upload the records, the catalog's part URLs and manuscript
data-availability statement must be updated to the actual versioned Zenodo
identifiers. Until then, local institutional restoration is available once the
bundle passes verification; complete public data access remains pending.
