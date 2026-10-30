# PhenoQuery frozen research data: manual Zenodo handoff

This bundle preserves all 10,662 members of the ten original frozen data
collections, with 6,308 distinct byte objects. Original filenames, modes and
component manifests are reconstructed by the supplied restore tool. Prediction
arrays and model states retain their original bytes. The code repository is
https://github.com/bio-ontology-research-group/link-gda.

The final institutional location is
`/datawaha/borg/multihop-gda-zenodo-20261030-v1` on `dm.kaust.edu.sa`.
`ARCHIVE_READY.json` in that directory confirms the completed DataWaha copy
verification. Until that receipt is present, staging is incomplete. No Zenodo
upload or publication is performed by this workflow.

## Upload later

1. Check that `ARCHIVE_READY.json` says `passed`. `SHA256SUMS` and
   `FILE_INVENTORY.json` describe the delivered payload, and
   `RESTORE_VERIFIED.json` records the ten original collection restorations.
2. Create one Zenodo record for each of the eleven directories listed in
   `records.json`. Upload every file immediately inside each directory.
   Each directory fits below 50 GB and 100 files. The `prepared-inputs` record
   additionally holds the master catalog, restore tools and verification.
   `metadata-draft.json` supplies suggested metadata; review the title, authors,
   description and upstream data licences before publication. The draft leaves
   licence selection to the authors and does not relicense upstream material.
3. Record the version-specific record identifiers, DOIs and file URLs.
   Add the corresponding URLs to each part in a separate copy named
   `catalog-zenodo.json`; preserve the verified original `catalog.json`.
   The original manifests contain historical GitHub URLs whose payloads were
   withdrawn; the new catalog supplies the replacement content locations.
   Publish the completed catalog with the master record and cross-reference
   the component records. Keep future releases as new Zenodo versions.
4. Update the manuscript data-availability paragraph and the code repository's
   data links to the actual versioned Zenodo DOIs. Public access remains
   pending until those records are published and tested without credentials.

## Restore from the institutional copy

Use Python 3.9 or later. Set `BUNDLE` to this directory or a copy containing all
eleven record directories, and `WORK` to a project-local working directory.
Keep the cache outside each restored workspace. The cache requires about
706 GB if all compressed parts and extracted objects are retained.

```bash
python3 "$BUNDLE/tools/code/reproduce/curated_archive.py" restore \
  --catalog "$BUNDLE/catalog.json" --bundle-root "$BUNDLE" \
  --collections controlled-query-scores-20261029-v1 \
  --root "$WORK/query-scores" --cache "$WORK/archive-cache" \
  --receipt "$WORK/query-restore.json"
```

The ten original collection names are the keys of `original_manifests` in
`catalog.json`. Restore each into a separate workspace while sharing the cache.
The restored views can be used by the existing scientific reproduction commands
in the code repository. They retain each original collection's provenance and
upstream notices. Scientific execution requires the separately pinned software
environments documented in that repository.

For a published Zenodo catalog containing valid URLs, omit `--bundle-root` and
point `--catalog` to `catalog-zenodo.json`. When starting with the Zenodo master
record, unpack `restore-tools.zip` to obtain `code/reproduce/curated_archive.py`
and its companion module. The manuscript's numerical inference verification is
separate from the archive's byte and restoration verification.
