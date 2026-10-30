# Artifact storage and access

On 30 October 2026, the authors requested removal of the temporary bulk data
attachments from this repository's GitHub Releases. All 422 archives
(401.13 GB) have been removed. The ten small manifest assets (4.16 MB) remain.
See the [verified removal record](../reproducibility/verification/temporary-release-asset-removal.json).
Code, pinned environments, result summaries, verification records and the small
release manifests remain on GitHub. Large archives are retained on the research
servers; their copies passed the preserved release checksums before removal.

The previous GitHub archive URLs have been withdrawn. Historical manifests and
verification records describe the bytes used for completed checks; their URLs
do not imply that those archives remain available for download. The archive
commands in the reproduction guides require access to retained copies until a
replacement public archive is published. Report-to-table reproduction and the
manuscript build use committed files and remain available without those archives.

## Working data

Keep intermediate checkpoints, prediction matrices, caches and incomplete runs
on compute or institutional storage. Transfer them directly between research
servers. Retain file identities and SHA256 checksums, and verify copied bytes
before scientific analysis. Do not upload working collections to GitHub as an
intermediate transfer step.

## Final paper archive

Before submission, publish a curated, versioned archive of the frozen inputs,
selected checkpoints and score collections needed for the documented workflows.
Use a research data repository with a persistent identifier and sufficient
capacity. Preserve the logical file layouts and checksum mappings when removing
duplicate blobs. Update manifests, download commands and manuscript availability
statements to the actual destination, and test those downloads before claiming
complete public reproduction.

The authors have chosen Zenodo and will perform the upload themselves. An
[upload-ready institutional bundle](frozen-archive.md) is being prepared with
deduplicated data, original file identities, restore instructions and metadata.

The replacement public archive has not yet been published. Completed numerical
checks remain valid, but complete public score replay, checkpoint inference and
training from the retained inputs currently depend on data access. Full matched
checkpoint inference remains an outstanding scientific verification step.
