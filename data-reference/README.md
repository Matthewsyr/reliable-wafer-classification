# Development reference

`development_reference.csv.gz` contains 54,355 **metadata rows**, all from the original WM-811K development (`train`) partition. It contains no wafer-map arrays. The original final-test partition is not included. This reference preserves the fixed development split used for the later repair and geometry studies; its `audit` partition has already been examined and is exploratory.

The adjacent `development_reference_provenance.json` records hashes of all three source artifacts and the merged file. The merge uses stable `source_row` identities and checks labels, lots, original membership, and exact-map hashes across sources. It also verifies zero lot or exact-map-hash overlap across roles. Distinct wafer identities with identical maps are retained, including conflicting labels; they stay in the same connected component.

| Fields | Source and interpretation |
| --- | --- |
| `source_row`, `label`, `lot`, `wafer_index`, `source_split` | Historical `development_metadata.csv.gz`; `wafer_index` is copied from the source extraction, never invented. `none` is the normal class, not a missing label. |
| `height`, `width`, `valid_die_count`, `failed_die_count`, `failed_die_rate`, `map_sha256` | Historical raw-map scanner. `valid_die_count` is measured from the map and is **not** represented as an independently copied original `dieSize` field. |
| `raw_data_offset`, `raw_data_nbytes`, `order`, `hash_matched` | Verified byte index into the exact original pickle identified by SHA-256 below; `order` is the stored array order. |
| `role`, `component`, `natural_role`, `stress_role`, `partition_reason`, `shape`, and other stress annotations | Frozen repair split manifest. `role` is the actual training/tuning/audit assignment; `natural_role` is retained for provenance only. |

The expected original `LSWMD.pkl` SHA-256 is:

```text
1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65
```

Obtain the dataset separately; do not commit it. After installing the package and CNN dependency, use the same reference for all three input roles:

```bash
python -m waferdefect prepare --raw-pickle /path/to/LSWMD.pkl --metadata data-reference/development_reference.csv.gz --raw-index data-reference/development_reference.csv.gz --existing-split data-reference/development_reference.csv.gz --output outputs/development --representation both --include-linearity
```

This validates the entire source-file checksum, then reads only selected development map payloads at the indexed offsets and checks every map hash. It avoids deserializing the full pickle and does not access final-test maps for preprocessing or fitting. The checksum operation necessarily reads source-file bytes; it does not interpret or evaluate final-test samples. The generated arrays remain local and are ignored by Git.

Without `--raw-index`, the pinned-pickle fallback deserializes the entire source and can need substantial RAM. The indexed command above is the preferred reproduction path. Preparation fails with an issue list for invalid labeled rows or repeated sample identities; it does not silently discard them. No `die_size` column is required; if supplied, it is checked against the measured valid-die count.

To rebuild this metadata merge from the three historical artifacts, use `scripts/build_development_reference.py`. Those original artifacts are not fabricated or reconstructed from model predictions. Their exact hashes are in the provenance file.
