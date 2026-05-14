# AEB Small 100 SGF Corpus

This directory contains 100 unmodified SGF files selected from Andries
Brouwer's Database of Go Games for the story #9 replay-batch milestone.
The SGFs are stored under stable local names in `sgf/` so the repository
does not expose the source archive's abbreviated tournament paths as the
primary browsing experience.

Source page: https://homepages.cwi.nl/~aeb/go/games/index.html
Archive URL: https://homepages.cwi.nl/~aeb/go/games/games.tgz
Archive observed date: 2026-05-13
Archive Last-Modified: Fri, 08 May 2026 20:25:21 GMT
Archive ETag: "2bd642b-65154322a739f"
Archive Content-Length: 45966379
Archive SHA-256: 0fdb8d2ef6f76845c49f05b1cfc0b04ed439df33e28a7fd3883fbd5170f942aa

The source page states that the collection is public domain and that the
games may be used freely.

Selection rule: sort every `.sgf` member in `games.tgz` lexicographically
by archive path, then keep the first 100 files that replay with status
`ok` under VantaGo diagnostics. The SGF bytes are copied as-is from
the archive; no normalization or rewriting is applied. The local filenames
are flattened to `aeb-small-0001.sgf` through `aeb-small-0100.sgf`.

See `MANIFEST.txt` for local IDs, local filenames, original archive paths,
event names, rounds, players, results, dates, move counts, byte counts,
per-file SHA-256 hashes, and skip reasons encountered before the 100
accepted files were found.

Verification command:

```bash
uv run vantago replay-batch data/raw/aeb-small-100
```
