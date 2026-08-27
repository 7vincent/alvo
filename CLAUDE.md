# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this directory.

## What this is

A **single-purpose, fully-offline pipeline** that separates photos of a military
ceremony (*solenidade*) into per-person folders using face recognition. Not a
library and not a service — four standalone CLI scripts run by hand, in order,
once per event.

```
fotos/
├─ Fotos/        # 475 JPGs, ~1.9 GB, flat (no subfolders). READ-ONLY input.
├─ projeto/      # the four scripts + LEIAME.md (the user-facing manual, in PT-BR)
└─ .venv/        # broken — see below
```

Everything (models, embeddings, clustering) runs locally. No image and no
embedding ever leaves the machine — this is a hard requirement, not a preference.

## Environment: read before running anything

**Do not build the venv with bare `python3`.** In this directory `python3`
resolves to Python **3.9.6** (VSCode auto-activates `.venv` and puts its `bin`
first on PATH; pyenv is set to `system`, which is the same 3.9.6 from Apple
CommandLineTools). The scripts cannot run on 3.9:

- `Path.hardlink_to()` (`buscar.py`, `distribuir.py`) requires **Python 3.10+**
- `sklearn.cluster.HDBSCAN` (`agrupar.py`) requires **scikit-learn ≥ 1.3**
- `onnxruntime` ships no cp39 wheels at all

Use **Python 3.12** explicitly — verified working end to end:

```bash
rm -rf .venv
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install insightface onnxruntime opencv-python-headless pillow pillow-heif scikit-learn numpy
```

Verified resolution: insightface 1.0.1, onnxruntime 1.29.0, scikit-learn 1.9.0,
numpy 2.5.2, opencv 5.0.0 — all wheels, nothing compiles. 3.14 is untested here;
`insightface` is sdist-only on PyPI, so a version with no matching wheel for its
dependencies falls back to building from source.

`indexar.py` downloads the `buffalo_l` model (~300 MB) on first run — the only
moment internet is needed.

**CoreML is broken on this machine and fails silently-ish.** The provider loads
fine, then throws per-image at inference (`Non-zero status code ... node
CoreMLExecutionProvider_*`). Because `indexar.py` catches that per photo and
`continue`s, you get `erro_det` on 100% of photos, **zero progress lines**, exit
code 0, and a cheerful `Pronto. 0 rostos`. Both scripts now default to
CPU-only; `--coreml` opts back in. Diagnose with
`SELECT status, COUNT(*) FROM fotos GROUP BY status`. CPU runs ~0.8 photos/s
(475 photos ≈ 10 min), which is fine at this scale.

If `python3 -m venv` is interrupted (Ctrl+C), it leaves a **half-built venv with
python symlinks but no `activate` and no `pip`** — `_setup_pip()` runs *before*
`setup_scripts()` in `venv/__init__.py`. The recovery is `rm -rf .venv` and
start over; re-running `venv` over the wreckage also works.

## The pipeline

Run from `projeto/`. The order is a hard dependency chain — each step reads what
the previous one wrote into SQLite.

Once someone is in `pessoas.db`, finding them in a *new* folder skips steps 2-5
entirely — and `--conhecidos` never loads the recognition model, so it returns
in well under a second:

```bash
python indexar.py ~/Fotos/OutroEvento --db outro.db
python buscar.py --conhecidos --db outro.db --saida entrega --copiar
```

```bash
python indexar.py ../Fotos --db solenidade.db   # 1. expensive, resumable
python buscar.py referencias/ --db solenidade.db # 2. optional: find known people
python agrupar.py --db solenidade.db             # 3. cluster everyone
#  4. MANUAL: rename clusters/cluster_007.jpg -> "cluster_007 - Cap Silva.jpg"
python distribuir.py --db solenidade.db          # 5. build per-person folders
```

`indexar.py` is idempotent and resumable: it skips paths already in `fotos`, so
re-running after a crash costs nothing. `agrupar.py` is **not** — it does
`DROP TABLE IF EXISTS grupos` on every run, which discards the cluster numbering
your manual renames refer to. Re-clustering means renaming again.

## The SQLite DB is the interface between scripts

`solenidade.db` is the only thing the four scripts share. Changing this schema
means touching every consumer.

| Table | Written by | Key columns |
|---|---|---|
| `fotos` | `indexar.py` | `caminho` (UNIQUE — the resume key), `n_rostos`, `status` |
| `rostos` | `indexar.py` | `emb` BLOB = float32[512] **already L2-normalized**, `x1..y2` relative 0..1, `nitidez`, `area_rel` |
| `grupos` | `agrupar.py` | `rosto_id` → `grupo`; recreated from scratch each run |

A **second, separate** DB — `pessoas.db`, via `memoria.py` — is the cross-event
face registry: `pessoas(nome)` + `referencias(pessoa_id, origem, fonte, emb)`,
`fonte` being `'referencia'` (a photo you supplied) or `'match'` (fed back from
a search via `--realimentar`). It is deliberately *not* in `solenidade.db`: the
event index is disposable, the registry is what makes the next event cheap.
`UNIQUE(pessoa_id, origem)` makes re-registration idempotent.

Note it is the only file in the project that binds a name to a biometric vector.

Two consequences of embeddings being pre-normalized: similarity is just a dot
product (`1.0 - embs @ q` is cosine distance, no division needed), and euclidean
distance is monotonic with cosine — which is why `agrupar.py` can hand raw
vectors to HDBSCAN's tree instead of materializing an N×N matrix. Store
normalized, or both break silently.

Bounding boxes are stored **relative** so crops survive the downscale
`carregar_imagem_bgr()` applies (`lado_max=2200`).

## Invariants — do not break these

- **`Fotos/` is read-only.** Nothing is moved, renamed, or deleted. Ever.
- **Output uses hardlinks**, not copies. 50 people × 100 photos ≈ 0 bytes.
  `--copiar` exists only for the final pendrive/zip handoff. Both scripts
  fall back to `shutil.copy2` on `OSError` (cross-filesystem targets).
- **`carregar_imagem_bgr()` must apply `ImageOps.exif_transpose`.** A portrait
  photo read without it arrives sideways and the detector finds no faces at
  all. This one line is load-bearing for accuracy.
- **The manual rename step is deliberate**, not a gap. Building a review UI was
  judged a week of work for no gain; the Finder in gallery mode does the job.
  Don't "improve" it into an app unless asked.

## Conventions

- **Portuguese naming throughout** — functions, variables, CLI flags
  (`--limiar`, `--saida`, `--copiar`, `--nitidez-min`). Match it.
- **`.py` files are ASCII-only**: unaccented Portuguese in identifiers, comments,
  and docstrings (`variancia`, `orientacao`, `nao`). The single exception is the
  en/em dash inside `distribuir.py`'s `PADRAO` regex, which is deliberate: it
  accepts `-`, `–` and `—` so a rename typed with any dash still parses.
  Markdown (`LEIAME.md`) uses full accents.
- **Heavy imports are function-local** (`insightface`, `cv2`, `sklearn`, `PIL`).
  Keeps `--help` instant and failures early and legible. Keep them there.
- No test suite, no linter, no CI, no git repo. Verification is by sampling the
  output folders visually.

## Tuning

Accuracy is threshold work, not code work. Expect **70–85% correct**, not 99% —
say so rather than implying otherwise.

| Symptom | Fix |
|---|---|
| Few faces found in formation shots | `indexar.py --det-size 1600` (reindex, ~2× slower) |
| Same person split across clusters | `agrupar.py --limiar 1.05` |
| Different people in one cluster | `agrupar.py --limiar 0.75` |
| Search missing photos | `buscar.py --limiar 0.55` |
| Search pulling wrong people | `buscar.py --limiar 0.35` |
| Blurry/tiny faces polluting groups | `agrupar.py --nitidez-min 60 --area-min 0.002` |
| Deliver only photos where the person is the subject | `distribuir.py --area-min 0.01` |

`buscar.py` prefixes output filenames with the cosine distance (`0.312_foo.jpg`),
so a folder sorts best-match-first — review from the bottom up, where the
mistakes are. Re-running with a higher `--limiar` only adds files; the distance
prefix of an already-written photo cannot change, so no duplicates appear.

**A person may hold many vectors, and matching is min-distance across all of
them** — not the centroid. This matters: averaging a frontal and a profile
embedding yields a blur that matches neither, while min-distance lets each
stored angle cover its own pose. `--realimentar N` feeds N *diversity-selected*
matches back in (`memoria.diversos()`, greedy max-min) rather than N
near-identical frontals. Measured here: 1 vector → 94 photos; 13 vectors → 99,
zero false positives, the 5 extra being extreme profiles and head-down shots.
Only ever realimentar a result you have visually checked — a contaminated
registry silently degrades every future search.

**Realimentar invalidates the threshold you just used.** Min-over-N shrinks
every distance, other people's included, so the old cut becomes permissive.
Re-derive it: Túlio's safe cut went 0.65 → 0.55 after feedback, while Tiberio's
stayed at 0.60 only because his separation was unusually wide. The gain is not
guaranteed either — Tiberio gained 5 photos, Túlio gained none.

**A reference cropped tight enough that the face fills the frame defeats the
detector at det_size 1024** — SCRFD matches against scale anchors, and a face
larger than the biggest anchor fires nothing, returning zero faces with no
error. Upscaling does not help (relative size is unchanged); lowering det_size
does. `embutir()` walks 1024 → 640 → 480 → 320 and reports which worked.

**Reference domain matters more than reference resolution.** A webcam frame of
Túlio put his best match at 0.375 with no clean gap anywhere; a 185x192 crop
from the event itself put it at 0.039 with a clean gap at 0.74. Prefer a
reference shot by the same camera at the same event, even a tiny one.

**The 0.45 default is tuned for frontal faces and silently drops profiles.**
ArcFace penalizes side views hard: in the solenidade set, one subject's frontal
shots landed at 0.22-0.45 while the *same person* in profile sat at 0.45-0.60,
and genuinely different people only started at 0.64. Verify before trusting the
default — dump the distance histogram over all indexed faces and look for the
gap. Here the real gap was at ~0.62, so `--limiar 0.60` returned 94 photos
against the default's 74, with no false positives. Multiple reference photos per person (a subfolder named after
them) averages the embeddings and beats a single shot substantially.

Quality gates in `indexar.py` are module constants, not flags:
`DET_SCORE_MIN = 0.60`, `LADO_MIN_PX = 45`.

## LGPD

`solenidade.db` holds **biometric data** (Lei 13.709/2018, art. 5º, II) of named
military personnel. Never version it, upload it, or pass it to any external
service — including as context to a remote tool. Keeping it makes the next
ceremony cheap (people stay recognized); that tradeoff is the command's call to
make, not ours.

## Known rough edges

- `LEIAME.md` and `agrupar.py`'s docstring say the contact sheets are `.png`;
  the code writes `.jpg`. The `.jpg` is correct.
- `buscar.py` and `distribuir.py` both default to `--saida saida`. Their folder
  names differ in practice (reference filename vs. person name), but a collision
  mixes distance-prefixed files with plain ones. Point one of them elsewhere.
- `LEIAME.md` examples use `~/Fotos/Solenidade`; the actual input is `../Fotos`.
