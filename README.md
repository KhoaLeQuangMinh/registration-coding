# registration-coding

Research code for registering a preoperative CT liver to a laparoscopic image.

## Layout

```
lus_eval/     Evaluation on LLR-LUS (real patients, tumour ground truth) and the liver-mask tools. See lus_eval/README.md
dataset/      All data (git-ignored, never committed). LLR-LUS, P2I-LReg, model weights
papers/       Reference PDFs (git-ignored)
```

## Setup

```bash
conda activate regis                # Python 3.11
python -m pip install -r requirements.txt
```

Run scripts from the repository root, for example `python lus_eval/score.py --method-dir ...`.

## Data rules

- Patient data, and anything derived from it (masks, renders, results), lives under `dataset/` and never enters git.
- LLR-LUS is research-use only: nothing derived from it may leave this computer.

## History

This branch (`clean`, 2026-11-27) keeps only the LUS evaluation. Earlier work is preserved:

| Where | What |
|---|---|
| tag `archive-2026-11` (= branch `preregistration`) | Exploration, benchmarks, all LUS experiments (`studies/`, `goal/`) |
| tag `archive-synthetic-2026-11-27` (= branch `synthetic`) | `regsim/`, `depth-noise-estimate/`, `docs/` |
| `../registration-coding_backup_2026-11-27.tar.gz` (local only) | Everything that was never committed: `synthetic-data/`, `my-pipeline/`, `studies/` and `goal/` outputs, `out/` |
