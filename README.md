# Synthetic data for laparoscopic liver registration

Goal: register a preoperative CT liver to a single laparoscopic photo, with the tumour where it really is.
The earlier studies (archive tag `archive-2026-11`, branch `preregistration`) found what breaks current methods:
the distance (monocular depth models put the liver 3-9x too far), the choice of rotation among look-alike poses,
untrustworthy depth shape, and deformation. This branch builds a synthetic dataset that contains those difficulties,
with the answer known, and trains models on it. Real data is used only for testing (P2I-LReg real frames for
comparison with the literature; LLR-LUS once, pre-registered).

- `docs/GENERATOR_SPEC.md`: what the generator simulates and why (PIVOTS replica first, then our additions)
- `regsim/p2ilreg.py`: P2I-LReg access, camera conventions, drawing helpers
- `regsim/anatomy.py`: the 6 liver parts and part / head-direction camera poses
- `regsim/depth_damage.py`: bumpy / flattened / slanted depth damage
- `regsim/generator.py`: prototype camera-view generator (to be rebuilt on the PIVOTS replica)
- `regsim/lab/`: Liver Deformation Lab (LIBR physics; record manipulations as generator templates)
- `regsim/preview/`: generator vs dataset vs real-photo preview page

Run from the repo root, e.g. `python -m regsim.lab.make_liver_sim --patient 06`. Data paths:
`REGSIM_P2ILREG` (default `dataset/liver-registration`), outputs in `REGSIM_OUT` (default `out/`, git-ignored).
Patient data and anything derived from LLR-LUS never enter git.
