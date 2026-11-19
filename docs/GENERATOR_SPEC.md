# Synthetic generator: specification (v1, 2026-11-19)

Goal: training and validation data for stage 1 (rigid: ranked poses) and stage 2 (LIBR bend + small pose fix), with the
answer known: true rigid pose, true deformation of every point, and virtual tumours (inside points). Real data is used
only for testing (P2I-LReg real for comparison; LLR-LUS once, pre-registered).

Primary reference: PIVOTS (Liu, ..., Speidel, Pfeiffer; Medical Image Analysis 113 (2026) 104139), Section 3.5 and
Fig. 3 (journal PDF in the repo root), plus its supplementary Table 1 (values read from the arXiv v1 HTML; marked [S1]).
PIVOTS builds on Pfeiffer et al. (V2S-Net, MICCAI 2020) and Pfeiffer's thesis (TU Dresden 2022).

| Step | PIVOTS (what they do) | Ours (decision) | Why we differ |
|---|---|---|---|
| **1. Shape** | random organ-like shapes (sphere + extrusions/subtractions, smoothed; convex front, concave hilum), size U(100, 300) mm | **real livers in patient axes**: 21 P2I-LReg CTs + AMOS (~360 labelled scans, test-only in PIVOTS); random shapes optional for stage 2 | stage 1 must learn anatomy (which part faces the camera), which random shapes do not have |
| **2. Scene** | 0-3 placeholder organs U(50, 200) mm; abdominal wall = convex hull, outset U(0, 50) mm + Perlin noise <= 20 mm | abdominal wall + placeholder organs as in PIVOTS (needed for ligament anchors and for "other organs pushing") | same |
| **3. Attachments** | fixed patch grown from a random face until 2-5 % of the surface is covered (Dirichlet zero displacement) | fixed patch 2-5 %, grown from a random face **inside the posterior-cranial region** (bare area / vena cava) | anatomy is known for real livers |
| **4. Ligaments** | 1-2 random paths <= 100 mm on the organ, extruded to the wall; linear springs, k U(100, 300) N/m, rest-length factor c0 U(0.9, 1.1) (c0 < 1 pulls, c0 > 1 pushes) | same springs and ranges, paths placed at **falciform / round ligament** (front, between left lobe and segment 4) and **coronary / triangular** ligaments (top) with probability, random paths otherwise | anatomical placement where known, PIVOTS randomness elsewhere |
| **5. Material** | homogeneous isotropic compressible **neo-Hookean**; E U(3, 30) kPa; nu U(0.45, 0.48); density U(1.05, 1.09) g/cm3; no viscosity | same | literature-backed ranges (PIVOTS cites Glozman 2010, Mattei 2016, Marchesseau 2017, Estermann 2020, Lemine 2024); healthy liver < 2.5 kPa, up to 16 kPa diseased |
| **6. Mesh** | GMSH + Netgen tetrahedra, max element 1 cm | GMSH + Netgen (pip `gmsh`), max element 1 cm for solving; our 20k-vertex surface embedded for drawing | boundary-conforming (our current 8 mm cubes are up to one layer too thick) |
| **7. Loads** | gravity + springs + fixed patch; magnitude / position / area of forces varied to stand in for the unknown pneumoperitoneum and tool forces | gravity as the **change from the CT's supine pose** (patient tilt sampled) + springs + fixed patch + **explicit tool loads from the Liver Deformation Lab recordings** (press: displacement along the shaft at a tip radius; grasp: prescribed displacement of a patch), varied around each template | tool-tissue interaction is a stated PIVOTS limitation; the CT shape already includes supine gravity |
| **8. Solver** | SOFA, implicit Euler, 50 ms steps, conjugate gradient (<= 1000 it, tol 1e-8), Rayleigh damping 0.1/0.1 [S1]; run to equilibrium but **export the step with the largest deformation** (includes transient overshoot) | phase 1: **as PIVOTS** (SOFA, largest step); phase 2 switch: static equilibrium | an overshoot snapshot is not a resting shape; measured as a switch, not assumed |
| **9. Rigid part** | none (PIVOTS presumes rigid pre-alignment) | true pose = pose of the attached region; remaining motion = deformation | stage 1 needs a well-defined true pose |
| **10. Observation** | partial point cloud: region grown by a random weighted mix of geodesic distance, normal angle and Poisson noise; surface amount U(10, 100) % [S1]; or a **camera view**: camera outside the mesh at 50-200 mm, aimed at a random surface point, 70 deg horizontal FOV, aspect 16:9 or 4:3 [S1] | **camera view only** (we are monocular): camera near the P2I-LReg synthetic poses (jittered), lens 60-90 deg wide, principal point offset N(0, 60 px); render the visible-liver mask, edge types (real / covered / border), relative depth, tools drawn where they touch | our input is a photo, not a reconstructed surface |
| **11. Noise** | Perlin offset (amplitude A_P U(0, 10) mm in training, 0-15 mm in the noise benchmark) + Gaussian (sigma U(0, 3) mm, 0-5 mm benchmark), subdivision U(5, 15), sparsification with Perlin frequency U(1, 5) [S1] | depth: **scale thrown away**, then damage of three kinds (bumpy = Perlin-like, flattened, slanted) up to ~35 deg mean tilt error, also logged as PIVOTS-style mm amplitudes; mask: edge wobble; occluders: tools from the contact model, fat patches (rare), scope circle | Depth Anything errors differ from stereo noise; sizes to be checked on SERV-CT |
| **12. Size** | 193,563 training samples, 3,008 synthetic test | start ~100k training, held-out patients for validation | ablation in PIVOTS: one third of the data costs < 2 mm |
| **13. Benchmarks** | test set split by deformation level (PRD bins), noise grid (A_P x sigma), visibility bins (0-10, 10-20, 20-30, 30-40 %) | same three stratifications, on held-out patients, for both stages | PIVOTS found < 10 % visible surface fails for every method: report it, do not hide it |

## Plan (user decision 2026-11-19): replicate PIVOTS first, then add ours

The PIVOTS generator is NOT public (repo: network, checkpoint `pivots_v5`, inference Docker `pfeiffermicha/pivots`
only; no simulation code, datasets "still uploading"). So phase 1 rebuilds it from the paper, supplementary Tables 1-2
and Pfeiffer's thesis (TU Dresden 2022: organ-shape algorithm, partial-surface extraction). Asking the authors for the
generator code is worth doing in parallel (the user's call).

**Phase 1: faithful replica, quirks included** (same rule as for Self-P2IR: copy the method, measure each quirk later)
- the PIVOTS column above, item by item: random organ-like shapes, placeholder organs, convex-hull wall with Perlin
  noise, spring ligaments, 2-5 % fixed patch, neo-Hookean in SOFA (implicit Euler 50 ms, CG), GMSH + Netgen 1 cm tets,
  **export of the largest-deformation step**, partial surfaces + Perlin / Gaussian noise
- runs on Linux: the lab's Blackwell machine (RTX PRO 5000, via `ssh blackwell` through lqd, NetBird on) or Kaggle;
  SOFA from the `sofa-framework` conda channel (sofa-python3, Linux 25.06) or the official v26.06 Linux binaries
- **fidelity check, fixed before generating:**
  1. sampled parameters match supplementary Table 1;
  2. the synthetic test set's mean initial displacement (PRD) is close to theirs (32.14 mm synthetic, 32.04 mm AMOS);
  3. the released PIVOTS checkpoint on OUR replicated test set gives an error close to its published one
     (MED 3.11 +- 3.38 mm synthetic, 4.06 +- 3.35 mm AMOS). Much worse = our replica differs from theirs.

**Phase 2 PRIORITY (user, 2026-11-19): real anatomy, distance, different shapes**
- real anatomy: P2I-LReg training-fold livers + public CT livers (AMOS, TotalSegmentator; MSD / LiTS for real tumour
  positions), all in patient axes, attachments and ligaments placed anatomically
- distance learning needs the camera-view observation (the PIVOTS copy cannot teach it: its input cloud is metric).
  Rules: CT kept at its real size in mm; lens K is an input and varies (60-90 deg, off-centre principal point); depth
  given without scale (divided by its median); wide distance range including close-ups and cut-off views (range from
  the laparoscopy literature, not from LLR-LUS). Check: distance error (mm, %) on held-out synthetic patients by distance
  and visible fraction, against the analytic size-and-centre step
- different shapes: hundreds of real livers + PIVOTS random shapes (stage 2), each then deformed by the physics

**Phase 2: our additions, one switch each, each measured against phase 1**
1. real livers in patient axes (P2I-LReg + AMOS) with the attachments and ligaments placed anatomically
2. gravity as the change from the supine CT (patient tilt)
3. explicit tool contacts from the Liver Deformation Lab recordings (press / grasp / lift)
4. static equilibrium instead of the largest transient step
5. camera-view observation for our monocular setting: mask, edge types, relative depth with scale removed and damaged
   (bumpy / flattened / slanted), tools drawn where they touch

## Downloads on gpu-a240 (into /data/khoalq, never /home; nothing downloaded yet)

| # | Dataset | What we keep | Size | Licence | Role |
|---|---|---|---|---|---|
| 1 | P2I-LReg (Kaggle) | per patient: CT liver mesh, synthetic camera poses, real camera K, real labels; NOT the large point clouds | small | as used | training-fold livers + camera poses; real test frames for comparison |
| 2 | TotalSegmentator v2.0.1 (Zenodo 10047292) | liver mask + image header (spacing, orientation) -> liver mesh in patient axes; CT deleted after meshing; only scans with the whole liver | 23.6 GB download | CC BY 4.0 (confirm on Zenodo) | main source of liver shapes |
| 3 | MSD Task 03 Liver (= LiTS training) | liver + tumour masks + header -> liver mesh + real tumour meshes | to check | CC BY-SA 4.0 | real tumour positions and sizes |
| 4 | 3D-IRCADb-01 | liver, tumour, vessel masks | small | CC BY-NC-ND 4.0 (research only, no redistribution of derived meshes) | vessels for stiffness differences (later) |
| 5 | AMOS22 (only if access is granted) | liver masks | to check | CC BY 4.0 per mirrors; latest version restricted | PIVOTS-style test livers |
| 6 | DePoLL (IRCAD) | only the CT volumes and marker positions of the 13 deformed states (not the videos) | ~110 GB in full; check parts first | CC BY-NC-ND 4.0 | calibrate deformation sizes (never trained on) |
| 7 | SERV-CT | images + true depth | small | to check | Depth Anything error kind and size (never trained on) |

Not downloaded: LLR-LUS (final test, already local), AbdomenAtlas (far more than needed), FLARE (research-only labels), CHAOS (optional).

## Deformations to simulate (the intraoperative liver vs the supine, gas-free CT)

| # | Cause in surgery | What it does to the liver | How we simulate it | Phase | Size backed by |
|---|---|---|---|---|---|
| D1 | Pneumoperitoneum (CO2, ~12-15 mmHg) | anterior support lost, diaphragm pushed up, liver shifts and its edges deform | phase 1: PIVOTS forces of varied size / place / area; phase 2: pressure on the exposed surface + moved wall / diaphragm contact | 1, 2 | DePoLL; pig CT before/under gas (IJCARS 2019, Surg Endosc 2024); tumour shift up to 28 mm (2012) |
| D2 | Patient position (head-up / reverse Trendelenburg, side tilt) | gravity pulls the liver in a new direction: sag and shift | gravity as the change from supine | 2 | positioning literature (to find) |
| D3 | Ligaments and mobilisation | ligaments hold or pull; cutting them frees lobes | springs (k 100-300 N/m, rest length x0.9-1.1) at falciform / round / coronary / triangular positions, randomly present or cut | 1 (random paths), 2 (anatomical) | PIVOTS ranges |
| D4 | Neighbouring organs (diaphragm, stomach, colon) | push the liver | placeholder organs + pushing springs (rest length > 1) | 1 | PIVOTS |
| D5 | Tools: press / retract | local dent + organ shift | displacement at a tip radius along the shaft (lab templates, varied) | 2 | lab recordings + tool-force literature |
| D6 | Tools: grasp and lift / flip a lobe | large rotation of a lobe about its attachment | prescribed displacement of a grasped patch, neo-Hookean (large deformation) | 2 | lab recordings |
| D7 | Ultrasound probe on the surface | broad indentation | wide contact (radius ~7 mm) press | 2 | lab recordings; LUS probe size |
| D8 | Breathing (ventilated) | mostly a head-feet shift + small deformation near the dome | diaphragm-side displacement; the shift goes to the pose | 2 | respiratory motion literature (to find) |
| D9 | Volume change between scan and surgery | small global swelling or shrinkage | compressible material (nu 0.45-0.48) + small uniform volume change | 2 | Leung 1986 (cited by PIVOTS) |
| D10 | Tissue differences | tumours and vessels are stiffer; cirrhotic livers stiffer overall | global stiffness 3-30 kPa (phase 1); stiffer tumour / vessel regions (phase 2, late) | 1, 2 | PIVOTS material refs; elastography literature |

Out of scope for v1: cutting / resection, bleeding, heartbeat (mm-scale), self-collision of folded lobes (noted, not solved).

## Still to back with sources (dev-set rule: existence AND size)

- pneumoperitoneum deformation size: DePoLL (13 states, porcine), porcine pre/post insufflation CT (IJCARS 2019),
  tumour shift up to 28 mm (2012 animal study), Surg Endosc 2024 porcine CT
- tool forces and indentation depths: lab recordings + literature on laparoscopic grasping forces
- Depth Anything error kind and size: SERV-CT measurement
