# depth-noise-estimate

How wrong is Depth Anything V2's depth SHAPE on real surgical images, and of which kind? The answer sets the depth
damage in the generator (`regsim/depth_damage.py`: bumpy / flattened / slanted), so the simulated depth noise is
measured, not guessed.

## Data

SERV-CT (Edwards et al., Medical Image Analysis 2022): 16 rectified stereo-endoscope images of two ex vivo pig
torsos (Experiment_1: 30-degree scope, Experiment_2: straight scope), each with a CT-derived depth map (~2 mm
accuracy) and a mask of pixels without a CT reference. figshare DOI 10.5522/04/26352199.v1; licence in the zip:
CC BY-NC-SA 4.0. On gpu-a240 it lives in `/data/khoalq/registration/depth-noise-estimate/data/SERV-CT`.

## Method

1. `predict.py`: Depth Anything V2 Large (relative) on every left image -> relative disparity per pixel.
2. `analyse.py`, per image:
   - align: fit the disparity's unknown scale and shift to the CT's inverse depth (least squares); the scale is
     never used afterwards, only the shape;
   - resample both depths to the generator's angular resolution (focal length 343 px = our 480-px render with a
     70-degree lens), so the degrees compare with the generator's damage settings;
   - divide both by their median and fit  z_pred ~ c + k (z_true - 1) + a x + b y + residual :
     **flattening** k, **slant** (a, b), **bumps** (residual); report the surface-tilt error each causes alone and
     in total, the bump size (autocorrelation half-width), and AbsRel for reference.
3. `summary.json` / `summary.csv`: median and 10th-90th percentile over the 16 images; `figures/`: one panel per image.

## Run (gpu-a240)

```
cd /data/khoalq/registration/code/depth-noise-estimate && sbatch run.sbatch
```

## Limits

16 images of pig tissue (not human liver, no breathing, no gas); whole scenes, not liver only (the simplified set
has no organ labels); the CT reference is ~2 mm accurate, so finer errors cannot be measured.
