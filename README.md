# CryoDenoising

Code for micrograph denoising experiments around the ICPR 2026 paper
**Enhancing Micrograph Denoising via Semantic-Aware Knowledge Learning**.

The paper uses semantic priors from a foundation segmentation model to guide
cryo-EM micrograph denoising. This repository now contains a self-contained
PyTorch implementation of that main idea:

- semantic mask guided feature correction;
- cross-local relationship modeling for semantic regions;
- spatial-frequency complementary modeling for non-semantic/noise regions;
- multi-scale fusion for final denoising;
- Noise2Noise-style training with odd/even frame micrograph pairs.

## Repository Layout

```text
.
|-- train_cryo.py                         # training entry point
|-- test.py                               # inference/evaluation entry point
|-- models/
|   |-- Semantic_SR_model.py              # training wrapper for semantic_sr
|   |-- networks.py                       # lazy network factory
|   `-- archs/semantic_aware_denoiser.py  # semantic-aware denoising network
|-- data/
|   `-- LQGT_semantic_dataset.py          # LQ/GT/mask dataset
|-- options/
|   |-- train_cryo/SemanticAware_10017.yml
|   `-- test/SemanticAware_10017.yml
|-- CryoSegNet-main/                      # optional mask generation/particle picking code
`-- requirements.txt
```


## Installation

Create an environment with Python 3.7+ and install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from
the official PyTorch instructions before installing the rest of the requirements.

The optional `CryoSegNet-main` subproject has its own environment file:

```bash
cd CryoSegNet-main
conda env create -f environment.yml
conda activate cryosegnet
```

## Data Format

The semantic denoising dataset expects paired noisy observations of the same
underlying micrograph, following the paper's Noise2Noise setup:

```text
datasets/
|-- train/
|   |-- odd/       # noisy input micrographs
|   |-- even/      # independent noisy targets
|   `-- masks/     # optional semantic masks aligned with odd/
`-- 10017/
    |-- odd/
    |-- even/
    `-- masks/
```

Files are paired by sorted order, so keep matching filenames or make sure the
directory sort order is identical across `odd`, `even`, and `masks`.

Supported image formats are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.ppm`, and `.mat`
where supported by the existing data utilities. Masks can be grayscale or RGB
images and are converted to one channel in `[0, 1]`.

If masks are not available, set `dataroot_mask: ~`. The model will still run,
but it will no longer use semantic priors.

## Semantic Mask Generation

The paper uses an enhanced segmentation prior based on CryoSegNet/SAM. This
checkout includes `CryoSegNet-main` as a separate project. A typical workflow is:

```bash
cd CryoSegNet-main
python predict_new_data_jpg.py --my_dataset_path ../datasets/train/odd --output_path ../datasets/train/seg_output
```

Then convert or copy the predicted binary mask images into:

```text
datasets/train/masks
datasets/10017/masks
```

The exact mask filenames should align with the noisy input images after sorting.
See `CryoSegNet-main/README.md` for model downloads and additional MRC/JPG
prediction commands.

## Training

Edit paths in `options/train_cryo/SemanticAware_10017.yml`, then run:

```bash
python train_cryo.py -opt options/train_cryo/SemanticAware_10017.yml
```

Important fields:

```yaml
datasets:
  train:
    dataroot_LQ: datasets/train/odd
    dataroot_GT: datasets/train/even
    dataroot_mask: datasets/train/masks
    GT_size: 128
    batch_size: 8
train:
  pixel_criterion: mse
  lr_G: 1e-3
```

Checkpoints and logs are written to:

```text
experiments/SemanticAware_10017/
results/val_images/SemanticAware_10017/
```

## Testing

Edit `options/test/SemanticAware_10017.yml` so `pretrain_model_G` points to a
checkpoint, then run inference:

```bash
python test.py -opt options/test/SemanticAware_10017.yml
```

To calculate PSNR/SSIM against the target directory:

```bash
python test.py -opt options/test/SemanticAware_10017.yml --with_gt
```

Outputs are saved under:

```text
results/SemanticAware_10017/
```
