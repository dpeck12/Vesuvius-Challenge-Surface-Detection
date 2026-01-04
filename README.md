# Vesuvius Challenge — Surface Detection

Basic, runnable Python pipeline for 3D surface segmentation on CT papyrus volumes.
Emphasizes tiling, overlap-aware stitching, and topology-friendly postprocessing.
Produces per-volume masks as `.tif` files ready to zip for Kaggle submission.

## Quick Start
- Create a Python environment and install dependencies (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Run a synthetic sanity test (no real data needed):

```powershell
python -m src.scripts.sanity
```

- Inference on local test images and write masks:

```powershell
python -m src.infer --images_dir path/to/test_images --output_dir ./submission_masks
python src\scripts\pack_submission.py --masks_dir ./submission_masks --output_zip ./submission.zip
```

## Kaggle Usage
- Notebook: use `kaggle_notebook.ipynb` which installs deps, detects input paths, runs inference, and packs `submission.zip` at repo root.
- Single-cell script: copy the content of `standalone_inference.py` into one Kaggle notebook cell and run. It inlines all logic (no repo imports), auto-installs minimal packages, ignores Jupyter kernel args, and creates `submission.zip`.

## Repo Structure
- `src/`
	- `config.py`: Defaults for tiling, overlap, dtype, thresholds
	- `data.py`: TIF volume I/O helpers
	- `tiling.py`: 3D sliding-window tiler with overlap and weighted stitching
	- `postprocess.py`: Morphological and connectivity cleanup for thin surfaces
	- `model.py`: Minimal 3D U-Net (PyTorch)
	- `train.py`: Training skeleton (volume-wise)
	- `infer.py`: CLI to predict masks and write `.tif` outputs
	- `scripts/pack_submission.py`: Zip masks into `submission.zip`
	- `scripts/sanity.py`: Synthetic test to verify the pipeline without data
- `kaggle_notebook.ipynb`: Kaggle-ready notebook (inference + packing)
- `standalone_inference.py`: Single-file inference + packing (no repo imports)

## Notes
- Configure output mask dtype in `src/config.py` to match competition expectations.
- On Windows, installs may use CPU-only PyTorch; training large volumes is best done on GPU.