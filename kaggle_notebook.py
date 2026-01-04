# Generated from kaggle_notebook.ipynb — code cells only
# Cell 1: Install dependencies
# Install dependencies (quietly). Kaggle often has numpy/scipy/torch preinstalled.
import sys, subprocess

def pip_install(req_path="requirements.txt"):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "-r", req_path])
        print("Installed requirements from", req_path)
    except Exception as e:
        print("pip install failed:", e)

pip_install()

# Cell 2: Configure imports and paths
import os, sys
from pathlib import Path
sys.path.append(".")
from src.infer import run_inference
from src.scripts.pack_submission import pack as pack_submission

candidate_dirs = [
    "/kaggle/input/vesuvius-challenge-surface-detection/test_images",
    "/kaggle/input/vesuvius-challenge-surface-detection/test",
    "./data/test_images",
    "./data/raw/test_images",
    "./data/raw/test",
]
IMAGES_DIR = next((d for d in candidate_dirs if os.path.isdir(d)), None)
if IMAGES_DIR is None:
    raise RuntimeError("Test images directory not found. Update IMAGES_DIR accordingly.")
OUTPUT_MASKS_DIR = "./submission_masks"
CHECKPOINT_PATH = "./outputs/model.pt" if os.path.isfile("./outputs/model.pt") else None
print("IMAGES_DIR:", IMAGES_DIR)
print("CHECKPOINT_PATH:", CHECKPOINT_PATH)

# Cell 3: Optional training (commented out)
# from src.train import train as train_model
# class Args: pass
# args = Args(); args.images_dir = IMAGES_DIR; args.output_dir = "./outputs"; args.epochs = 1
# train_model(args)

# Cell 4: Inference
run_inference(IMAGES_DIR, OUTPUT_MASKS_DIR, CHECKPOINT_PATH)

# Cell 5: Pack submission
SUBMISSION_ZIP = "./submission.zip"
pack_submission(OUTPUT_MASKS_DIR, SUBMISSION_ZIP)
print("Submission size (MB):", round(os.path.getsize(SUBMISSION_ZIP)/1e6, 3))

# Cell 6: Sanity check
from tifffile import imread
masks = sorted(Path(OUTPUT_MASKS_DIR).glob("*.tif"))
if masks:
    m = imread(str(masks[0]))
    print("Mask shape:", m.shape, "dtype:", m.dtype, "min/max:", int(m.min()), int(m.max()))
else:
    print("No masks written in", OUTPUT_MASKS_DIR)
