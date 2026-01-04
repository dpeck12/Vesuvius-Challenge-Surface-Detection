import argparse
from pathlib import Path
import zipfile


def pack(masks_dir: str, output_zip: str):
    src = Path(masks_dir)
    out = Path(output_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(src.glob("*.tif"))
    if not tif_files:
        raise RuntimeError(f"No .tif masks found in {src}")
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in tif_files:
            # Store at archive root as [image_id].tif per Kaggle format
            zf.write(f, arcname=f.name)
    print(f"Created {out} with {len(tif_files)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_dir", type=str, required=True)
    ap.add_argument("--output_zip", type=str, default="submission.zip")
    args = ap.parse_args()
    pack(args.masks_dir, args.output_zip)


if __name__ == "__main__":
    main()