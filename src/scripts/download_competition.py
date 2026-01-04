import argparse
from pathlib import Path
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi


def download_competition(competition: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    zip_path = out / f"{competition}.zip"
    api.competition_download_files(competition, path=str(out), force=True, quiet=False)
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out)
        print(f"Unzipped {zip_path} into {out}")
    else:
        print(f"Zip {zip_path} not found; files may already be extracted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", type=str, required=True)
    ap.add_argument("--out", type=str, default="./data/raw")
    args = ap.parse_args()
    download_competition(args.competition, args.out)


if __name__ == "__main__":
    main()