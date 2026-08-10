#!/usr/bin/env python3
"""
01_inspect_anom_data.py

Walk anom_data/ and report what's actually in each patient folder before any
nnU-Net staging. Writes cohort_inventory.csv with one row per (patient, file).

Usage:
    python scripts/01_inspect_anom_data.py \
        --root data/anom_data \
        --out qc/cohort_inventory.csv

Per spec from Oona (2026-05-28): anom_data/ has subfolders for ~100 positive
and ~100 negative patients with PET + MRI niftis; positives also have a mask.
The exact folder/file layout is not yet known, so this script makes no
assumptions about naming — it lists every *.nii / *.nii.gz under each
subfolder and reads geometry headers.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def is_mask_like(arr: np.ndarray, max_unique: int = 8) -> bool:
    """Heuristic: integer-ish array with few unique values."""
    if arr.size == 0:
        return False
    sample = arr if arr.size < 5_000_000 else arr.ravel()[::100]
    uniq = np.unique(sample)
    if uniq.size > max_unique:
        return False
    return bool(np.all(np.isfinite(uniq))) and bool(
        np.allclose(uniq, np.round(uniq), atol=1e-6)
    )


def describe_image(path: Path) -> dict:
    """Return shape/spacing/origin/direction + mask diagnostics for a NIfTI."""
    row = {
        "file": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "shape": "",
        "spacing": "",
        "origin": "",
        "direction": "",
        "dtype": "",
        "intensity_min": "",
        "intensity_max": "",
        "intensity_mean": "",
        "n_unique": "",
        "looks_like_mask": "",
        "nonzero_voxels": "",
        "error": "",
    }
    try:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)  # (z, y, x)
        row["shape"] = "x".join(str(s) for s in arr.shape)
        row["spacing"] = ",".join(f"{s:.4f}" for s in img.GetSpacing())
        row["origin"] = ",".join(f"{o:.3f}" for o in img.GetOrigin())
        row["direction"] = ",".join(f"{d:.3f}" for d in img.GetDirection())
        row["dtype"] = str(arr.dtype)
        # quick stats on a subsample if huge
        sample = arr if arr.size < 5_000_000 else arr.ravel()[::100]
        row["intensity_min"] = f"{float(np.min(sample)):.4f}"
        row["intensity_max"] = f"{float(np.max(sample)):.4f}"
        row["intensity_mean"] = f"{float(np.mean(sample)):.4f}"
        uniq = np.unique(sample)
        row["n_unique"] = str(uniq.size)
        if is_mask_like(arr):
            row["looks_like_mask"] = "yes"
            row["nonzero_voxels"] = str(int(np.count_nonzero(arr)))
        else:
            row["looks_like_mask"] = "no"
    except Exception as exc:  # noqa: BLE001
        row["error"] = repr(exc)
    return row


def iter_patient_dirs(root: Path):
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="anom_data root")
    ap.add_argument("--out", type=Path, required=True, help="CSV output path")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_folder",
        "patient_id_guess",
        "label_guess",
        "n_files_in_folder",
        "file",
        "filename",
        "size_bytes",
        "shape",
        "spacing",
        "origin",
        "direction",
        "dtype",
        "intensity_min",
        "intensity_max",
        "intensity_mean",
        "n_unique",
        "looks_like_mask",
        "nonzero_voxels",
        "error",
    ]

    n_pat = 0
    n_files = 0
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for pdir in iter_patient_dirs(root):
            n_pat += 1
            # gather any nifti at any depth under the patient folder
            niftis = sorted(
                [p for p in pdir.rglob("*.nii.gz")]
                + [p for p in pdir.rglob("*.nii") if not str(p).endswith(".nii.gz")]
            )
            # weak heuristics: parent folder name may hint at label
            parent_name = pdir.parent.name.lower()
            label_guess = ""
            if "pos" in parent_name:
                label_guess = "pos"
            elif "neg" in parent_name:
                label_guess = "neg"

            if not niftis:
                writer.writerow(
                    {
                        "patient_folder": str(pdir),
                        "patient_id_guess": pdir.name,
                        "label_guess": label_guess,
                        "n_files_in_folder": 0,
                        "file": "",
                        "filename": "",
                        "size_bytes": "",
                        "shape": "",
                        "spacing": "",
                        "origin": "",
                        "direction": "",
                        "dtype": "",
                        "intensity_min": "",
                        "intensity_max": "",
                        "intensity_mean": "",
                        "n_unique": "",
                        "looks_like_mask": "",
                        "nonzero_voxels": "",
                        "error": "no .nii/.nii.gz under this folder",
                    }
                )
                continue

            for nii in niftis:
                row = describe_image(nii)
                row["patient_folder"] = str(pdir)
                row["patient_id_guess"] = pdir.name
                row["label_guess"] = label_guess
                row["n_files_in_folder"] = len(niftis)
                writer.writerow(row)
                n_files += 1

            print(f"[{n_pat:4d}] {pdir.name}: {len(niftis)} nifti(s)")

    print(f"\nDone: {n_pat} patient folders, {n_files} nifti files.")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
