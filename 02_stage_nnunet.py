#!/usr/bin/env python3
"""
02_stage_nnunet.py

Build Dataset006_HNCancer_PET (1ch: PET) and Dataset007_HNCancer_PETMRI
(2ch: PET + MRI) for nnU-Net v2 from anom_data/.

Per Oona 2026-05-28:
  - Use all 221 pts (104 pos + 117 neg) in both datasets.
  - One heterogeneous MRI channel for D007 (option a) — sequence per scan.
  - Binarize masks: m > 0.5.
  - Patient 35 kept in both.

Writes:
  nnunet_staging/Dataset006_HNCancer_PET/
    imagesTr/case_XXXX_0000.nii.gz   (PET)
    labelsTr/case_XXXX.nii.gz
    dataset.json
  nnunet_staging/Dataset007_HNCancer_PETMRI/
    imagesTr/case_XXXX_0000.nii.gz   (PET)
    imagesTr/case_XXXX_0001.nii.gz   (MRI)
    labelsTr/case_XXXX.nii.gz
    dataset.json
  patient_manifest.csv  (case_XXXX -> anom_data folder, label, mri_sequence)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def find_subfolder(pdir: Path, keyword: str) -> Path | None:
    """Case-insensitive: first child folder whose name contains keyword."""
    for c in pdir.iterdir():
        if c.is_dir() and keyword.lower() in c.name.lower():
            return c
    return None


def find_pet_folder(pdir: Path) -> Path | None:
    """PET folder: name contains 'pet' but not 'maski' and not 'mri'."""
    for c in pdir.iterdir():
        if not c.is_dir():
            continue
        n = c.name.lower()
        if 'pet' in n and 'maski' not in n and 'mri' not in n:
            return c
    return None


def find_mri_folder(pdir: Path) -> Path | None:
    for c in pdir.iterdir():
        if not c.is_dir():
            continue
        n = c.name.lower()
        if 'mri' in n and 'maski' not in n:
            return c
    return None


def find_hdr(folder: Path) -> Path | None:
    hdrs = sorted(list(folder.glob('*.hdr')) + list(folder.glob('*.HDR')))
    return hdrs[0] if hdrs else None


def parse_mri_sequence(folder: Path, hdr: Path) -> str:
    """Return sequence label like 't2w', 'spir', 't1c', 't1w', 't1w_tse', 'tse', 'unknown'."""
    # try filename first (most specific), then folder name
    for src in (hdr.stem.lower(), folder.name.lower()):
        m = re.search(r'mri[_-]([a-z0-9_]+)', src)
        if m:
            return m.group(1)
    return 'unknown'


def read_analyze(hdr: Path) -> sitk.Image:
    return sitk.ReadImage(str(hdr))


def save_nifti(img: sitk.Image, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out), useCompression=True)


def make_zero_mask(ref: sitk.Image) -> sitk.Image:
    arr = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return out


def binarize_mask(img: sitk.Image, ref: sitk.Image | None = None) -> sitk.Image:
    """Binarize mask. If ref provided, stamp ref's geometry on the result
    (use this when raw mask Analyze header has lost origin/direction but
    voxel array is co-registered to ref by construction)."""
    arr = (sitk.GetArrayFromImage(img) > 0.5).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    if ref is not None:
        out.CopyInformation(ref)
    else:
        out.CopyInformation(img)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True, help='anom_data root')
    ap.add_argument('--out',  type=Path, required=True, help='nnunet_staging root')
    ap.add_argument('--manifest', type=Path, required=True, help='patient_manifest.csv path')
    args = ap.parse_args()

    d006 = args.out / 'Dataset006_HNCancer_PET'
    d007 = args.out / 'Dataset007_HNCancer_PETMRI'
    for d in (d006, d007):
        (d / 'imagesTr').mkdir(parents=True, exist_ok=True)
        (d / 'labelsTr').mkdir(parents=True, exist_ok=True)

    # collect patients, sort deterministically: positives first (sorted by folder name), then negatives
    pos = sorted([p for p in (args.root/'positiiviset').iterdir() if p.is_dir()], key=lambda p: p.name)
    neg = sorted([p for p in (args.root/'negatiiviset').iterdir() if p.is_dir()], key=lambda p: p.name)
    patients = [('pos', p) for p in pos] + [('neg', p) for p in neg]

    manifest_rows = []
    n_ok = n_skip = 0
    for idx, (label, pdir) in enumerate(patients, start=1):
        case_id = f'case_{idx:04d}'

        pet_folder = find_pet_folder(pdir)
        mri_folder = find_mri_folder(pdir)
        mask_folder = find_subfolder(pdir, 'maski')

        if pet_folder is None or mri_folder is None:
            print(f'SKIP {case_id} {label} {pdir.name}: missing PET or MRI folder')
            manifest_rows.append({'case_id': case_id, 'anom_folder': pdir.name, 'label': label,
                                  'mri_sequence': '', 'status': 'skipped_missing_modality'})
            n_skip += 1
            continue

        pet_hdr  = find_hdr(pet_folder)
        mri_hdr  = find_hdr(mri_folder)
        mask_hdr = find_hdr(mask_folder) if mask_folder else None

        if pet_hdr is None or mri_hdr is None:
            print(f'SKIP {case_id} {label} {pdir.name}: missing .hdr files')
            manifest_rows.append({'case_id': case_id, 'anom_folder': pdir.name, 'label': label,
                                  'mri_sequence': '', 'status': 'skipped_missing_hdr'})
            n_skip += 1
            continue

        if label == 'pos' and mask_hdr is None:
            print(f'SKIP {case_id} {label} {pdir.name}: positive but no mask .hdr')
            manifest_rows.append({'case_id': case_id, 'anom_folder': pdir.name, 'label': label,
                                  'mri_sequence': '', 'status': 'skipped_no_mask'})
            n_skip += 1
            continue

        try:
            pet = read_analyze(pet_hdr)
            mri = read_analyze(mri_hdr)

            # mask: binarize for positives, synthesize zeros for negatives, geom from PET
            if label == 'pos':
                raw_mask = read_analyze(mask_hdr)
                # If voxel arrays match (same shape), stamp PET geometry on the mask
                # since Analyze headers often lose origin/direction (stored as 0,0,0 / I).
                # The voxel arrays are co-registered by construction (mask drawn on PET).
                if raw_mask.GetSize() == pet.GetSize():
                    mask = binarize_mask(raw_mask, ref=pet)
                else:
                    print(f'WARN {case_id}: mask size {raw_mask.GetSize()} != pet {pet.GetSize()}, resampling mask to PET grid then binarizing')
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(pet)
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    raw_mask = resampler.Execute(raw_mask)
                    mask = binarize_mask(raw_mask, ref=pet)
            else:
                mask = make_zero_mask(pet)

            seq = parse_mri_sequence(mri_folder, mri_hdr)

            # Dataset006: PET + label (no MRI)
            save_nifti(pet,  d006/'imagesTr'/f'{case_id}_0000.nii.gz')
            save_nifti(mask, d006/'labelsTr'/f'{case_id}.nii.gz')

            # Dataset007: PET + MRI + label. nnU-Net requires identical geometry
            # across channels. Analyze headers often have (0,0,0) origin even when
            # voxel arrays are on the same grid, so stamp PET geometry when sizes
            # match; only do a real resample if sizes/spacings differ.
            if mri.GetSize() == pet.GetSize() and np.allclose(mri.GetSpacing(), pet.GetSpacing(), atol=1e-3):
                mri_arr = sitk.GetArrayFromImage(mri)
                mri = sitk.GetImageFromArray(mri_arr)
                mri.CopyInformation(pet)
            else:
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(pet)
                resampler.SetInterpolator(sitk.sitkLinear)
                mri = resampler.Execute(mri)
            save_nifti(pet,  d007/'imagesTr'/f'{case_id}_0000.nii.gz')
            save_nifti(mri,  d007/'imagesTr'/f'{case_id}_0001.nii.gz')
            save_nifti(mask, d007/'labelsTr'/f'{case_id}.nii.gz')

            manifest_rows.append({'case_id': case_id, 'anom_folder': pdir.name, 'label': label,
                                  'mri_sequence': seq, 'status': 'ok'})
            n_ok += 1
            if idx % 20 == 0:
                print(f'  ... {idx}/{len(patients)} done')
        except Exception as exc:  # noqa: BLE001
            print(f'ERROR {case_id} {pdir.name}: {exc!r}')
            manifest_rows.append({'case_id': case_id, 'anom_folder': pdir.name, 'label': label,
                                  'mri_sequence': '', 'status': f'error:{exc!r}'})
            n_skip += 1

    # dataset.json for both
    n_train_006 = len([r for r in manifest_rows if r['status'] == 'ok'])
    dj_common = {
        'description': 'HN cancer recurrence on post-CRT PET-MRI (Liedes 2023 extension)',
        'reference': 'Liedes et al 2023, J Med Biol Eng 43:532',
        'licence': 'internal',
        'release': '0.1.0',
        'file_ending': '.nii.gz',
        'numTraining': n_train_006,
        'labels': {'background': 0, 'tumor': 1},
    }
    with (d006/'dataset.json').open('w') as fh:
        json.dump({**dj_common,
                   'channel_names': {'0': 'PET'},
                   'name': 'Dataset006_HNCancer_PET'}, fh, indent=2)
    with (d007/'dataset.json').open('w') as fh:
        json.dump({**dj_common,
                   'channel_names': {'0': 'PET', '1': 'MRI'},
                   'name': 'Dataset007_HNCancer_PETMRI'}, fh, indent=2)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['case_id', 'anom_folder', 'label', 'mri_sequence', 'status'])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f'\nDone. ok={n_ok}  skipped/errored={n_skip}')
    print(f'  D006 cases: {len(list((d006/"imagesTr").glob("*_0000.nii.gz")))}')
    print(f'  D007 cases: {len(list((d007/"imagesTr").glob("*_0000.nii.gz")))}')
    print(f'  manifest:   {args.manifest}')
    return 0 if n_skip == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
