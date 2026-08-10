#!/usr/bin/env python3
"""
02b_stage_d008.py

Build Dataset008_HNCancer_PETMRIcond: PET + T1_MRI + T2_MRI (3 channels) with
zero-padded channel for the absent sequence per patient. This is the
sequence-aware variant of D007 — instead of mixing T1 and T2 into one channel,
each MRI is routed to its sequence-specific channel and the other channel is
filled with zeros. The network can learn "which channel has data" implicitly.

Sequence classification (per Jussi, 2026-05-28):
  T2-family (-> channel 0002 T2_MRI): t2w, t2
  T1-family (-> channel 0001 T1_MRI): spir (=T1C), t1c, t1w, t1w_tse, tse, unknown

Reuses geometry/mask handling from 02_stage_nnunet.py: PET geometry stamped on
mask + MRI when voxel shapes match (Analyze headers often drop origin).

Writes:
  nnunet_staging/Dataset008_HNCancer_PETMRIcond/
    imagesTr/case_XXXX_0000.nii.gz   PET
    imagesTr/case_XXXX_0001.nii.gz   T1 MRI (or zeros)
    imagesTr/case_XXXX_0002.nii.gz   T2 MRI (or zeros)
    labelsTr/case_XXXX.nii.gz
    dataset.json
  patient_manifest_d008.csv   (case_id, anom_folder, label, mri_sequence,
                                channel_with_mri, status)

Patient ID numbering matches D006/D007 (same sort order) so case_XXXX maps to
the same patient across all three datasets.
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

T1_BARE_TOKENS = {'spir', 'tse', 'unknown'}  # carry T1 weighting in this cohort


def classify(seq: str) -> str:
    """Map filename-derived sequence tag to T1 or T2.
    Rules (in priority order):
      1. contains 't2'  -> T2
      2. contains 't1'  -> T1
      3. in {spir, tse, unknown} -> T1 (per Jussi's visual review 2026-05-28)
    """
    s = seq.lower()
    if 't2' in s:
        return 'T2'
    if 't1' in s:
        return 'T1'
    if s in T1_BARE_TOKENS:
        return 'T1'
    raise ValueError(f'unrecognized MRI sequence: {seq!r}')


def find_subfolder(pdir: Path, keyword: str) -> Path | None:
    for c in pdir.iterdir():
        if c.is_dir() and keyword.lower() in c.name.lower():
            return c
    return None


def find_pet_folder(pdir: Path) -> Path | None:
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
    for src in (hdr.stem.lower(), folder.name.lower()):
        m = re.search(r'mri[_-]([a-z0-9_]+)', src)
        if m:
            return m.group(1)
    return 'unknown'


def save_nifti(img: sitk.Image, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out), useCompression=True)


def zeros_like(ref: sitk.Image) -> sitk.Image:
    arr = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.float32)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return out


def stamp_geometry(src: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """Copy src's voxel data, stamp ref's origin/direction/spacing on it.
    Use when voxel arrays are co-registered but src's Analyze header is stripped."""
    arr = sitk.GetArrayFromImage(src)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return out


def binarize_mask(img: sitk.Image, ref: sitk.Image) -> sitk.Image:
    arr = (sitk.GetArrayFromImage(img) > 0.5).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out',  type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    args = ap.parse_args()

    d008 = args.out / 'Dataset008_HNCancer_PETMRIcond'
    (d008 / 'imagesTr').mkdir(parents=True, exist_ok=True)
    (d008 / 'labelsTr').mkdir(parents=True, exist_ok=True)

    pos = sorted([p for p in (args.root/'positiiviset').iterdir() if p.is_dir()], key=lambda p: p.name)
    neg = sorted([p for p in (args.root/'negatiiviset').iterdir() if p.is_dir()], key=lambda p: p.name)
    patients = [('pos', p) for p in pos] + [('neg', p) for p in neg]

    rows = []
    n_ok = n_skip = 0
    for idx, (label, pdir) in enumerate(patients, start=1):
        case_id = f'case_{idx:04d}'

        pet_folder  = find_pet_folder(pdir)
        mri_folder  = find_mri_folder(pdir)
        mask_folder = find_subfolder(pdir, 'maski')

        if pet_folder is None or mri_folder is None:
            print(f'SKIP {case_id} {pdir.name}: missing PET or MRI folder')
            rows.append({'case_id': case_id, 'anom_folder': pdir.name, 'label': label,
                         'mri_sequence': '', 'channel_with_mri': '', 'status': 'skipped'})
            n_skip += 1
            continue

        pet_hdr  = find_hdr(pet_folder)
        mri_hdr  = find_hdr(mri_folder)
        mask_hdr = find_hdr(mask_folder) if mask_folder else None
        if pet_hdr is None or mri_hdr is None or (label == 'pos' and mask_hdr is None):
            print(f'SKIP {case_id} {pdir.name}: missing .hdr')
            rows.append({'case_id': case_id, 'anom_folder': pdir.name, 'label': label,
                         'mri_sequence': '', 'channel_with_mri': '', 'status': 'skipped_no_hdr'})
            n_skip += 1
            continue

        try:
            pet = sitk.ReadImage(str(pet_hdr))
            mri_raw = sitk.ReadImage(str(mri_hdr))
            seq = parse_mri_sequence(mri_folder, mri_hdr)
            channel = classify(seq)  # 'T1' or 'T2'

            # Stamp PET geometry on MRI when voxel grids match
            if mri_raw.GetSize() == pet.GetSize() and np.allclose(mri_raw.GetSpacing(), pet.GetSpacing(), atol=1e-3):
                mri = stamp_geometry(mri_raw, pet)
            else:
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(pet)
                resampler.SetInterpolator(sitk.sitkLinear)
                mri = resampler.Execute(mri_raw)

            # Mask
            if label == 'pos':
                raw_mask = sitk.ReadImage(str(mask_hdr))
                if raw_mask.GetSize() == pet.GetSize():
                    mask = binarize_mask(raw_mask, ref=pet)
                else:
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(pet)
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    raw_mask = resampler.Execute(raw_mask)
                    mask = binarize_mask(raw_mask, ref=pet)
            else:
                arr = np.zeros(sitk.GetArrayFromImage(pet).shape, dtype=np.uint8)
                mask = sitk.GetImageFromArray(arr)
                mask.CopyInformation(pet)

            # Route MRI into T1 (channel 0001) or T2 (channel 0002); zero the other
            if channel == 'T1':
                t1_img = mri
                t2_img = zeros_like(pet)
            else:
                t1_img = zeros_like(pet)
                t2_img = mri

            save_nifti(pet,    d008/'imagesTr'/f'{case_id}_0000.nii.gz')
            save_nifti(t1_img, d008/'imagesTr'/f'{case_id}_0001.nii.gz')
            save_nifti(t2_img, d008/'imagesTr'/f'{case_id}_0002.nii.gz')
            save_nifti(mask,   d008/'labelsTr'/f'{case_id}.nii.gz')

            rows.append({'case_id': case_id, 'anom_folder': pdir.name, 'label': label,
                         'mri_sequence': seq, 'channel_with_mri': channel, 'status': 'ok'})
            n_ok += 1
            if idx % 20 == 0:
                print(f'  ... {idx}/{len(patients)} done')
        except Exception as exc:  # noqa: BLE001
            print(f'ERROR {case_id} {pdir.name}: {exc!r}')
            rows.append({'case_id': case_id, 'anom_folder': pdir.name, 'label': label,
                         'mri_sequence': '', 'channel_with_mri': '', 'status': f'error:{exc!r}'})
            n_skip += 1

    # dataset.json
    dj = {
        'name': 'Dataset008_HNCancer_PETMRIcond',
        'description': 'HN cancer recurrence on post-CRT PET-MRI, MRI routed by sequence (T1 vs T2) into separate channels',
        'reference': 'Liedes et al 2023, J Med Biol Eng 43:532; sister to Dataset006/007',
        'licence': 'internal',
        'release': '0.1.0',
        'file_ending': '.nii.gz',
        'numTraining': n_ok,
        'labels': {'background': 0, 'tumor': 1},
        'channel_names': {'0': 'PET', '1': 'T1_MRI', '2': 'T2_MRI'},
    }
    with (d008/'dataset.json').open('w') as fh:
        json.dump(dj, fh, indent=2)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['case_id', 'anom_folder', 'label',
                                           'mri_sequence', 'channel_with_mri', 'status'])
        w.writeheader()
        w.writerows(rows)

    # Quick channel-fill summary
    n_t1 = sum(1 for r in rows if r['channel_with_mri'] == 'T1')
    n_t2 = sum(1 for r in rows if r['channel_with_mri'] == 'T2')
    print(f'\nDone. ok={n_ok}  skipped={n_skip}')
    print(f'  T1 channel populated: {n_t1}')
    print(f'  T2 channel populated: {n_t2}')
    print(f'  manifest: {args.manifest}')
    return 0 if n_skip == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
