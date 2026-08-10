#!/usr/bin/env python3
"""
08_audit_mask_location.py

For every positive patient, compute where the GT mask lives along the z
(slice) axis as a fraction of the full PET volume. Flag cases whose mask
centroid is in the upper portion of the volume (likely brain rather than
H&N).

Reads raw Analyze masks/PET from data/anom_data/positiiviset/.
Writes results/mask_location_audit.csv and prints a summary.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path('/Users/jussihirvonen/Dropbox/HNMRI/PET-MRI-cancer')
ANOM = ROOT / 'data' / 'anom_data' / 'positiiviset'
MANIFEST = ROOT / 'patient_manifest.csv'

# Map anom_folder -> case_id from manifest
manifest = {r['anom_folder']: r for r in csv.DictReader(MANIFEST.open()) if r['label'] == 'pos'}

def find_subfolder(pdir: Path, keyword: str) -> Path | None:
    for c in pdir.iterdir():
        if c.is_dir() and keyword.lower() in c.name.lower():
            return c
    return None

def find_pet_folder(pdir: Path) -> Path | None:
    for c in pdir.iterdir():
        if not c.is_dir(): continue
        n = c.name.lower()
        if 'pet' in n and 'maski' not in n and 'mri' not in n:
            return c
    return None

def find_hdr(folder: Path) -> Path | None:
    hdrs = sorted(list(folder.glob('*.hdr')) + list(folder.glob('*.HDR')))
    return hdrs[0] if hdrs else None

rows = []
for pdir in sorted(ANOM.iterdir()):
    if not pdir.is_dir(): continue
    af = pdir.name
    m = manifest.get(af, {})
    case_id = m.get('case_id', '?')
    seq = m.get('mri_sequence', '?')

    pet_folder = find_pet_folder(pdir)
    msk_folder = find_subfolder(pdir, 'maski')
    if not pet_folder or not msk_folder:
        continue
    pet_hdr = find_hdr(pet_folder)
    msk_hdr = find_hdr(msk_folder)
    if not pet_hdr or not msk_hdr:
        continue

    pet = sitk.GetArrayFromImage(sitk.ReadImage(str(pet_hdr)))  # (z, y, x)
    msk = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_hdr))) > 0.5
    nz = pet.shape[0]
    if msk.shape != pet.shape:
        rows.append({'case_id': case_id, 'anom_folder': af, 'mri_sequence': seq,
                     'note': f'shape mismatch pet={pet.shape} mask={msk.shape}',
                     'centroid_z_frac': '', 'min_z_frac': '', 'max_z_frac': '',
                     'n_ref_vox': int(msk.sum()), 'nz': nz, 'possibly_brain': ''})
        continue

    # mask voxel z indices
    zs = np.where(msk.sum(axis=(1,2)) > 0)[0]
    if zs.size == 0:
        rows.append({'case_id': case_id, 'anom_folder': af, 'mri_sequence': seq,
                     'note': 'empty mask', 'centroid_z_frac': '',
                     'min_z_frac': '', 'max_z_frac': '',
                     'n_ref_vox': 0, 'nz': nz, 'possibly_brain': ''})
        continue

    # weighted centroid by per-slice mask voxel count
    weights = msk.sum(axis=(1,2))[zs]
    centroid_z = float((zs * weights).sum() / weights.sum())
    centroid_z_frac = centroid_z / (nz - 1)
    min_z_frac = float(zs.min()) / (nz - 1)
    max_z_frac = float(zs.max()) / (nz - 1)

    # In PET-MRI head/neck imaging, the cohort PET volumes have z increasing
    # from inferior (neck) to superior (head). A mask whose centroid is in
    # the upper quarter (z_frac > 0.75) is suspicious for being in the brain.
    flag = 'POSSIBLE_BRAIN' if centroid_z_frac > 0.75 else (
           'UPPER_HALF' if centroid_z_frac > 0.5 else '')

    rows.append({'case_id': case_id, 'anom_folder': af, 'mri_sequence': seq,
                 'note': '', 'centroid_z_frac': f'{centroid_z_frac:.3f}',
                 'min_z_frac': f'{min_z_frac:.3f}', 'max_z_frac': f'{max_z_frac:.3f}',
                 'n_ref_vox': int(msk.sum()), 'nz': nz, 'possibly_brain': flag})

out = ROOT / 'results' / 'mask_location_audit.csv'
with out.open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {out}')

# Summary
total = sum(1 for r in rows if r['centroid_z_frac'])
brain = sum(1 for r in rows if r['possibly_brain'] == 'POSSIBLE_BRAIN')
upper = sum(1 for r in rows if r['possibly_brain'] in ('POSSIBLE_BRAIN', 'UPPER_HALF'))
print(f'\n{total} positives with valid mask location')
print(f'  Centroid in upper QUARTER (likely brain): {brain}')
print(f'  Centroid in upper HALF: {upper}')
print(f'\nPOSSIBLE_BRAIN cases:')
for r in rows:
    if r['possibly_brain'] == 'POSSIBLE_BRAIN':
        print(f'  {r["case_id"]} (anom {r["anom_folder"]}, {r["mri_sequence"]}): centroid_z={r["centroid_z_frac"]}, span {r["min_z_frac"]}-{r["max_z_frac"]}, {r["n_ref_vox"]} vox')
