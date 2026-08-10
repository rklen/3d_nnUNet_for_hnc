#!/usr/bin/env python3
"""
09_render_multislice.py — render multi-slice (1 row × N slices) PET montages
with GT contour (red) and D007 prediction contour (lime) so the 3D location
of the mask is visible.

For each requested case: 7 evenly-spaced slices through the mask z-extent
(or whole volume if no mask).
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/jussihirvonen/Dropbox/HNMRI/PET-MRI-cancer')
HANDOFF = ROOT / 'handoff_oona'
ANOM = ROOT / 'data' / 'anom_data' / 'positiiviset'

# cases: list of (group, case_id, anom_folder, fold)
manifest = {r['case_id']: r for r in csv.DictReader((HANDOFF/'handoff_manifest.csv').open())}

NEAR_MISS = ['case_0055', 'case_0062', 'case_0025', 'case_0069', 'case_0084']
BRAIN_AUDIT = ['case_0020', 'case_0030', 'case_0056']  # case_0084 already in NEAR_MISS

groups = (
    ('near_miss', NEAR_MISS),
    ('brain_audit', BRAIN_AUDIT),
)

def find_pet_hdr(af):
    pdir = ANOM / af
    for sub in pdir.iterdir():
        if sub.is_dir() and 'pet' in sub.name.lower() and 'maski' not in sub.name.lower() and 'mri' not in sub.name.lower():
            hdrs = sorted(list(sub.glob('*.hdr')) + list(sub.glob('*.HDR')))
            if hdrs: return hdrs[0]
    return None

def find_mask_hdr(af):
    pdir = ANOM / af
    for sub in pdir.iterdir():
        if sub.is_dir() and 'maski' in sub.name.lower():
            hdrs = sorted(list(sub.glob('*.hdr')) + list(sub.glob('*.HDR')))
            if hdrs: return hdrs[0]
    return None

def load(p): return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))

OUT_DIR = HANDOFF / 'qc_montages' / 'multislice'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SLICES = 7
for group, cases in groups:
    for cid in cases:
        r = manifest[cid]
        af = r['anom_folder']
        fold = int(r['fold'])
        pet = load(find_pet_hdr(af))
        gt_hdr = find_mask_hdr(af)
        gt = (load(gt_hdr) > 0.5).astype(np.uint8) if gt_hdr else np.zeros_like(pet, dtype=np.uint8)
        d7 = load(HANDOFF / 'D007_PETMRI' / f'fold_{fold}' / f'{cid}.nii.gz')

        # pick N evenly-spaced slices through the GT z-range
        nz = pet.shape[0]
        zs = np.where(gt.sum(axis=(1,2)) > 0)[0]
        if zs.size:
            zmin, zmax = int(zs.min()), int(zs.max())
            # pad span a bit
            span = max(zmax - zmin, 5)
            zmin = max(0, zmin - span // 4)
            zmax = min(nz - 1, zmax + span // 4)
        else:
            zmin, zmax = 0, nz - 1
        slices = np.linspace(zmin, zmax, N_SLICES).round().astype(int)

        fig, axes = plt.subplots(1, N_SLICES, figsize=(3*N_SLICES, 3.5))
        for ax, z in zip(axes, slices):
            sl = pet[z]
            lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
            ax.imshow(sl, cmap='inferno', vmin=lo, vmax=hi)
            if gt[z].any():
                ax.contour(gt[z], levels=[0.5], colors='red', linewidths=1.2)
            if d7[z].any():
                ax.contour(d7[z], levels=[0.5], colors='lime', linewidths=1.0)
            ax.set_title(f'z={z}/{nz-1}', fontsize=10)
            ax.axis('off')

        d6_dice = float(r['d006_dice']) if r['d006_dice'] not in ('', 'nan') else float('nan')
        d7_dice = float(r['d007_dice']) if r['d007_dice'] not in ('', 'nan') else float('nan')
        d8_dice = float(r['d008_dice']) if r['d008_dice'] not in ('', 'nan') else float('nan')
        plt.suptitle(f'{group.upper()} — pt {af} ({cid}, seq={r["mri_sequence"]}, fold {fold})\n'
                     f'red = GT,  lime = D007 pred  |  Dice: D006={d6_dice:.3f}  D007={d7_dice:.3f}  D008={d8_dice:.3f}'
                     f'  |  GT z-span {zmin}-{zmax}',
                     fontsize=11, y=1.02)
        plt.tight_layout()
        out = OUT_DIR / f'{group}_{af}_{cid}.png'
        plt.savefig(out, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'  wrote {out.name}')
