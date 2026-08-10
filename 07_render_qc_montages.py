#!/usr/bin/env python3
"""
07_render_qc_montages.py — render PET + GT + per-model predictions for a handful
of representative cases for Oona's handoff bundle.

Selects 5 "winners" (D007 mean Dice in top tier across all 3 models) and 5
"losers" (positives missed by all 3 models). For each case, picks the
ground-truth slice with most tumor and renders 5 panels:
  PET | GT contour | D006 pred contour | D007 pred contour | D008 pred contour

Output: handoff_oona/qc_montages/{winners,losers}/{anom_folder}_{case_id}.png
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
ANOM = ROOT / 'data' / 'anom_data'

# ---------- pick cases from manifest ----------
manifest = list(csv.DictReader((HANDOFF / 'handoff_manifest.csv').open()))
for r in manifest:
    for col in ('d006_dice', 'd007_dice', 'd008_dice'):
        try:
            r[col] = float(r[col])
        except (ValueError, KeyError):
            r[col] = float('nan')

positives = [r for r in manifest if r['label'] == 'pos']

# Winners: positives where all 3 models did well (sort by mean dice across the 3)
def mean3(r):
    vals = [r['d006_dice'], r['d007_dice'], r['d008_dice']]
    vals = [v for v in vals if v == v]  # drop NaN
    return sum(vals) / len(vals) if vals else 0

winners = sorted(positives, key=mean3, reverse=True)[:5]
# Near-miss cases: at least one model produced a prediction but mean Dice is poor (0.05-0.30).
# These are diagnostic — they show WHERE the models thought the tumor was vs where it actually was.
near_miss = [r for r in positives if 0.05 < mean3(r) < 0.30]
near_miss.sort(key=mean3)  # worst first
losers = near_miss[:5]

print('winners  :', [(r['anom_folder'], r['case_id'], f"{mean3(r):.3f}") for r in winners])
print('near-miss:', [(r['anom_folder'], r['case_id'], f"{mean3(r):.3f}") for r in losers])

# ---------- helpers ----------
def find_pet_hdr(anom_folder: str) -> Path | None:
    for parent in (ANOM / 'positiiviset', ANOM / 'negatiiviset'):
        pdir = parent / anom_folder
        if not pdir.exists():
            continue
        for sub in pdir.iterdir():
            if sub.is_dir() and 'pet' in sub.name.lower() and 'maski' not in sub.name.lower() and 'mri' not in sub.name.lower():
                hdrs = sorted(list(sub.glob('*.hdr')) + list(sub.glob('*.HDR')))
                if hdrs:
                    return hdrs[0]
    return None

def find_mask_hdr(anom_folder: str) -> Path | None:
    for parent in (ANOM / 'positiiviset',):
        pdir = parent / anom_folder
        if not pdir.exists():
            continue
        for sub in pdir.iterdir():
            if sub.is_dir() and 'maski' in sub.name.lower():
                hdrs = sorted(list(sub.glob('*.hdr')) + list(sub.glob('*.HDR')))
                if hdrs:
                    return hdrs[0]
    return None

def load_nifti_arr(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

def fold_pred_path(ds_short: str, fold: int, case_id: str) -> Path:
    return HANDOFF / ds_short / f'fold_{fold}' / f'{case_id}.nii.gz'

# ---------- render ----------
def render(group: str, case_rows):
    out_dir = HANDOFF / 'qc_montages' / group
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in case_rows:
        cid = r['case_id']
        af = r['anom_folder']
        fold = int(r['fold'])
        pet_hdr = find_pet_hdr(af)
        msk_hdr = find_mask_hdr(af)
        if not pet_hdr:
            print(f'  skip {cid} {af}: no PET found'); continue
        pet = load_nifti_arr(pet_hdr)
        gt = (load_nifti_arr(msk_hdr) > 0.5).astype(np.uint8) if msk_hdr else np.zeros_like(pet, dtype=np.uint8)

        d6 = load_nifti_arr(fold_pred_path('D006_PET', fold, cid))
        d7 = load_nifti_arr(fold_pred_path('D007_PETMRI', fold, cid))
        d8 = load_nifti_arr(fold_pred_path('D008_PETMRIcond', fold, cid))

        # pick slice with most GT (or fall back to slice with most prediction if no GT)
        if gt.sum() > 0:
            z = int(np.argmax(gt.sum(axis=(1,2))))
        elif d7.sum() > 0:
            z = int(np.argmax(d7.sum(axis=(1,2))))
        else:
            z = pet.shape[0] // 2

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
        lo, hi = np.percentile(pet[z], 1), np.percentile(pet[z], 99)
        titles = [
            f'PET (slice {z}/{pet.shape[0]-1})',
            f'GT mask (red)  ref={int(gt.sum())} vox',
            f'D006 PET  dice={r["d006_dice"]:.3f}',
            f'D007 PET+MRI  dice={r["d007_dice"]:.3f}',
            f'D008 PET+T1+T2  dice={r["d008_dice"]:.3f}',
        ]
        preds = [None, gt, d6, d7, d8]
        for ax, title, pred in zip(axes, titles, preds):
            ax.imshow(pet[z], cmap='inferno', vmin=lo, vmax=hi)
            if pred is not None and pred[z].any():
                color = 'lime' if title.startswith('GT') is False and not title.startswith('GT') else 'red'
                color = 'red' if title.startswith('GT') else 'lime'
                ax.contour(pred[z], levels=[0.5], colors=color, linewidths=1.2)
            if title.startswith('GT') is False and pred is not None and pred is not gt:
                # also draw GT in red for context
                if gt[z].any():
                    ax.contour(gt[z], levels=[0.5], colors='red', linewidths=0.8, linestyles='--')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        plt.suptitle(f'{group.upper()} — anom_folder {af}  ({cid}, fold {fold}, seq={r["mri_sequence"]})',
                     fontsize=12, y=1.02)
        plt.tight_layout()
        out = out_dir / f'{af}_{cid}.png'
        plt.savefig(out, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'  wrote {out.name}')

render('winners', winners)
# replace existing losers/ dir with the near-miss montages
import shutil
shutil.rmtree(HANDOFF / 'qc_montages' / 'losers', ignore_errors=True)
render('near_miss',  losers)
print('done')
