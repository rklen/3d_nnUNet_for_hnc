#!/usr/bin/env python3
"""
10_slice_level_liedes_comparable.py

Compute slice-level metrics from our 3D OOF predictions in the same shape as
Liedes 2023 reported:
  - per slice, label as TP / TN / FP / FN
    TP = GT has tumor AND model predicted tumor
    TN = GT empty   AND model predicted empty
    FP = GT empty   AND model predicted something
    FN = GT has tumor AND model predicted empty
  - Dice computed only on TP slices (Liedes Table 2 "TP-only mean")
  - Slice-level sensitivity / specificity (Liedes Table 3)
  - 5-fold mean ± SD per dataset

Reads GT from original Analyze masks in data/anom_data/, predictions from
handoff_oona/D00X/fold_Y/case_XXXX.nii.gz.

Outputs:
  results/slice_level_metrics.csv      (per dataset × fold)
  results/slice_level_summary.txt      (human-readable, Liedes-comparable)
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import SimpleITK as sitk

ROOT = Path('/Users/jussihirvonen/Dropbox/HNMRI/PET-MRI-cancer')
HANDOFF = ROOT / 'handoff_oona'
ANOM = ROOT / 'data' / 'anom_data'

manifest = {r['case_id']: r for r in csv.DictReader((HANDOFF/'handoff_manifest.csv').open())}

def find_subfolder(pdir, kw):
    for c in pdir.iterdir():
        if c.is_dir() and kw.lower() in c.name.lower():
            return c
    return None

def find_pet_folder(pdir):
    for c in pdir.iterdir():
        if not c.is_dir(): continue
        n = c.name.lower()
        if 'pet' in n and 'maski' not in n and 'mri' not in n:
            return c
    return None

def find_hdr(folder):
    if folder is None: return None
    hdrs = sorted(list(folder.glob('*.hdr')) + list(folder.glob('*.HDR')))
    return hdrs[0] if hdrs else None

def load_arr(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))

def get_gt(case_id):
    r = manifest[case_id]
    af = r['anom_folder']
    if r['label'] == 'pos':
        pdir = ANOM / 'positiiviset' / af
        msk = find_hdr(find_subfolder(pdir, 'maski'))
        pet = find_hdr(find_pet_folder(pdir))
        if msk is None or pet is None: return None
        pet_arr = load_arr(pet)
        m = load_arr(msk)
        # Reuse staging guarantee: if shape matches PET, use as-is binarized
        if m.shape == pet_arr.shape:
            return (m > 0.5).astype(np.uint8), pet_arr.shape
        # else resample to PET shape
        # (rare; staging script handled this)
        from SimpleITK import ResampleImageFilter, sitkNearestNeighbor
        ref = sitk.ReadImage(str(pet))
        mi  = sitk.ReadImage(str(msk))
        rs = ResampleImageFilter(); rs.SetReferenceImage(ref); rs.SetInterpolator(sitkNearestNeighbor)
        out = rs.Execute(mi)
        a = sitk.GetArrayFromImage(out)
        return (a > 0.5).astype(np.uint8), pet_arr.shape
    else:
        # negative: all-zero of PET shape
        pdir = ANOM / 'negatiiviset' / af
        pet = find_hdr(find_pet_folder(pdir))
        if pet is None: return None
        shp = load_arr(pet).shape
        return np.zeros(shp, dtype=np.uint8), shp

def slice_metrics(gt, pred):
    """Return list of (gt_has, pred_has, dice_2d) per slice."""
    out = []
    for z in range(gt.shape[0]):
        gs = gt[z]; ps = pred[z]
        gh = gs.any(); ph = ps.any()
        inter = int(np.logical_and(gs, ps).sum())
        gsum = int(gs.sum()); psum = int(ps.sum())
        denom = gsum + psum
        d = (2 * inter / denom) if denom > 0 else float('nan')
        out.append((bool(gh), bool(ph), d))
    return out

# ---------- run ----------
fold_rows = []
ds_rollups = {}

for ds_short, ds_dir in (('D006', 'D006_PET'), ('D007', 'D007_PETMRI'), ('D008', 'D008_PETMRIcond')):
    per_fold = []
    for f in range(5):
        pred_dir = HANDOFF / ds_dir / f'fold_{f}'
        cases = sorted([p.name.split('.')[0] for p in pred_dir.glob('*.nii.gz')])

        n_TP = n_TN = n_FP = n_FN = 0
        tp_dices = []
        for cid in cases:
            gt_pack = get_gt(cid)
            if gt_pack is None:
                continue
            gt, shp = gt_pack
            pred = load_arr(pred_dir / f'{cid}.nii.gz')
            if pred.shape != gt.shape:
                # mismatch shouldn't happen since staging fixed geometry
                continue

            for gh, ph, d in slice_metrics(gt, pred):
                if   gh and ph:   n_TP += 1; tp_dices.append(d)
                elif gh and not ph: n_FN += 1
                elif not gh and ph: n_FP += 1
                else:             n_TN += 1

        sens = n_TP / (n_TP + n_FN) if (n_TP + n_FN) else float('nan')
        spec = n_TN / (n_TN + n_FP) if (n_TN + n_FP) else float('nan')
        tp_dice_mean = float(np.mean(tp_dices)) if tp_dices else float('nan')
        tp_dice_med  = float(np.median(tp_dices)) if tp_dices else float('nan')

        per_fold.append({
            'dataset': ds_short, 'fold': f,
            'n_TP_slices': n_TP, 'n_TN_slices': n_TN,
            'n_FP_slices': n_FP, 'n_FN_slices': n_FN,
            'sens_slice': sens, 'spec_slice': spec,
            'dice_TP_mean': tp_dice_mean, 'dice_TP_median': tp_dice_med,
        })
        fold_rows.append(per_fold[-1])

    # 5-fold mean/sd
    def col(k): return [r[k] for r in per_fold if not (isinstance(r[k], float) and r[k]!=r[k])]
    ds_rollups[ds_short] = {
        'dice_TP_mean_mean': float(np.mean([r['dice_TP_mean'] for r in per_fold])),
        'dice_TP_mean_sd':   float(np.std ([r['dice_TP_mean'] for r in per_fold], ddof=1)),
        'dice_TP_med_mean':  float(np.mean([r['dice_TP_median'] for r in per_fold])),
        'dice_TP_med_sd':    float(np.std ([r['dice_TP_median'] for r in per_fold], ddof=1)),
        'sens_mean':         float(np.mean([r['sens_slice'] for r in per_fold])),
        'sens_sd':           float(np.std ([r['sens_slice'] for r in per_fold], ddof=1)),
        'spec_mean':         float(np.mean([r['spec_slice'] for r in per_fold])),
        'spec_sd':           float(np.std ([r['spec_slice'] for r in per_fold], ddof=1)),
        'total_TP': sum(r['n_TP_slices'] for r in per_fold),
        'total_TN': sum(r['n_TN_slices'] for r in per_fold),
        'total_FP': sum(r['n_FP_slices'] for r in per_fold),
        'total_FN': sum(r['n_FN_slices'] for r in per_fold),
    }

# ---------- write outputs ----------
csv_out = ROOT / 'results' / 'slice_level_metrics.csv'
with csv_out.open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(fold_rows[0].keys()))
    w.writeheader(); w.writerows(fold_rows)

lines = []
lines.append('=' * 78)
lines.append('Slice-level metrics (Liedes 2023-comparable) — from 5-fold OOF predictions')
lines.append('=' * 78)
lines.append('')
lines.append(f'Per-dataset rollup (5-fold mean ± SD):')
lines.append('')
lines.append(f'{"":7s} {"slice Dice (TP)":>22s} {"slice Dice (TP) med":>22s} {"slice sens":>14s} {"slice spec":>14s}')
for d, s in ds_rollups.items():
    lines.append(f'{d:7s} '
                 f'{s["dice_TP_mean_mean"]:.3f} ± {s["dice_TP_mean_sd"]:.3f}     '
                 f'{s["dice_TP_med_mean"]:.3f} ± {s["dice_TP_med_sd"]:.3f}     '
                 f'{s["sens_mean"]:.3f} ± {s["sens_sd"]:.3f} '
                 f'{s["spec_mean"]:.3f} ± {s["spec_sd"]:.3f}')

lines.append('')
lines.append('Per-fold detail:')
lines.append(f'{"ds":>5s} {"fold":>4s} {"TP":>5s} {"TN":>6s} {"FP":>5s} {"FN":>5s} '
             f'{"sens":>6s} {"spec":>6s} {"Dice_TP_mean":>13s} {"Dice_TP_med":>13s}')
for r in fold_rows:
    lines.append(f'{r["dataset"]:>5s} {r["fold"]:>4d} '
                 f'{r["n_TP_slices"]:>5d} {r["n_TN_slices"]:>6d} {r["n_FP_slices"]:>5d} {r["n_FN_slices"]:>5d} '
                 f'{r["sens_slice"]:>6.3f} {r["spec_slice"]:>6.3f} '
                 f'{r["dice_TP_mean"]:>13.3f} {r["dice_TP_median"]:>13.3f}')

lines.append('')
lines.append('Liedes 2023 numbers (for comparison):')
lines.append('  PET-MRI slice Dice TP-only mean = 0.84')
lines.append('  PET-only slice Dice TP-only mean = 0.79')
lines.append('  PET-MRI slice sensitivity = 0.77,  specificity = 0.68')
lines.append('  Note: their 0.84 was over 20 slices in a 66-slice test set; ours is over ALL TP slices across 5 folds.')

txt = '\n'.join(lines)
(ROOT / 'results' / 'slice_level_summary.txt').write_text(txt)
print(txt)
