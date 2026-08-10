#!/usr/bin/env python3
"""
05_analyze_oof.py — read all 15 fold validation summaries, join with patient
manifest (case_id -> pos/neg + MRI sequence), and compute:

  per-fold:
    Dice_pos_mean   = mean Dice on positives only         (Liedes-comparable)
    Dice_pos_med    = median Dice on positives only
    Dice_all_mean   = mean Dice over all cases incl. negatives
    detect_rate     = fraction of positives with Dice > 0
    fp_rate         = fraction of negatives with any prediction (>0 voxels)
  per-dataset:
    5-fold mean ± SD for each of the above
  per-case:
    one tidy CSV row per (dataset, case_id) with all metrics + label

Inputs:
    results/summaries/d{6,7,8}_fold{0..4}.json   (one per fold)
    patient_manifest.csv                          (case_id, anom_folder, label, mri_sequence, status)

Outputs:
    results/oof_per_case.csv
    results/oof_per_fold.csv
    results/oof_summary.txt   (human-readable)
"""
from __future__ import annotations

import csv
import json
import numpy as np

class _S:
    @staticmethod
    def mean(xs): return float(np.nanmean(xs)) if xs else float('nan')
    @staticmethod
    def median(xs): return float(np.nanmedian(xs)) if xs else float('nan')
    @staticmethod
    def stdev(xs):
        a = np.asarray(xs, dtype=float)
        a = a[~np.isnan(a)]
        return float(a.std(ddof=1)) if a.size > 1 else float('nan')

st = _S
from pathlib import Path

ROOT = Path('/Users/jussihirvonen/Dropbox/HNMRI/PET-MRI-cancer')
SUMMARIES = ROOT / 'results' / 'summaries'
OUT = ROOT / 'results'

# ----- load manifest -----
manifest = {}  # case_id -> {label, anom_folder, mri_sequence}
with (ROOT / 'patient_manifest.csv').open() as fh:
    for row in csv.DictReader(fh):
        manifest[row['case_id']] = row

# ----- per-case rows -----
per_case_rows = []
per_fold_rows = []

for d in (6, 7, 8):
    fold_means_pos = []
    fold_means_all = []
    fold_medians_pos = []
    fold_detect_rates = []
    fold_fp_rates = []
    for f in (0, 1, 2, 3, 4):
        with (SUMMARIES / f'd{d}_fold{f}.json').open() as fh:
            data = json.load(fh)
        cases = data['metric_per_case']

        pos_dices = []
        neg_dices = []
        n_pos_detected = 0
        n_neg_fp = 0
        all_dices = []

        for entry in cases:
            case_id = Path(entry['prediction_file']).name.split('.')[0]  # case_XXXX from case_XXXX.nii.gz
            metrics = entry['metrics']['1']
            dice = float(metrics['Dice'])
            tp = int(metrics['TP'])
            fp = int(metrics['FP'])
            n_pred = int(metrics['n_pred'])
            n_ref = int(metrics['n_ref'])

            m = manifest.get(case_id, {})
            label = m.get('label', '?')
            seq = m.get('mri_sequence', '')

            per_case_rows.append({
                'dataset': f'D{d:03d}', 'fold': f, 'case_id': case_id,
                'anom_folder': m.get('anom_folder', ''),
                'label': label, 'mri_sequence': seq,
                'dice': dice, 'tp': tp, 'fp': fp, 'n_pred': n_pred, 'n_ref': n_ref,
            })

            all_dices.append(dice)
            if label == 'pos':
                pos_dices.append(dice)
                if dice > 0:
                    n_pos_detected += 1
            elif label == 'neg':
                neg_dices.append(dice)
                if n_pred > 0:
                    n_neg_fp += 1

        n_pos = len(pos_dices)
        n_neg = len(neg_dices)
        mean_pos = st.mean(pos_dices) if pos_dices else float('nan')
        med_pos = st.median(pos_dices) if pos_dices else float('nan')
        mean_all = st.mean(all_dices) if all_dices else float('nan')
        det = n_pos_detected / n_pos if n_pos else float('nan')
        fpr = n_neg_fp / n_neg if n_neg else float('nan')

        fold_means_pos.append(mean_pos)
        fold_medians_pos.append(med_pos)
        fold_means_all.append(mean_all)
        fold_detect_rates.append(det)
        fold_fp_rates.append(fpr)

        per_fold_rows.append({
            'dataset': f'D{d:03d}', 'fold': f,
            'n_pos': n_pos, 'n_neg': n_neg,
            'dice_pos_mean': mean_pos, 'dice_pos_median': med_pos,
            'dice_all_mean': mean_all,
            'detect_rate': det, 'fp_rate': fpr,
        })

    # 5-fold summary row
    per_fold_rows.append({
        'dataset': f'D{d:03d}', 'fold': 'mean',
        'n_pos': sum(r['n_pos'] for r in per_fold_rows[-5:]),
        'n_neg': sum(r['n_neg'] for r in per_fold_rows[-5:]),
        'dice_pos_mean':   st.mean(fold_means_pos),
        'dice_pos_median': st.mean(fold_medians_pos),
        'dice_all_mean':   st.mean(fold_means_all),
        'detect_rate':     st.mean(fold_detect_rates),
        'fp_rate':         st.mean(fold_fp_rates),
    })
    per_fold_rows.append({
        'dataset': f'D{d:03d}', 'fold': 'sd',
        'n_pos': '', 'n_neg': '',
        'dice_pos_mean':   st.stdev(fold_means_pos),
        'dice_pos_median': st.stdev(fold_medians_pos),
        'dice_all_mean':   st.stdev(fold_means_all),
        'detect_rate':     st.stdev(fold_detect_rates),
        'fp_rate':         st.stdev(fold_fp_rates),
    })

# ----- write outputs -----
with (OUT / 'oof_per_case.csv').open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(per_case_rows[0].keys()))
    w.writeheader()
    w.writerows(per_case_rows)

with (OUT / 'oof_per_fold.csv').open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(per_fold_rows[0].keys()))
    w.writeheader()
    w.writerows(per_fold_rows)

# Human-readable summary
lines = []
lines.append('=' * 78)
lines.append('PET-MRI HN-Cancer 5-fold CV — OOF results')
lines.append('=' * 78)
lines.append('')
lines.append('Datasets:')
lines.append('  D006 = PET only (1 channel)')
lines.append('  D007 = PET + MRI heterogeneous (2 channels)')
lines.append('  D008 = PET + T1_MRI + T2_MRI sequence-conditioned (3 channels)')
lines.append('')
lines.append(f'{"":7s} {"fold":>4s} {"n_pos":>5s} {"n_neg":>5s} '
             f'{"Dice_pos_mean":>14s} {"Dice_pos_med":>13s} '
             f'{"Dice_all_mean":>14s} {"detect":>7s} {"FP_rate":>8s}')
for row in per_fold_rows:
    label = f"{row['dataset']}" if row['fold'] not in ('mean','sd') else f"{row['dataset']}"
    fold = row['fold']
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and not (v != v) else str(v)
    lines.append(
        f'{label:7s} {str(fold):>4s} {str(row["n_pos"]):>5s} {str(row["n_neg"]):>5s} '
        f'{fmt(row["dice_pos_mean"]):>14s} {fmt(row["dice_pos_median"]):>13s} '
        f'{fmt(row["dice_all_mean"]):>14s} {fmt(row["detect_rate"]):>7s} '
        f'{fmt(row["fp_rate"]):>8s}'
    )
    if fold == 'sd':
        lines.append('')

lines.append('Comparison to Liedes 2023:')
lines.append('  Their PET-MRI median Dice on whole test set = 0.81')
lines.append('  Their PET-MRI mean Dice on TP-only slices    = 0.84')
lines.append('  Their PET-only mean Dice on TP-only slices    = 0.79')
lines.append('')

(OUT / 'oof_summary.txt').write_text('\n'.join(lines))
print('\n'.join(lines))
