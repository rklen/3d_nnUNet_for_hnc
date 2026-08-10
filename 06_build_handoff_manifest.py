#!/usr/bin/env python3
"""
06_build_handoff_manifest.py

Build the Oona-handoff manifest by pivoting per-case OOF metrics from all
three datasets into one wide-format CSV keyed by case_id + anom_folder.

One row per patient. Columns:
  case_id, anom_folder, label, mri_sequence, fold,
  d006_dice, d006_tp, d006_fp, d006_n_pred, d006_n_ref,
  d007_dice, d007_tp, d007_fp, d007_n_pred, d007_n_ref,
  d008_dice, d008_tp, d008_fp, d008_n_pred, d008_n_ref,
  ref_vox  (= max(d006_n_ref, d007_n_ref, d008_n_ref) — should be identical)
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path('/Users/jussihirvonen/Dropbox/HNMRI/PET-MRI-cancer')
percase = list(csv.DictReader((ROOT / 'results' / 'oof_per_case.csv').open()))

# index: case_id -> {dataset -> row}
by_case: dict[str, dict[str, dict]] = defaultdict(dict)
for r in percase:
    by_case[r['case_id']][r['dataset']] = r

rows_out = []
for case_id in sorted(by_case.keys()):
    per_ds = by_case[case_id]
    # base info from D006 (same for all)
    base = per_ds.get('D006') or per_ds.get('D007') or per_ds.get('D008')
    if not base:
        continue
    n_refs = [int(per_ds[d]['n_ref']) for d in ('D006','D007','D008') if d in per_ds]
    row = {
        'case_id': case_id,
        'anom_folder': base['anom_folder'],
        'label': base['label'],
        'mri_sequence': base['mri_sequence'],
        'fold': base['fold'],
        'ref_vox': max(n_refs) if n_refs else '',
    }
    for ds_short in ('d006', 'd007', 'd008'):
        ds_key = ds_short.upper().replace('006','006').replace('007','007').replace('008','008')
        # actually just upper:
        ds_key = ds_short.upper()
        r = per_ds.get(ds_key, {})
        row[f'{ds_short}_dice']   = f"{float(r['dice']):.4f}" if r else ''
        row[f'{ds_short}_tp']     = r.get('tp', '')
        row[f'{ds_short}_fp']     = r.get('fp', '')
        row[f'{ds_short}_n_pred'] = r.get('n_pred', '')
    rows_out.append(row)

# group by label, sort by anom_folder
rows_out.sort(key=lambda r: (r['label'], int(r['anom_folder'].rstrip('_anon')) if r['anom_folder'].rstrip('_anon').isdigit() else 0, r['anom_folder']))

out = ROOT / 'handoff_oona' / 'handoff_manifest.csv'
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

print(f'wrote {out}')
print(f'rows: {len(rows_out)} ({sum(1 for r in rows_out if r["label"]=="pos")} pos + {sum(1 for r in rows_out if r["label"]=="neg")} neg)')
