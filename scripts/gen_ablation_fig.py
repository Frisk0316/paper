#!/usr/bin/env python3
"""
gen_ablation_fig.py — Generate ablation study bar chart (PDF).

Reads:
    machine_learning_for_crypto_v6/outputs/ablation_study.csv

Produces:
    paper/figures/ablation_bar.pdf

Usage:
    python paper/scripts/gen_ablation_fig.py
"""

import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

V6_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '../../machine_learning_for_crypto_v6'))
PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.path.join(PAPER_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH = os.path.join(V6_DIR, 'outputs', 'ablation_study.csv')


def main():
    if not os.path.exists(CSV_PATH):
        print(f"  [SKIP] {CSV_PATH} not found — run ablation_study.py first")
        return

    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("  [SKIP] ablation_study.csv is empty")
        return

    # Group by feature config
    configs = []
    seen = set()
    for r in rows:
        fc = r.get('feature_config', r.get('Feature Config', ''))
        if fc not in seen:
            configs.append(fc)
            seen.add(fc)

    variants = []
    seen_v = set()
    for r in rows:
        v = r.get('variant', r.get('Variant', ''))
        if v not in seen_v:
            variants.append(v)
            seen_v.add(v)

    # Build data matrix
    data = {}
    for r in rows:
        fc = r.get('feature_config', r.get('Feature Config', ''))
        v = r.get('variant', r.get('Variant', ''))
        sr = float(r.get('sr_ew', r.get('SR_EW', 0)))
        data[(v, fc)] = sr

    n_variants = len(variants)
    n_configs = len(configs)
    x = np.arange(n_configs)
    width = 0.8 / max(n_variants, 1)
    colors = ['#bdbdbd', '#6baed6', '#2166ac'][:n_variants]

    fig, ax = plt.subplots(figsize=(5, 3))
    for i, v in enumerate(variants):
        srs = [data.get((v, c), 0) for c in configs]
        offset = (i - n_variants / 2 + 0.5) * width
        ax.bar(x + offset, srs, width, label=v, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20)
    ax.set_ylabel('Annualised Sharpe (EW)')
    ax.set_title('Ablation Study: Stepwise v6 Improvements')
    ax.legend()
    ax.axhline(0, color='grey', ls='--', lw=0.5)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'ablation_bar.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


if __name__ == '__main__':
    print("=== Generating ablation bar chart ===")
    main()
    print("=== Done ===")
