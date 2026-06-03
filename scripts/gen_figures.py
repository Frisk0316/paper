#!/usr/bin/env python3
"""
gen_figures.py — Generate publication-quality PDF figures for the paper.

Reads saved results from the v6 checkpoints and produces:
  1. Per-model cumulative return plots (PDF, 300 dpi)
  2. Combined cumulative return plot (best config per model)
  3. Decile Sharpe bar chart for CS-Gated +Onchain
  4. TFT feature importance

Usage:
    python paper/scripts/gen_figures.py
"""

import sys
import os
import json
import numpy as np

PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPO_DIR = os.path.abspath(os.path.join(PAPER_DIR, '..'))
V6_DIR = os.path.join(REPO_DIR, 'machine_learning_for_crypto_v6')
SUBMISSION_DIR = os.path.join(REPO_DIR, 'R12723071')

# Add v6 project to path so we can import evaluate helpers.
sys.path.insert(0, V6_DIR)

from evaluate import load_results, compute_portfolio_metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Publication rcParams ──────────────────────────────────────────────
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

FIG_DIR = os.path.join(PAPER_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

CONFIG_CANDIDATES = [
    os.environ.get('PAPER_CONFIG'),
    os.path.join(V6_DIR, 'config.json'),
    os.path.join(SUBMISSION_DIR, 'Codes', 'config.json'),
]
CONFIG_PATH = next((p for p in CONFIG_CANDIDATES if p and os.path.exists(p)), None)
if CONFIG_PATH is None:
    raise FileNotFoundError('Could not find config.json for figure generation.')

with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = json.load(f)

FEAT_CONFIGS = list(cfg['feature_configs'].keys())
FEAT_NAMES = cfg.get('_feature_names', [])
def _resolve_checkpoint_dir():
    candidates = [
        os.environ.get('PAPER_CHECKPOINT_DIR'),
        os.path.join(V6_DIR, cfg.get('output_dir', 'checkpoints')),
        os.path.join(V6_DIR, 'checkpoints'),
        os.path.join(SUBMISSION_DIR, 'Codes', 'checkpoints'),
    ]
    required = ['cs_gated_results.npz', 'lstm_results.npz', 'tft_results.npz',
                'random_forest_results.npz']
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if all(os.path.exists(os.path.join(candidate, name)) for name in required):
            return candidate
    raise FileNotFoundError('Could not find saved paper checkpoints.')


CKPT_DIR = _resolve_checkpoint_dir()

# Best config for each model (from cross_model_comparison.csv)
BEST_CONFIG = {
    'cs_gated': '+Onchain',
    'lstm':     '+Trump+ETF',
    'tft':      '+Trump+ETF',
    'random_forest': 'All',
}

MODELS = ['cs_gated', 'lstm', 'tft', 'random_forest']
MODEL_LABELS = {
    'cs_gated': 'CS-Gated',
    'lstm': 'LSTM',
    'tft': 'TFT',
    'random_forest': 'RF',
}
MODEL_COLORS = {
    'cs_gated': '#2166ac',
    'lstm': '#b2182b',
    'tft': '#1b7837',
    'random_forest': '#7f3b08',
}


def load_model(model_type):
    """Load results and compute metrics for all feature configs."""
    npz_path = os.path.join(CKPT_DIR, f'{model_type}_results.npz')
    if not os.path.exists(npz_path):
        print(f"  [SKIP] {npz_path} not found")
        return None, None, None
    results, dates, assets, test_times = load_results(npz_path, FEAT_CONFIGS)
    all_metrics = {}
    for name, res in results.items():
        m = compute_portfolio_metrics(
            res['ensemble_preds'], res['ensemble_tgts'],
            res['ensemble_msks'])
        all_metrics[name] = m
    return all_metrics, dates, test_times


def fig1_per_model_cumulative():
    """Per-model cumulative return plots."""
    for model in MODELS:
        all_metrics, dates, test_times = load_model(model)
        if all_metrics is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        for name, m in all_metrics.items():
            cum_pw = np.cumsum(m['weekly_pw'])
            cum_ew = np.cumsum(m['weekly_ew'])
            axes[0].plot(range(len(cum_pw)), cum_pw,
                         label=f"{name} ({m['sr_pw']:+.2f})", lw=1)
            axes[1].plot(range(len(cum_ew)), cum_ew,
                         label=f"{name} ({m['sr_ew']:+.2f})", lw=1)

        for ax, title in zip(axes, ['Price-Weighted', 'Equal-Weighted']):
            ax.set_title(f'{MODEL_LABELS[model]} L/S: {title}')
            ax.set_xlabel('Test Week')
            ax.set_ylabel('Cumulative Return')
            ax.axhline(0, color='grey', ls='--', lw=0.5)
            ax.legend(fontsize=6, ncol=2)

        plt.tight_layout()
        path = os.path.join(FIG_DIR, f'cum_returns_{model}.pdf')
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


def fig2_combined_cumulative():
    """Combined cumulative returns: four best strategy lines per panel."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for model in MODELS:
        all_metrics, dates, test_times = load_model(model)
        if all_metrics is None:
            continue
        best = BEST_CONFIG[model]
        m = all_metrics[best]
        label = f'{MODEL_LABELS[model]} ({best})'
        color = MODEL_COLORS[model]

        cum_pw = np.cumsum(m['weekly_pw'])
        cum_ew = np.cumsum(m['weekly_ew'])
        axes[0].plot(range(len(cum_pw)), cum_pw,
                     label=f"{label} SR={m['sr_pw']:.2f}",
                     color=color, lw=1.2)
        axes[1].plot(range(len(cum_ew)), cum_ew,
                     label=f"{label} SR={m['sr_ew']:.2f}",
                     color=color, lw=1.2)

    # Add zero line
    for ax, title in zip(axes, ['Price-Weighted', 'Equal-Weighted']):
        ax.set_title(f'Best Model Comparison: {title}')
        ax.set_xlabel('Test Week')
        ax.set_ylabel('Cumulative L/S Return')
        ax.axhline(0, color='grey', ls='--', lw=0.5)
        ax.legend(fontsize=6)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'cum_returns_combined.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig3_decile_sharpe():
    """Decile Sharpe bar chart for CS-Gated +Onchain.

    compute_portfolio_metrics stores decile 0 as the highest predicted-rank
    bucket and decile 9 as the lowest. The paper reports the conventional
    D1-to-D10 order, where D1 is the short leg and D10 is the long leg.
    """
    all_metrics, _, _ = load_model('cs_gated')
    if all_metrics is None:
        return

    best = BEST_CONFIG['cs_gated']
    m = all_metrics[best]

    fig, ax = plt.subplots(figsize=(4, 3))
    labels = [f'D{d}' for d in range(1, 11)]
    x = np.arange(10)
    w = 0.35

    sr_pw = [m['decile_sr_pw'][d] for d in range(9, -1, -1)]
    sr_ew = [m['decile_sr_ew'][d] for d in range(9, -1, -1)]

    ax.bar(x - w/2, sr_pw, w, label='PW', color='steelblue')
    ax.bar(x + w/2, sr_ew, w, label='EW', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.axhline(0, color='grey', ls='--', lw=0.5)
    ax.set_ylabel('Annualised Sharpe')
    ax.set_title(f'CS-Gated ({best}) Decile Sharpe Ratios')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'decile_sharpe_cs_gated.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig4_feat_importance():
    """TFT feature importance from VSN weights."""
    npz_path = os.path.join(CKPT_DIR, 'tft_results.npz')
    if not os.path.exists(npz_path):
        print("  [SKIP] tft_results.npz not found")
        return

    results, _, _, _ = load_results(npz_path, FEAT_CONFIGS)
    best = BEST_CONFIG['tft']
    fi = results[best].get('avg_feat_importance')
    if fi is None:
        print("  [SKIP] No feature importance in TFT results")
        return

    config_feats = cfg['feature_configs'][best]
    names = [FEAT_NAMES[j] if j < len(FEAT_NAMES) else f'f{j}'
             for j in config_feats]

    order = np.argsort(fi)[::-1]
    fi_sorted = fi[order]
    names_sorted = [names[j] for j in order]

    fig, ax = plt.subplots(figsize=(4, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(fi_sorted)))
    ax.barh(range(len(fi_sorted)), fi_sorted, color=colors)
    ax.set_yticks(range(len(fi_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel('VSN Weight')
    ax.set_title(f'TFT Variable Importance ({best})')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'feat_importance_tft.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


if __name__ == '__main__':
    print("=== Generating publication figures ===")
    fig1_per_model_cumulative()
    fig2_combined_cumulative()
    fig3_decile_sharpe()
    fig4_feat_importance()
    print("=== Done ===")
