"""Create descriptive figures; does not select or train any candidate."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
p = ROOT / 'results'
matrix = json.loads((p / 'matrix.json').read_text())
summary = json.loads((p / 'summary.json').read_text())
random = json.loads((p / 'random_search_100x6.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].scatter([r['dev_exp'] for r in matrix], [r['test_exp'] for r in matrix], s=16, alpha=.5, color='#4777a3', label='189 parameter points')
best = summary['matrix_dev_selected']
axes[0].scatter(best['dev_exp'], best['test_exp'], s=95, marker='*', color='#b63b34', label='Development-selected')
agent = summary['historical']['agent']
axes[0].scatter(agent['dev_exp'], agent['test_exp'], s=45, marker='D', color='#207e69', label='Historical agent')
axes[0].set(xlabel='Development nDCG (exp gain) @10', ylabel='Public-test nDCG (exp gain) @10', title='Post-hoc development / test landscape')
axes[0].legend(fontsize=8)
scores = np.array([r['selected']['test_exp'] for r in random])
axes[1].hist(scores, bins=16, color='#86a9c7', edgecolor='white')
axes[1].axvline(agent['test_exp'], color='#207e69', label='One historical LLM trajectory')
axes[1].axvline(summary['historical']['preset6']['test_exp'], color='#b63b34', linestyle='--', label='Preset six-point search')
axes[1].set(xlabel='Selected candidate public-test nDCG @10', ylabel='Random-search repetitions', title='100 searches, six development calls each')
axes[1].legend(fontsize=8)
fig.tight_layout()
for suffix in ('png', 'svg'):
    fig.savefig(p / f'phase2_comparison.{suffix}', dpi=170)
plt.close(fig)
ks = summary['protocol']['matrix']['k']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
for ax, key, title in zip(axes, ['dev_exp', 'test_exp'], ['Development surface', 'Public-test surface (diagnostic only)']):
    data = np.array([row[key] for row in matrix]).reshape(len(ks), 21)
    pic = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    ax.set(yticks=range(len(ks)), yticklabels=ks, xticks=[0, 5, 10, 15, 20], xticklabels=['0', '.25', '.50', '.75', '1'], xlabel='BM25 weight', ylabel='RRF k', title=title)
    fig.colorbar(pic, ax=ax, label='nDCG (exp gain) @10')
fig.tight_layout()
for suffix in ('png', 'svg'):
    fig.savefig(p / f'phase2_surfaces.{suffix}', dpi=170)
