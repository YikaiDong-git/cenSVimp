"""Best-of-N ensemble: per SV, keep the strategy with the highest cross-validated R^2.

Ties are broken by fewer nonzero coefficients, then by a seeded random draw (seed 42; one draw
per available strategy per SV, SVs in sv_index order, strategies in the order given), which makes
the selection reproducible. Strategies must have been trained on the same SV components, so that
sv_index refers to the same SV in every strategy.

Output: one models table and one weights table holding the selected model of each SV whose
cross-validated R^2 reaches `min_cv_r2`, ready for impute.py, plus the full selection log.
"""
import random
from pathlib import Path

import pandas as pd

STRATEGIES_8 = ['edge500', 'edge1000', 'edge2000', 'fksize100k', 'fksize200k', 'fksize500k',
                'midpoint500', 'midpoint1000']
STRATEGIES_6 = [s for s in STRATEGIES_8 if not s.startswith('midpoint')]
SEED = 42


def select(metrics_by_strategy, strategies):
    """metrics_by_strategy: {strategy: DataFrame with sv_index, cv_r2, n_nonzero_coef}.

    Returns a DataFrame: sv_index, strategy, cv_r2, n_nonzero_coef, n_strategies, cv_r2_<s>...
    """
    table = {}
    for s in strategies:
        if s not in metrics_by_strategy:
            continue
        m = metrics_by_strategy[s]
        cv = pd.to_numeric(m['cv_r2'], errors='coerce')
        nnz = pd.to_numeric(m['n_nonzero_coef'], errors='coerce')
        for si, c, z in zip(pd.to_numeric(m['sv_index']), cv, nnz):
            if pd.isna(si):
                continue
            table.setdefault(int(si), {})[s] = (0.0 if pd.isna(c) else float(c),
                                                999 if pd.isna(z) else int(z))
    rng = random.Random(SEED)
    rows = []
    for si in sorted(table):
        cands = sorted(((c, -z, rng.random(), s) for s, (c, z) in table[si].items()), reverse=True)
        c, neg_z, _, s = cands[0]
        row = {'sv_index': si, 'strategy': s, 'cv_r2': c, 'n_nonzero_coef': -neg_z,
               'n_strategies': len(table[si])}
        row.update({f'cv_r2_{t}': table[si][t][0] if t in table[si] else '' for t in strategies})
        rows.append(row)
    return pd.DataFrame(rows)


def build_ensemble(base_dir, chrom, strategies, outdir, min_cv_r2=0.1):
    """Read <base_dir>/<strategy>/<chrom>/{sv_metrics,models,weights}, write the ensemble."""
    base = Path(base_dir)
    metrics = {s: pd.read_csv(base / s / chrom / 'sv_metrics.tsv', sep='\t')
               for s in strategies if (base / s / chrom / 'sv_metrics.tsv').exists()}
    sel = select(metrics, strategies)
    out = Path(outdir) / chrom
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / 'selection.tsv', sep='\t', index=False)

    keep = sel[sel['cv_r2'] >= min_cv_r2]
    models, weights = [], []
    for s, grp in keep.groupby('strategy', sort=False):
        idx = set(grp['sv_index'])
        m = pd.read_csv(base / s / chrom / 'models.tsv', sep='\t')
        w = pd.read_csv(base / s / chrom / 'weights.tsv.gz', sep='\t')
        models.append(m[m['sv_index'].isin(idx)].assign(strategy=s))
        weights.append(w[w['sv_index'].isin(idx)])
    m = pd.concat(models).sort_values('sv_index') if models else pd.DataFrame()
    w = pd.concat(weights).sort_values('sv_index', kind='stable') if weights else pd.DataFrame()
    m.to_csv(out / 'models.tsv', sep='\t', index=False)
    w.to_csv(out / 'weights.tsv.gz', sep='\t', index=False, float_format='%.17g')
    return out
