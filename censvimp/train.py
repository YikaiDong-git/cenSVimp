"""Per-SV elastic-net models: pericentromeric SNP genotypes -> SV genotype (0/1/2).

Inputs: a SNP panel (panel.py) and SV component genotypes (cenSVhap output: <chr>_sv_genotypes.npz,
<chr>_sv_variants.tsv, <chr>_sv_samples.txt). Samples are matched by ID.

Per SV: SNPs standardized (StandardScaler); scikit-learn ElasticNetCV over l1_ratio {0.3, 0.5, 1.0}
and 50 alphas per l1_ratio, K shuffled folds (random_state 42), max_iter 2000. Cross-validated
R^2 = 1 - min(mean CV MSE) / Var(y).

Outputs per chromosome:
  sv_metrics.tsv   one row per SV (sv_index = row among the frequency-filtered SVs)
  models.tsv       one row per SV: identity, cv_r2, intercept, number of SNPs used
  weights.tsv.gz   one row per (SV, SNP with nonzero coefficient): SNP key, scaler mean and
                   scale, coefficient. Dosage = clip(intercept + sum coef * (x - mean) / scale, 0, 2).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

L1_RATIOS = [0.3, 0.5, 1.0]
N_ALPHAS = 50
SEED = 42
SV_ID_COLUMNS = ['id', 'chrom', 'pos', 'end', 'svtype', 'svlen', 'sv_category', 'original_sv_id',
                 'mapped_haplogroup']


def load_matrix(npz_path, samples_path):
    gt = np.load(npz_path)['genotypes']
    samples = [l.strip() for l in Path(samples_path).read_text().splitlines() if l.strip()]
    return gt, samples


def align(x, x_samples, y, y_samples):
    shared = sorted(set(x_samples) & set(y_samples))
    xi = {s: i for i, s in enumerate(x_samples)}
    yi = {s: i for i, s in enumerate(y_samples)}
    return x[:, [xi[s] for s in shared]], y[:, [yi[s] for s in shared]], shared


def carrier_filter(mat, rows, lo, hi):
    """Keep rows whose carrier frequency (fraction of samples with genotype 1 or 2) is in [lo, hi]."""
    f = np.mean((mat == 1) | (mat == 2), axis=1)
    keep = (f >= lo) & (f <= hi)
    return mat[keep], rows[keep].reset_index(drop=True)


def fit_one(X, y, n_folds, n_jobs):
    cv = LeaveOneOut() if n_folds == 0 else KFold(n_splits=min(n_folds, len(y)), shuffle=True,
                                                  random_state=SEED)
    model = ElasticNetCV(l1_ratio=L1_RATIOS, cv=cv, alphas=N_ALPHAS, max_iter=2000, n_jobs=n_jobs,
                         random_state=SEED)
    model.fit(X, y)
    return model


def metrics_row(model, X, y, sv, snps):
    y_p = np.clip(model.predict(X), 0, 2)
    mse = mean_squared_error(y, y_p)
    p_r, p_p = pearsonr(y, y_p) if np.std(y_p) > 1e-9 and np.std(y) > 1e-9 else (0.0, 1.0)
    cv_mse = np.min(np.mean(model.mse_path_, axis=-1))
    coef = model.coef_
    nz = np.where(coef != 0)[0]
    order = np.argsort(np.abs(coef))[::-1]
    pos = snps.iloc[nz]['pos'].tolist()
    return {
        'r2': r2_score(y, y_p),
        'cv_r2': 1.0 - cv_mse / np.var(y) if np.var(y) > 1e-9 else 0.0,
        'max_coef': np.max(np.abs(coef)) if np.any(coef) else 0.0,
        'max_5_coef': coef[order[:5]].tolist() if len(coef) else [],
        'n_nonzero_coef': len(nz),
        'gt_counts': "/".join(f"{int(i)}:{int(c)}" for i, c in zip(*np.unique(y, return_counts=True))),
        'l1_ratio': model.l1_ratio_,
        'svfreq': np.sum(y) / (2 * len(y)),
        'svpos': sv['pos'], 'svend': sv['end'], 'svtype': sv['svtype'], 'svlen': sv['svlen'],
        'alpha': model.alpha_,
        'snp_pos': pos,
        'mse': mse, 'rmse': np.sqrt(mse), 'cv_mse': cv_mse, 'cv_rmse': np.sqrt(cv_mse),
        'pearson_r': p_r, 'pearson_p': p_p,
        'mae': mean_absolute_error(y, y_p),
        'selected_sv_snp_dist': [abs(p - sv['pos']) for p in pos],
        'selected_snp_coefs': coef[nz].tolist(),
        'selected_snp_pos': pos,
        'selected_snp_id': snps.iloc[nz]['id'].tolist(),
        'sv_id': sv['id'],
        'sv_category': sv.get('sv_category', 'standard'),
        'original_sv_id': sv.get('original_sv_id', ''),
        'mapped_haplogroup': sv.get('mapped_haplogroup', ''),
        'r2_score': sv.get('r2_score', None),
    }


def train_chromosome(snp_dir, sv_dir, chrom, outdir, n_folds=4, n_jobs=1, min_snp_freq=0.05,
                     max_snp_freq=0.99, min_sv_freq=0.05, max_sv_freq=0.99, sv_start=None,
                     sv_end=None, log=print):
    snp_dir, sv_dir = Path(snp_dir), Path(sv_dir)
    X_raw, x_samples = load_matrix(snp_dir / f"{chrom}_snp_genotypes.npz",
                                   snp_dir / f"{chrom}_snp_samples.txt")
    snps = pd.read_csv(snp_dir / f"{chrom}_snp_variants.tsv", sep='\t')
    Y_raw, y_samples = load_matrix(sv_dir / f"{chrom}_sv_genotypes.npz",
                                   sv_dir / f"{chrom}_sv_samples.txt")
    svs = pd.read_csv(sv_dir / f"{chrom}_sv_variants.tsv", sep='\t')

    Xa, Ya, shared = align(X_raw, x_samples, Y_raw, y_samples)
    Xa, snps = carrier_filter(Xa, snps, min_snp_freq, max_snp_freq)
    Ya, svs = carrier_filter(Ya, svs, min_sv_freq, max_sv_freq)
    scaler = StandardScaler()
    X = scaler.fit_transform(Xa.T.astype(np.float64))

    n = Ya.shape[0]
    lo = max(0, sv_start or 0)
    hi = min(n, n if sv_end is None else sv_end)
    log(f"{chrom}: {len(shared)} samples, {X.shape[1]} SNPs, training SVs [{lo}, {hi}) of {n}")

    metrics, models, weights = [], [], []
    for k in range(lo, hi):
        sv = svs.iloc[k]
        y = Ya[k].astype(np.float64)
        model = fit_one(X, y, n_folds, n_jobs)
        row = metrics_row(model, X, y, sv, snps)
        row.update(sv_index=k, chrom=chrom)
        metrics.append(row)
        nz = np.where(model.coef_ != 0)[0]
        models.append({'sv_index': k, **{c: sv[c] for c in SV_ID_COLUMNS if c in sv.index},
                       'cv_r2': row['cv_r2'], 'intercept': float(model.intercept_),
                       'n_snps': len(nz)})
        for j in nz:
            s = snps.iloc[j]
            weights.append({'sv_index': k, 'snp_chrom': s['chrom'], 'snp_pos': int(s['pos']),
                            'snp_ref': s['ref'], 'snp_alt': s['alt'],
                            'mean': float(scaler.mean_[j]), 'scale': float(scaler.scale_[j]),
                            'coef': float(model.coef_[j])})

    out = Path(outdir) / chrom
    out.mkdir(parents=True, exist_ok=True)
    tag = '' if (lo, hi) == (0, n) else f".{lo}-{hi}"
    pd.DataFrame(metrics).to_csv(out / f"sv_metrics{tag}.tsv", sep='\t', index=False)
    pd.DataFrame(models).to_csv(out / f"models{tag}.tsv", sep='\t', index=False)
    pd.DataFrame(weights, columns=['sv_index', 'snp_chrom', 'snp_pos', 'snp_ref', 'snp_alt', 'mean',
                                   'scale', 'coef']).to_csv(out / f"weights{tag}.tsv.gz", sep='\t',
                                                            index=False, float_format='%.17g')
    return out


def merge_chunks(chrom_dir):
    """Concatenate chunked outputs (sv_metrics.<a>-<b>.tsv etc.) in sv_index order."""
    d = Path(chrom_dir)
    for stem, ext in (('sv_metrics', '.tsv'), ('models', '.tsv'), ('weights', '.tsv.gz')):
        parts = sorted(d.glob(f"{stem}.*-*{ext}"), key=lambda p: int(p.name.split('.')[1].split('-')[0]))
        if parts:
            df = pd.concat([pd.read_csv(p, sep='\t') for p in parts], ignore_index=True)
            kw = {'float_format': '%.17g'} if stem == 'weights' else {}
            df.to_csv(d / f"{stem}{ext}", sep='\t', index=False, **kw)
