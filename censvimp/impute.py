"""Impute SV dosages into SNP-genotyped samples from a models + weights table.

For each model: dosage = clip(intercept + sum_k coef_k * (x_k - mean_k) / scale_k, 0, 2), where x_k
is the target sample's alternate-allele count at SNP k. SNPs are matched on chromosome, position,
reference and alternate allele; SNPs absent from the target VCF, allele-mismatched, or with a
missing genotype take x = 0 (homozygous reference), as in the published analysis.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import cyvcf2

from .vcf import fetch


def target_genotypes(vcf_path, chrom, keys, samples):
    """Alternate-allele counts, samples x len(keys); unmatched keys stay 0."""
    col = {k: i for i, k in enumerate(keys)}
    reader = cyvcf2.VCF(str(vcf_path), gts012=True, samples=list(samples))
    names = list(reader.samples)
    order = [names.index(s) for s in samples if s in names]
    found = [s for s in samples if s in names]
    X = np.zeros((len(found), len(keys)), dtype=np.float64)
    pos = [k[1] for k in keys]
    matched = set()
    for v in fetch(reader, vcf_path, f"{chrom}:{max(0, min(pos) - 1000)}-{max(pos) + 1000}"):
        if len(v.ALT) != 1 or len(v.REF) != 1 or len(v.ALT[0]) != 1:
            continue
        j = col.get((v.CHROM, v.POS, v.REF, v.ALT[0]))
        if j is None:
            continue
        g = v.gt_types[order].astype(np.float64)
        g[g == 3] = 0
        X[:, j] = g
        matched.add(j)
    reader.close()
    return X, found, len(matched)


def impute_chromosome(models_tsv, weights_tsv, vcf_path, chrom, samples, outdir, min_cv_r2=0.1):
    models = pd.read_csv(models_tsv, sep='\t')
    weights = pd.read_csv(weights_tsv, sep='\t')
    models = models[(models['chrom'] == chrom) & (models['cv_r2'] >= min_cv_r2)]
    weights = weights[weights['sv_index'].isin(models['sv_index'])]
    keys = sorted(set(zip(weights['snp_chrom'], weights['snp_pos'], weights['snp_ref'],
                          weights['snp_alt'])), key=lambda k: k[1])
    X, found, n_matched = target_genotypes(vcf_path, chrom, keys, samples)
    col = {k: i for i, k in enumerate(keys)}

    dos = {}
    for si, w in weights.groupby('sv_index', sort=False):
        j = [col[k] for k in zip(w['snp_chrom'], w['snp_pos'], w['snp_ref'], w['snp_alt'])]
        z = (X[:, j] - w['mean'].values) / w['scale'].values
        dos[si] = z @ w['coef'].values
    rows = []
    for _, m in models.iterrows():
        d = np.clip(dos.get(m['sv_index'], np.zeros(len(found))) + m['intercept'], 0, 2)
        rows.append({'sv_index': m['sv_index'], 'id': m['id'],
                     **{s: round(float(x), 4) for s, x in zip(found, d)}})

    out = Path(outdir) / chrom
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{chrom}_imputed_dosages.tsv", sep='\t', index=False)
    (out / f"{chrom}_imputed_samples.txt").write_text('\n'.join(found) + '\n')
    (out / f"{chrom}_snp_matching.tsv").write_text(
        f"metric\tvalue\nmodel_snps\t{len(keys)}\nmatched_snps\t{n_matched}\n"
        f"n_target_samples\t{len(found)}\nn_svs\t{len(rows)}\n")
    return out
