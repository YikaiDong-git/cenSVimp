"""Pericentromeric predictor-SNP panels.

Three selection strategies around each centromere:
  fixed:<bp>     SNPs within <bp> outward of each centromere edge (edges extended past adjacent
                 heterochromatin, see `extended_edges`)
  edge:<n>       the <n> SNPs nearest each centromere edge, outward, searched up to 20 Mb
  midpoint:<n>   the <n> SNPs nearest the centromere midpoint on each side, searched up to 20 Mb

Retained SNPs are biallelic, FILTER=PASS, minor allele frequency >= 0.05 and < 5% missing genotypes
among the panel samples. Output per chromosome (the format train.py reads):
  <chr>_snp_genotypes.npz   genotypes: SNPs x samples, 0/1/2 (3 = missing)
  <chr>_snp_variants.tsv    chrom, pos, ref, alt, id, ac, an, af, maf, n_miss
  <chr>_snp_samples.txt     sample IDs
"""
from pathlib import Path

import numpy as np
import pandas as pd
import cyvcf2

from .vcf import fetch

SEARCH = 20_000_000


def parse_strategy(text):
    """'fixed:100000' / 'edge:1000' / 'midpoint:500' -> (kind, value)."""
    kind, _, value = text.partition(':')
    if kind not in ('fixed', 'edge', 'midpoint') or not value.isdigit():
        raise ValueError(f"strategy must be fixed:<bp>, edge:<n> or midpoint:<n>, got {text!r}")
    return kind, int(value)


def read_bed(path, names):
    return pd.read_csv(path, sep='\t', header=None, names=names)


def extended_edges(cen_start, cen_end, het, proximity=1_000_000):
    """Centromere edges extended outward over heterochromatin blocks within `proximity`.

    `het` holds the blocks of one chromosome (columns start, end, type). Short-arm blocks of the
    acrocentric chromosomes are ignored; blocks inside the centromere change nothing; a block
    overlapping a boundary extends both edges to cover it.
    """
    left, right = cen_start, cen_end
    for _, h in het.iterrows():
        hs, he = h['start'], h['end']
        if h['type'] == 'short_arm' or (hs >= cen_start and he <= cen_end):
            continue
        if he <= cen_start:
            if cen_start - he <= proximity:
                left = min(left, hs)
        elif hs >= cen_end:
            if hs - cen_end <= proximity:
                right = max(right, he)
        else:
            left, right = min(left, hs), max(right, he)
    return left, right


def fixed_windows(centromeres, het, size, unextended=(), proximity=1_000_000):
    """BED rows (chrom, start, end): the left and right windows of `size` bp per chromosome.

    Chromosomes listed in `unextended` use their centromere edges as given.
    """
    rows = []
    for _, c in centromeres.iterrows():
        blocks = het.iloc[0:0] if c['chrom'] in unextended else het[het['chrom'] == c['chrom']]
        left, right = extended_edges(c['start'], c['end'], blocks, proximity)
        rows += [(c['chrom'], max(0, left - size), left), (c['chrom'], right, right + size)]
    return rows


def read_snps(vcf_path, regions, samples, min_maf=0.05, max_missing=0.05, count_haploid_once=True):
    """Genotypes (SNPs x samples) and variant rows for biallelic PASS SNPs in `regions`.

    regions: 1-based inclusive region strings, e.g. 'chr9:42100001-42200000'.
    The MAF filter counts called alleles (haploid calls once) unless count_haploid_once=False,
    which counts a haploid call as homozygous. The stored genotypes and the ac/an/af/maf columns
    always use the 0/1/2 coding, a haploid alternate call being 2. The two counts differ only on
    haploid calls, i.e. chrX in males.
    """
    reader = cyvcf2.VCF(str(vcf_path), gts012=True, samples=list(samples))
    names = list(reader.samples)
    n = len(names)
    gts, rows = [], []
    for region in regions:
        for v in fetch(reader, vcf_path, region):
            if v.FILTER is not None or len(v.ALT) != 1 or len(v.REF) != 1 or len(v.ALT[0]) != 1:
                continue
            gt = v.gt_types.astype(np.int8)
            known = gt != 3
            n_known = int(known.sum())
            if n_known == 0:
                continue
            ac = int(gt[known].sum())
            an = 2 * n_known
            af = ac / an
            maf = min(af, 1.0 - af)
            if count_haploid_once:
                # MAF filter on called alleles: a haploid call (chrX in males) is one allele
                al = v.genotype.array()[:, :-1]
                f = (al == 1).sum() / (al >= 0).sum()
                filter_maf = min(f, 1.0 - f)
            else:
                filter_maf = maf
            if filter_maf < min_maf or 1.0 - n_known / n > max_missing:
                continue
            gts.append(gt)
            rows.append({'chrom': v.CHROM, 'pos': v.POS, 'ref': v.REF, 'alt': v.ALT[0],
                         'id': v.ID or f"{v.CHROM}_{v.POS}_{v.REF}_{v.ALT[0]}",
                         'ac': ac, 'an': an, 'af': round(af, 6), 'maf': round(maf, 6),
                         'n_miss': n - n_known})
    reader.close()
    return (np.vstack(gts) if gts else None), rows, names


def nearest(positions, anchor_left, anchor_right, n, right_inclusive):
    """Indices of the n positions nearest below `anchor_left` and the n nearest above
    `anchor_right` (at or above when right_inclusive), returned in position order."""
    left = np.where(positions < anchor_left)[0]
    left = left[np.argsort(anchor_left - positions[left])[:n]]
    right = np.where(positions >= anchor_right if right_inclusive else positions > anchor_right)[0]
    right = right[np.argsort(positions[right] - anchor_right)[:n]]
    return np.sort(np.concatenate([left, right]))


def build_panel(vcf_path, chrom, strategy, centromeres, het, samples, unextended=(),
                min_maf=0.05, max_missing=0.05, count_haploid_once=True):
    """Return (genotypes, variant rows, sample names) for one chromosome and strategy."""
    kind, value = parse_strategy(strategy)
    c = centromeres.set_index('chrom').loc[chrom]
    start, end = int(c['start']), int(c['end'])
    if kind == 'fixed':
        regions = [f"{ch}:{s + 1}-{e}" for ch, s, e in fixed_windows(
            centromeres[centromeres['chrom'] == chrom], het, value, unextended)]
        return read_snps(vcf_path, regions, samples, min_maf, max_missing, count_haploid_once)
    if kind == 'edge':
        anchor_l, anchor_r, inclusive = start, end, False
    else:
        mid = int((start + end) / 2)
        anchor_l, anchor_r, inclusive = mid, mid, True
    region = f"{chrom}:{max(0, anchor_l - SEARCH)}-{anchor_r + SEARCH}"
    gts, rows, names = read_snps(vcf_path, [region], samples, min_maf, max_missing, count_haploid_once)
    if gts is None:
        return None, [], names
    keep = nearest(np.array([r['pos'] for r in rows]), anchor_l, anchor_r, value, inclusive)
    return gts[keep], [rows[i] for i in keep], names


def write_panel(outdir, chrom, gts, rows, names):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{chrom}_snp_genotypes.npz", genotypes=gts)
    pd.DataFrame(rows).to_csv(out / f"{chrom}_snp_variants.tsv", sep='\t', index=False)
    (out / f"{chrom}_snp_samples.txt").write_text('\n'.join(names) + '\n')
