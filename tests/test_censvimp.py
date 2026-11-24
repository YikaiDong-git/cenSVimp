import numpy as np
import pandas as pd
import pytest

from censvimp.ensemble import select
from censvimp.impute import impute_chromosome
from censvimp.panel import extended_edges, fixed_windows, nearest
from censvimp.train import train_chromosome

HET_COLS = ['chrom', 'start', 'end', 'type']


def het(rows):
    return pd.DataFrame(rows, columns=HET_COLS)


def test_edges_extend_over_adjacent_block_only():
    h = het([('c', 50, 90, 'heterochromatin'),        # 10 bp left of the centromere: extend
             ('c', 300, 400, 'heterochromatin'),      # 100 bp right: beyond proximity 50
             ('c', 0, 80, 'short_arm')])              # short arm: ignored
    assert extended_edges(100, 200, h, proximity=50) == (50, 200)
    cen = pd.DataFrame([('c', 100, 200)], columns=['chrom', 'start', 'end'])
    assert fixed_windows(cen, h, 10, proximity=50) == [('c', 40, 50), ('c', 200, 210)]
    assert fixed_windows(cen, h, 10, unextended={'c'}, proximity=50) == [('c', 90, 100), ('c', 200, 210)]


def test_block_over_boundary_extends_both_sides():
    assert extended_edges(100, 200, het([('c', 150, 260, 'heterochromatin')])) == (100, 260)


def test_nearest_selects_outward_and_sorts():
    pos = np.array([10, 40, 90, 95, 150, 205, 230, 300])
    assert nearest(pos, 100, 200, 2, False).tolist() == [2, 3, 5, 6]
    assert nearest(pos, 150, 150, 1, True).tolist() == [3, 4]


def test_select_ranks_by_cv_r2_then_sparsity():
    a = pd.DataFrame({'sv_index': [0, 1], 'cv_r2': [0.5, 0.2], 'n_nonzero_coef': [9, 3]})
    b = pd.DataFrame({'sv_index': [0, 1], 'cv_r2': [0.4, 0.2], 'n_nonzero_coef': [1, 2]})
    sel = select({'a': a, 'b': b}, ['a', 'b'])
    assert sel['strategy'].tolist() == ['a', 'b']          # SV 1 tie on cv_r2 -> fewer coefficients


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    """150 samples, 60 SNPs; SV dosage driven by SNPs 5 and 20."""
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp('run')
    n, m = 150, 60
    samples = [f"S{i:03d}" for i in range(n)]
    snp = rng.binomial(2, rng.uniform(0.2, 0.5, m)[:, None], (m, n)).astype(np.int8)
    sv = np.clip(np.round((snp[5] + snp[20]) / 2 + rng.normal(0, 0.2, n)), 0, 2).astype(np.int8)
    noise_sv = rng.binomial(2, 0.3, n).astype(np.int8)
    (d / 'panel').mkdir()
    np.savez_compressed(d / 'panel/chrS_snp_genotypes.npz', genotypes=snp)
    pd.DataFrame({'chrom': 'chrS', 'pos': 1000 + 100 * np.arange(m), 'ref': 'A', 'alt': 'G',
                  'id': [f"s{j}" for j in range(m)]}).to_csv(d / 'panel/chrS_snp_variants.tsv', sep='\t', index=False)
    (d / 'panel/chrS_snp_samples.txt').write_text('\n'.join(samples) + '\n')
    (d / 'sv').mkdir()
    # SV file lists the samples in reverse order (columns reversed to match): alignment is by ID
    np.savez_compressed(d / 'sv/chrS_sv_genotypes.npz', genotypes=np.vstack([sv, noise_sv])[:, ::-1])
    pd.DataFrame({'id': ['SV_real', 'SV_noise'], 'chrom': 'chrS', 'pos': [5000, 6000], 'end': [5100, 6100],
                  'svtype': 'DEL', 'svlen': 100}).to_csv(d / 'sv/chrS_sv_variants.tsv', sep='\t', index=False)
    (d / 'sv/chrS_sv_samples.txt').write_text('\n'.join(samples[::-1]) + '\n')
    out = train_chromosome(d / 'panel', d / 'sv', 'chrS', d / 'models')
    return d, out, snp, samples


def test_training_recovers_signal(trained):
    _, out, _, _ = trained
    m = pd.read_csv(out / 'sv_metrics.tsv', sep='\t').set_index('sv_id')
    assert m.loc['SV_real', 'cv_r2'] > 0.3
    assert m.loc['SV_noise', 'cv_r2'] < 0.1
    w = pd.read_csv(out / 'weights.tsv.gz', sep='\t')
    top = w[w.sv_index == 0].assign(a=lambda x: x.coef.abs()).nlargest(2, 'a')
    assert set(top.snp_pos) == {1500, 3000}


def write_vcf(path, snp, samples, pos, drop=(), flip=()):
    with open(path, 'w') as fh:
        fh.write('##fileformat=VCFv4.2\n##contig=<ID=chrS,length=100000>\n'
                 '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples) + '\n')
        for j, p in enumerate(pos):
            if j in drop:
                continue
            gt = '\t'.join(['0/0', '0/1', '1/1'][g] for g in snp[j])
            fh.write(f"chrS\t{p}\t.\tA\t{'T' if j in flip else 'G'}\t.\tPASS\t.\tGT\t{gt}\n")


def test_imputation_matches_weights_and_handles_missing_snps(trained, tmp_path):
    d, out, snp, samples = trained
    pos = 1000 + 100 * np.arange(snp.shape[0])
    w = pd.read_csv(out / 'weights.tsv.gz', sep='\t')
    used = sorted(set(w[w.sv_index == 0].snp_pos))
    gone, flipped = list(pos).index(used[0]), list(pos).index(used[1])
    vcf = tmp_path / 't.vcf'
    write_vcf(vcf, snp, samples, pos, drop={gone}, flip={flipped})
    res = impute_chromosome(out / 'models.tsv', out / 'weights.tsv.gz', vcf, 'chrS', samples[:20],
                            tmp_path / 'imp', min_cv_r2=0.1)
    dos = pd.read_csv(res / 'chrS_imputed_dosages.tsv', sep='\t').set_index('id')
    assert list(dos.index) == ['SV_real']

    m = pd.read_csv(out / 'models.tsv', sep='\t').set_index('id').loc['SV_real']
    w0 = w[w.sv_index == 0]
    x = np.array([[0.0 if p in (pos[gone], pos[flipped]) else snp[list(pos).index(p), i]
                   for p in w0.snp_pos] for i in range(20)])
    expect = np.clip(m.intercept + ((x - w0['mean'].values) / w0['scale'].values) @ w0.coef.values, 0, 2)
    assert np.allclose(dos.loc['SV_real', samples[:20]].values.astype(float), np.round(expect, 4), atol=1e-12)
