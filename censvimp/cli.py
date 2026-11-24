"""censvimp: impute centromeric SV genotypes from pericentromeric SNPs.

  censvimp panel    build a predictor-SNP panel for one strategy
  censvimp train    train per-SV elastic-net models on a panel
  censvimp merge    merge chunked training outputs
  censvimp select   best-of-N ensemble across strategies
  censvimp impute   impute dosages into SNP-genotyped samples
"""
import argparse
import importlib.resources as ir
from pathlib import Path

import pandas as pd

from . import __version__

UNEXTENDED = 'chr2,chr16,chr21'


def resource(name):
    return str(ir.files('censvimp') / 'resources' / name)


def read_ids(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith('#')]


def cmd_panel(a):
    from .panel import build_panel, read_bed, write_panel
    cen = read_bed(a.centromeres, ['chrom', 'start', 'end'])
    het = read_bed(a.heterochromatin, ['chrom', 'start', 'end', 'type'])
    unext = {c for c in a.unextended.split(',') if c}
    for chrom in a.chrom:
        vcf = a.vcf.format(chrom=chrom)
        gts, rows, names = build_panel(vcf, chrom, a.strategy, cen, het, read_ids(a.samples), unext,
                                       a.min_maf, a.max_missing, not a.count_haploid_twice)
        if gts is None:
            print(f"{chrom}: no SNPs pass the filters")
            continue
        write_panel(a.out, chrom, gts, rows, names)
        print(f"{chrom}: {len(rows)} SNPs x {len(names)} samples -> {a.out}")


def cmd_train(a):
    from .train import train_chromosome
    for chrom in a.chrom:
        train_chromosome(a.panel, Path(a.svs) / chrom, chrom, a.out, a.folds, a.threads, a.min_snp_freq,
                         a.max_snp_freq, a.min_sv_freq, a.max_sv_freq, a.sv_start, a.sv_end)


def cmd_merge(a):
    from .train import merge_chunks
    for chrom in a.chrom:
        merge_chunks(Path(a.dir) / chrom)


def cmd_select(a):
    from .ensemble import STRATEGIES_6, STRATEGIES_8, build_ensemble
    strategies = {'best8': STRATEGIES_8, 'best6': STRATEGIES_6}.get(a.strategies) or a.strategies.split(',')
    for chrom in a.chrom:
        build_ensemble(a.models, chrom, strategies, a.out, a.min_cv_r2)


def cmd_impute(a):
    from .impute import impute_chromosome
    samples = read_ids(a.samples)
    if a.exclude:
        drop = set(read_ids(a.exclude))
        samples = [s for s in samples if s not in drop]
    for chrom in a.chrom:
        d = Path(a.models) / chrom
        impute_chromosome(d / 'models.tsv', d / 'weights.tsv.gz', a.vcf.format(chrom=chrom), chrom,
                          samples, a.out, a.min_cv_r2)


def main(argv=None):
    p = argparse.ArgumentParser(prog='censvimp', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('panel', help='build a predictor-SNP panel')
    s.add_argument('--vcf', required=True, help='SNP VCF; may contain {chrom}, e.g. 1kGP.{chrom}.vcf.gz')
    s.add_argument('--samples', required=True, help='panel sample IDs, one per line')
    s.add_argument('--strategy', required=True, help='fixed:<bp> | edge:<n> | midpoint:<n>')
    s.add_argument('--chrom', required=True, nargs='+')
    s.add_argument('--out', required=True)
    s.add_argument('--centromeres', default=resource('grch38_centromeres.bed'))
    s.add_argument('--heterochromatin', default=resource('grch38_heterochromatin.bed'))
    s.add_argument('--unextended', default=UNEXTENDED,
                   help=f'chromosomes whose fixed windows start at the raw centromere edge (default {UNEXTENDED})')
    s.add_argument('--min-maf', type=float, default=0.05)
    s.add_argument('--max-missing', type=float, default=0.05)
    s.add_argument('--count-haploid-twice', action='store_true',
                   help='MAF filter counts a haploid call (chrX in males) as two alleles; '
                        'reproduces the published edge and midpoint panels on chrX')
    s.set_defaults(func=cmd_panel)

    s = sub.add_parser('train', help='train per-SV elastic-net models')
    s.add_argument('--panel', required=True, help='directory written by `censvimp panel`')
    s.add_argument('--svs', required=True, help='cenSVhap output directory (one subfolder per chromosome)')
    s.add_argument('--chrom', required=True, nargs='+')
    s.add_argument('--out', required=True)
    s.add_argument('--folds', type=int, default=4)
    s.add_argument('--threads', type=int, default=1)
    s.add_argument('--min-snp-freq', type=float, default=0.05)
    s.add_argument('--max-snp-freq', type=float, default=0.99)
    s.add_argument('--min-sv-freq', type=float, default=0.05)
    s.add_argument('--max-sv-freq', type=float, default=0.99)
    s.add_argument('--sv-start', type=int, help='train a chunk: first SV index (after filtering)')
    s.add_argument('--sv-end', type=int, help='train a chunk: last SV index, exclusive')
    s.set_defaults(func=cmd_train)

    s = sub.add_parser('merge', help='merge chunked training outputs')
    s.add_argument('--dir', required=True)
    s.add_argument('--chrom', required=True, nargs='+')
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser('select', help='best-of-N ensemble across strategies')
    s.add_argument('--models', required=True, help='directory holding one training output folder per strategy')
    s.add_argument('--strategies', default='best8', help='best8 | best6 | comma-separated folder names')
    s.add_argument('--chrom', required=True, nargs='+')
    s.add_argument('--out', required=True)
    s.add_argument('--min-cv-r2', type=float, default=0.1)
    s.set_defaults(func=cmd_select)

    s = sub.add_parser('impute', help='impute SV dosages')
    s.add_argument('--models', required=True, help='ensemble (or single-strategy) directory')
    s.add_argument('--vcf', required=True, help='target SNP VCF; may contain {chrom}')
    s.add_argument('--samples', required=True, help='target sample IDs')
    s.add_argument('--exclude', help='sample IDs to leave out (e.g. the training samples)')
    s.add_argument('--chrom', required=True, nargs='+')
    s.add_argument('--out', required=True)
    s.add_argument('--min-cv-r2', type=float, default=0.1)
    s.set_defaults(func=cmd_impute)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
