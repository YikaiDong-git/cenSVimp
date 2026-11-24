# cenSVimp

Imputation of centromeric structural variants from pericentromeric SNPs.

```bash
pip install .
censvimp impute --models weights/ --vcf cohort.{chrom}.vcf.gz --samples ids.txt --chrom chr1 chrX --out imputed/
```

`weights/` holds the published models (188 HPRC individuals, GRCh38 SNPs, components with cross-validated
R^2 >= 0.1). Training on new data: `censvimp panel`, `train`, `select` (see `censvimp --help`), with SV
genotypes from [cenSVhap](https://github.com/YikaiDong-git/cenSVhap).

License: MIT. Contact: yikai.dong@emory.edu, josh.weinstock@emory.edu

## Citation

Dong Y, Cutler D, Weinstock JS. Common centromeric variation associates with molecular and complex
traits in humans. *medRxiv* 2026. [10.64898/2026.10.09.26365154](https://doi.org/10.64898/2026.10.09.26365154)

```bibtex
@article{dong2026centromeric,
  title   = {Common centromeric variation associates with molecular and complex traits in humans},
  author  = {Dong, Yikai and Cutler, David and Weinstock, Joshua S.},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.09.26365154}
}
```
