"""Region iteration over a VCF, indexed (tabix/csi) or plain."""
from pathlib import Path


def is_indexed(path):
    return any(Path(str(path) + s).exists() for s in ('.tbi', '.csi'))


def fetch(reader, path, region):
    """Records of `region` ('chrom:start-end', 1-based inclusive): a region query when the VCF is
    indexed, otherwise a scan of the file."""
    if is_indexed(path):
        return reader(region)
    chrom, _, span = region.partition(':')
    start, _, end = span.partition('-')
    lo, hi = int(start), int(end)
    return (v for v in reader if v.CHROM == chrom and lo <= v.POS <= hi)
