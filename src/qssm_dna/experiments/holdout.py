"""Independent promoter holdout on chromosomes 9 and 10. CPU only.

Positives are non-TATA EPDnew promoters, negatives are gene-body windows
after the first exon, and no window overlaps a Genomic Benchmarks interval;
RULES spells out the construction and is stored in the manifest.

Sources (downloaded once into data/raw/, SHA-256 recorded):
  EPDnew Homo sapiens release 006, hg38: Hs_EPDnew_006_hg38.sga (1-based TSS)
      and db/promoter_motifs.txt (TATA-box flag)
  Genomic Benchmarks human_nontata_promoters GRCh38 coordinates
      (datasets/human_nontata_promoters/{train,test}/{positive,negative}.csv.gz;
      0-based start, exclusive end)
  UCSC hg38 refGene.txt.gz (gene bodies for negatives)
  hg38 chr9/chr10 FASTA already in data/fasta (UCSC, hashed in
      data/data_manifest.json)
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from ..data.fasta import download, read_fasta_sequences
from ..data.tokenizer import canonical_hash

RAW = Path("data/raw")
OUT = Path("data/holdout_promoter_chr9_chr10.npz")
EPD = "https://epd.expasy.org/ftp/epdnew/H_sapiens/006"
GB = ("https://raw.githubusercontent.com/ML-Bioinfo-CEITEC/"
      "genomic_benchmarks/main/datasets/human_nontata_promoters")
SOURCES = {
    "epd_sga": f"{EPD}/Hs_EPDnew_006_hg38.sga",
    "epd_motifs": f"{EPD}/db/promoter_motifs.txt",
    "refgene": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/"
               "refGene.txt.gz",
    **{f"gb_{s}_{c}": f"{GB}/{s}/{c}.csv.gz"
       for s in ("train", "test") for c in ("positive", "negative")},
}
CHROMS = {"NC_000009.12": "chr9", "NC_000010.11": "chr10"}
WIN = 251
NEG_SEED = 2026
NEG_RATIO = 4119 / 4915
TSS_PAD = 1000
ISOLATION_BP = 10_000
COMP = str.maketrans("ACGTN", "TGCAN")
RULES = {
    "positives": "EPDnew Hs 006 hg38 promoters on chr9 and chr10 with TATA "
                 "flag 0; window -200..+50 around the 1-based TSS, reverse "
                 "complemented on the minus strand",
    "negatives": "longest NM_ refGene transcript with at least two exons per "
                 "gene; 251-base windows downstream of the first exon in "
                 "sense orientation, drawn uniformly without overlap (seed "
                 "2026); n_neg = round(n_pos * 4119 / 4915), split across "
                 "chromosomes in proportion to eligible positions",
    "exclusions": "any overlap with a Genomic Benchmarks promoter-task "
                  "interval; any N; for negatives also +/-1000 bases around "
                  "every EPDnew TSS and the selected positives",
    "duplicates": "a window whose sequence equals a benchmark sequence "
                  "(exact or reverse complement) is removed after the "
                  "negatives are drawn",
    "isolated10kb": "positives with no benchmark positive interval within "
                    "10 kb (negatives are flagged true)",
}


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def fetch_sources() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    info = {}
    for key, url in SOURCES.items():
        info[key] = download(url, RAW / f"{key}{Path(url).suffix}"
                             if not url.endswith(".csv.gz")
                             else RAW / f"{key}.csv.gz")
    return info


def _read_text(p: Path) -> str:
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def load_gb_intervals() -> list[tuple]:
    """(chrom, start, end, strand, split, label) for every benchmark row."""
    rows = []
    for split in ("train", "test"):
        for cls, lab in (("positive", 1), ("negative", 0)):
            txt = _read_text(RAW / f"gb_{split}_{cls}.csv.gz")
            lines = txt.strip().splitlines()
            head = lines[0].split(",")
            ix = {k: head.index(k) for k in
                  ("region", "start", "end", "strand")}
            for ln in lines[1:]:
                f = ln.split(",")
                rows.append((f[ix["region"]], int(f[ix["start"]]),
                             int(f[ix["end"]]), f[ix["strand"]], split, lab))
    return rows


def load_epd() -> list[dict]:
    tata = {}
    for ln in _read_text(RAW / "epd_motifs.txt").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split()
        tata[f[0]] = int(f[1])
    out = []
    for ln in _read_text(RAW / "epd_sga.sga").splitlines():
        f = ln.split()
        if len(f) < 6:
            continue
        out.append({"acc": f[0], "chrom": CHROMS.get(f[0]), "tss": int(f[2]),
                    "strand": f[3], "name": f[5],
                    "tata": tata.get(f[5])})
    return out


def load_refgene() -> list[dict]:
    out = []
    for ln in _read_text(RAW / "refgene.gz").splitlines():
        f = ln.split("\t")
        if len(f) < 13 or not f[1].startswith("NM_"):
            continue
        out.append({"name": f[1], "chrom": f[2], "strand": f[3],
                    "txStart": int(f[4]), "txEnd": int(f[5]),
                    "exonCount": int(f[8]),
                    "exonStarts": [int(x) for x in f[9].strip(",").split(",")],
                    "exonEnds": [int(x) for x in f[10].strip(",").split(",")],
                    "gene": f[12]})
    return out


_CODE = np.full(256, -1, dtype=np.int64)
for _i, _c in enumerate("ACGT"):
    _CODE[ord(_c)] = _i


def _kmer_codes(seq: str, k: int = 31) -> np.ndarray:
    """2-bit integer codes of every N-free k-mer (k <= 31)."""
    a = _CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]
    n = len(a) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    bad = np.convolve((a < 0).astype(np.int64), np.ones(k, np.int64),
                      "valid") > 0
    a = np.where(a < 0, 0, a)
    code = np.zeros(n, dtype=np.int64)
    for j in range(k):
        code = (code << 2) | a[j:j + n]
    return code[~bad]


def _window(tss1: int, strand: str) -> tuple[int, int]:
    """0-based half-open genomic interval of the -200..+50 window."""
    if strand == "+":
        return tss1 - 201, tss1 + 50
    return tss1 - 51, tss1 + 200


def _mask(length: int, intervals) -> np.ndarray:
    m = np.zeros(length + 1, dtype=np.int32)
    for s, e in intervals:
        s, e = max(s, 0), min(e, length)
        if e > s:
            m[s] += 1
            m[e] -= 1
    return np.cumsum(m[:-1]) > 0


def _clean_windows(blocked: np.ndarray) -> np.ndarray:
    """bool[i]: window [i, i+WIN) contains no blocked base."""
    c = np.concatenate([[0], np.cumsum(blocked, dtype=np.int64)])
    n = len(blocked) - WIN + 1
    return (c[WIN:WIN + n] - c[:n]) == 0


def build() -> dict:
    src = fetch_sources()
    gb = load_gb_intervals()
    epd = load_epd()
    genes = load_refgene()
    fasta = {}
    for ch in ("chr9", "chr10"):
        fasta[ch] = read_fasta_sequences(f"data/fasta/{ch}.fa.gz")[ch]

    # --- verification: GB coordinates reproduce our cached benchmark copy
    npz = np.load("data/benchmarks.npz", allow_pickle=False)
    gb_seqs = set(npz["train_seq"].tolist()) | set(npz["test_seq"].tolist())
    gb_canon = {canonical_hash(s) for s in gb_seqs}
    coord_check = {"n": 0, "found_in_cached_copy": 0}
    for ch, s, e, st, _, _ in gb:
        if ch not in fasta:
            continue
        seq = fasta[ch][s:e].upper()
        seq = rc(seq) if st == "-" else seq
        coord_check["n"] += 1
        coord_check["found_in_cached_copy"] += int(seq in gb_seqs)

    rng = np.random.default_rng(NEG_SEED)
    pos_rows, neg_rows = [], []
    epd_tss = {ch: [r["tss"] for r in epd if r["chrom"] == ch]
               for ch in fasta}
    counts = {}
    for ch, chseq in fasta.items():
        L = len(chseq)
        gb_iv = [(s, e) for c, s, e, *_ in gb if c == ch]
        gb_pos_iv = sorted((s, e) for c, s, e, _, _, lab in gb
                           if c == ch and lab == 1)
        nbase = np.frombuffer(chseq.upper().encode(), dtype=np.uint8) \
            == ord("N")
        gb_mask = _mask(L, gb_iv)
        # positives
        cand = [r for r in epd if r["chrom"] == ch]
        n_cand = len(cand)
        n_tata1 = sum(1 for r in cand if r["tata"] == 1)
        n_tata_missing = sum(1 for r in cand if r["tata"] is None)
        seen = set()
        n_gb_overlap = n_n = n_dup = 0
        for r in cand:
            if r["tata"] != 0:
                continue
            s, e = _window(r["tss"], r["strand"])
            if (s, e, r["strand"]) in seen:
                n_dup += 1
                continue
            if gb_mask[s:e].any():
                n_gb_overlap += 1
                continue
            if nbase[s:e].any():
                n_n += 1
                continue
            seen.add((s, e, r["strand"]))
            seq = chseq[s:e].upper()
            seq = rc(seq) if r["strand"] == "-" else seq
            # distance to the nearest benchmark positive interval
            d = min((max(ps - e, s - pe, 0) for ps, pe in gb_pos_iv),
                    default=10 ** 9)
            pos_rows.append({"seq": seq, "y": 1, "chrom": ch, "start": s,
                             "end": e, "strand": r["strand"],
                             "source": r["name"],
                             "isolated10kb": bool(d > ISOLATION_BP)})
        n_pos_ch = sum(1 for p in pos_rows if p["chrom"] == ch)
        # negatives
        blocked = gb_mask | nbase | _mask(
            L, [(t - 1 - TSS_PAD, t + TSS_PAD) for t in epd_tss[ch]]) \
            | _mask(L, [(p["start"], p["end"]) for p in pos_rows
                        if p["chrom"] == ch])
        clean = _clean_windows(blocked)
        best = {}   # longest NM_ transcript per gene; ties -> smaller name
        for g in genes:
            if g["chrom"] != ch or g["exonCount"] < 2:
                continue
            k = g["gene"]
            span = g["txEnd"] - g["txStart"]
            if (k not in best or span > best[k][0]
                    or (span == best[k][0] and g["name"] < best[k][1]["name"])):
                best[k] = (span, g)
        starts, strands, labels = [], [], []
        for k in sorted(best):
            g = best[k][1]
            if g["strand"] == "+":
                a, b = g["exonEnds"][0], g["txEnd"]
            else:
                a, b = g["txStart"], g["exonStarts"][-1]
            if b - a < WIN:
                continue
            idx = np.arange(a, b - WIN + 1)
            idx = idx[clean[idx]]
            starts.append(idx)
            strands.append(np.full(len(idx), g["strand"] == "+"))
            labels += [f"{k}|{g['name']}"] * len(idx)
        allst = np.concatenate(starts)
        allpl = np.concatenate(strands)
        lab_arr = np.array(labels)
        counts[ch] = {"epd_promoters": n_cand, "tata1": n_tata1,
                      "tata_missing": n_tata_missing,
                      "nontata_dropped_gb_overlap": n_gb_overlap,
                      "nontata_dropped_N": n_n,
                      "nontata_duplicate_windows": n_dup,
                      "positives": n_pos_ch,
                      "negative_genes": len(best),
                      "eligible_negative_positions": int(len(allst))}
        counts[ch]["_alloc"] = (allst, allpl, lab_arr)

    n_pos = len(pos_rows)
    n_neg = int(round(n_pos * NEG_RATIO))
    # allocate negatives across chromosomes in proportion to eligible
    # positions, then sample positions uniformly without overlap
    elig = {ch: counts[ch]["eligible_negative_positions"] for ch in fasta}
    tot = sum(elig.values())
    quota = {ch: int(round(n_neg * elig[ch] / tot)) for ch in fasta}
    quota[list(fasta)[-1]] += n_neg - sum(quota.values())
    for ch in fasta:
        allst, allpl, lab_arr = counts[ch].pop("_alloc")
        order = rng.permutation(len(allst))
        taken = np.zeros(len(fasta[ch]) + WIN, dtype=bool)
        got = 0
        for i in order:
            s = int(allst[i])
            if taken[s:s + WIN].any():
                continue
            taken[s:s + WIN] = True
            seq = fasta[ch][s:s + WIN].upper()
            plus = bool(allpl[i])
            neg_rows.append({"seq": seq if plus else rc(seq), "y": 0,
                             "chrom": ch, "start": s, "end": s + WIN,
                             "strand": "+" if plus else "-",
                             "source": str(lab_arr[i]),
                             "isolated10kb": True})
            got += 1
            if got == quota[ch]:
                break
        counts[ch]["negatives"] = got

    rows = pos_rows + neg_rows
    # overlap clusters (bootstrap units)
    cl = np.zeros(len(rows), dtype=np.int64)
    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i]["chrom"], rows[i]["start"]))
    cid, last_ch, last_end = -1, None, -1
    for i in order:
        r = rows[i]
        if r["chrom"] != last_ch or r["start"] >= last_end:
            cid += 1
            last_end = r["end"]
        else:
            last_end = max(last_end, r["end"])
        last_ch = r["chrom"]
        cl[i] = cid

    # a window whose sequence equals a benchmark sequence located elsewhere
    # in the genome (exact or reverse complement) is removed; the other
    # windows keep their cluster numbers
    dup = {i for i, r in enumerate(rows)
           if canonical_hash(r["seq"]) in gb_canon}
    removed = [{k: rows[i][k] for k in ("chrom", "start", "end", "strand",
                                        "source", "y")} for i in sorted(dup)]
    keep = [i for i in range(len(rows)) if i not in dup]
    rows = [rows[i] for i in keep]
    cl = cl[keep]

    # verification of the finished set
    canon_dup = sum(canonical_hash(r["seq"]) in gb_canon for r in rows)
    gb_iv_all = {}
    for c, s, e, *_ in gb:
        gb_iv_all.setdefault(c, []).append((s, e))
    overlap = 0
    for r in rows:
        ivs = gb_iv_all.get(r["chrom"], [])
        overlap += any(s < r["end"] and r["start"] < e for s, e in ivs)
    gb_k = np.unique(np.concatenate(
        [_kmer_codes(t) for s in gb_seqs for t in (s, rc(s))]))
    share = sum(bool(np.isin(_kmer_codes(r["seq"]), gb_k,
                             assume_unique=False).any()) for r in rows)
    verification = {
        "gb_coordinate_reproduction_chr9_chr10": coord_check,
        "interval_overlap_with_benchmark": int(overlap),
        "exact_or_rc_duplicates_with_benchmark": int(canon_dup),
        "windows_sharing_a_31mer_with_benchmark": int(share),
        "windows_total": len(rows),
        "contains_N": int(sum("N" in r["seq"] for r in rows)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, seq=np.array([r["seq"] for r in rows]),
        y=np.array([r["y"] for r in rows], dtype=np.int64),
        chrom=np.array([r["chrom"] for r in rows]),
        start=np.array([r["start"] for r in rows], dtype=np.int64),
        end=np.array([r["end"] for r in rows], dtype=np.int64),
        strand=np.array([r["strand"] for r in rows]),
        source=np.array([r["source"] for r in rows]),
        isolated10kb=np.array([r["isolated10kb"] for r in rows]),
        cluster=cl)
    man = {"rules": RULES,
           "sources": src,
           "n_pos": int(sum(r["y"] == 1 for r in rows)),
           "n_neg": int(sum(r["y"] == 0 for r in rows)),
           "n_pos_isolated10kb": int(sum(r["isolated10kb"] for r in rows
                                         if r["y"] == 1)),
           "n_clusters": int(len(np.unique(cl))), "per_chrom": counts,
           "removed_duplicates": removed,
           "verification": verification,
           "npz_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}
    OUT.with_suffix(".manifest.json").write_text(
        json.dumps(man, indent=2, default=str))
    return man
