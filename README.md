# QSSM-DNA

Code and result files for

> H. Kong, "QSSM-DNA: DNA pretraining and transfer in a four-qubit quantum state-space model", *Physica Scripta* (2026).

QSSM-DNA is a recurrent DNA encoder whose state is the density matrix of four qubits. Each nucleotide selects a dissipative channel: amplitude damping on every qubit, followed by single-qubit rotations and ZZ couplings along the chain. The encoder is pretrained on next-base prediction in exact classical simulation, where the state after every token can be supervised from one forward pass, and it is read only at its terminal state, in the Z and X bases. The paper tests how the positions supervised in each training window affect what the pretrained encoder transfers to promoter and enhancer classification.

This repository contains the simulator, the training and evaluation code, the configuration of every condition in the paper, the tests, the construction of the independent promoter holdout with the holdout itself, and the result files behind the tables and figures.

## Layout

| Path | Content |
|---|---|
| `src/qssm_dna/quantum` | gates, amplitude damping, density-matrix updates, readouts, evaluation-time noise, a dense NumPy reference |
| `src/qssm_dna/models` | token controller, QSSM encoder, heads, GRU encoder |
| `src/qssm_dna/train` | window sampling, next-base pretraining, checkpoints, linear probes |
| `src/qssm_dna/experiments` | evaluation sets, the promoter holdout, probe evaluation, k-mer and CNN references, noise and runtime, statistics, the state-affine form |
| `configs` | one file per pretraining condition; `experiments.json` lists seeds, budgets and settings |
| `scripts` | analysis scripts that write `results/` and `figures/` |
| `tests` | unit tests of the quantum core, the sequence semantics, training and the experiment helpers |
| `data` | the promoter holdout on chromosomes 9 and 10 (sequences, coordinates, labels) |
| `results` | per-seed metrics and every contrast reported in the paper, as CSV |

## Installation

Python 3.10 or later and PyTorch 2.0 or later.

```bash
git clone https://github.com/Gumgo91/QSSM-DNA.git
cd QSSM-DNA
pip install -e ".[test,data,analysis]"
```

The extras are optional: `test` (pytest), `data` (the `genomic-benchmarks` package, needed to download the benchmark tasks), `analysis` (matplotlib, for the figures) and `reference` (PennyLane, used by one cross-check in the tests).

## Tests

```bash
pytest
```

The tests run on a CPU in under a minute and need no data. They compare the PyTorch updates with a dense NumPy reference and with the ancilla dilation of the damping channel, and check unitarity, trace, positivity, gradients, padding, reverse complements, chunked execution, checkpoint reload and the experiment helpers.

## Using the encoder

```python
import torch
from qssm_dna.config import load_config
from qssm_dna.data.tokenizer import encode, pad_tokens
from qssm_dna.models.encoder import QSSMEncoder, encoder_kwargs
from qssm_dna.quantum import readout

cfg = load_config("configs/allpos.yaml")
enc = QSSMEncoder(**encoder_kwargs(cfg))            # four qubits, untrained

tokens, mask = pad_tokens([encode("ACGTTGCA"), encode("GGCATC")])
rho = enc.forward_state(torch.from_numpy(tokens), torch.from_numpy(mask))   # [2, 16, 16]

raw11 = readout.terminal_readout(rho, enc.q)        # <Z_i>, <X_i>, <Z_i Z_i+1>
subset30 = readout.subset_readout(rho, enc.q)       # all Z-parities and X-parities
```

## Reproducing the experiments

All commands are run from the repository root. Run directories are written to `runs/`; set `QSSM_DNA_RUNS` to a list of directories (separated by the path separator of the operating system) to read runs from elsewhere.

### 1. Data

```bash
python -m qssm_dna prepare-data --config configs/endpoint.yaml
python -m qssm_dna experiment --part fetch-tasks
```

The first command downloads hg38 chromosomes 22 (pretraining), 7 and 8 (validation) and 9 and 10 from UCSC and the `human_nontata_promoters` benchmark; the second downloads `human_enhancers_cohn` and `human_ocr_ensembl`. The promoter holdout is in `data/`; `python -m qssm_dna experiment --part holdout` rebuilds it from EPDnew release 006, the benchmark coordinates and the UCSC refGene table.

### 2. Pretraining

```bash
python -m qssm_dna pretrain --config configs/dense8.yaml --model qssm --seed 22
python -m qssm_dna pretrain --config configs/dense8.yaml --model qssm --seed 22 --token-budget 2500000
```

Each condition was run with seeds 22, 33, 44, 55, 66, 77, 88, 99, 110 and 121.

| Condition | Config | Model | Token budgets |
|---|---|---|---|
| `qssm_endpoint` (one target per window) | `endpoint.yaml` | `qssm` | 0.25 M, 2.5 M |
| `qssm_dense2`, `qssm_dense4` | `dense2.yaml`, `dense4.yaml` | `qssm` | 0.25 M |
| `qssm_dense8` (last eight positions) | `dense8.yaml` | `qssm` | 0.25 M, 2.5 M |
| `qssm_spread8` (eight evenly spaced positions) | `spread8.yaml` | `qssm` | 0.25 M |
| `qssm_allpos` (all positions) | `allpos.yaml` | `qssm` | 0.25 M, 2.5 M |
| `qssm_dense8_w256` (256-base windows) | `dense8_w256.yaml` | `qssm` | 2.5 M |
| `qssm_dense8_zerogamma`, `_staticgamma`, `_norzz` | `dense8_zerogamma.yaml`, `dense8_staticgamma.yaml`, `dense8_norzz.yaml` | `qssm` | 0.25 M |
| `qssm_allpos_norzz` | `allpos_norzz.yaml` | `qssm` | 0.25 M |
| `qssm_allpos_q6`, `qssm_allpos_norzz_q6` (six qubits) | `allpos_q6.yaml`, `allpos_norzz_q6.yaml` | `qssm` | 0.25 M |
| `gru64_endpoint`, `gru64_dense8`, `gru64_allpos` | `endpoint.yaml`, `dense8.yaml`, `allpos.yaml` | `gru` | 0.25 M (`gru64_dense8` also 2.5 M) |
| `gru8_dense8`, `gru8_allpos` | `gru8_dense8.yaml`, `gru8_allpos.yaml` | `gru` | 0.25 M |

Runs at 2.5 M tokens carry the suffix `_b2500000` in their condition name.

### 3. Frozen-probe evaluation

```bash
python -m qssm_dna experiment --part eval-encoder --condition qssm_dense8 --seed 22
python -m qssm_dna experiment --part eval-encoder --condition qssm_random_init --seed 22
python -m qssm_dna experiment --part kmer-lr
python -m qssm_dna experiment --part cnn --seed 22
```

`eval-encoder` loads the pretrained run of a condition and seed, computes the terminal features of every evaluation set, fits the linear probes on the training split of each task and scores the test sets and the promoter holdout. `qssm_random_init` is the untrained register of a seed (`qssm_random_init_q6` for six qubits). The six-qubit comparison uses `--datasets gb_promoter_s6k,holdout_promoter`.

### 4. Shots, noise and runtime

```bash
python -m qssm_dna experiment --part noise --condition qssm_allpos --seed 22
python -m qssm_dna experiment --part runtime
python scripts/noise_reference.py
```

`noise_reference.py` evaluates the untrained registers and the encoders of the 2.5 M budget under shots and noise on a CPU.

### 5. Tables and figures

| Script | Output in `results/` |
|---|---|
| `make_results.py` | per-seed metrics (`all_metrics_long.csv`, `per_seed_mcc.csv`), condition table, every paired contrast, shot contrasts, `noise_metrics.csv`, `runtime.csv` |
| `bootstrap_primary.py` | sequence-level bootstrap of the primary contrasts |
| `nll_large.py` | next-base loss on 10,240 held-out windows with Markov references |
| `pauli_features.py`, `readout_table.py` | all 255 Pauli expectations and the readout comparison |
| `probe_robustness.py` | contrasts with a standardized logistic-regression probe |
| `matched_reference.py` | ablated encoders against untrained registers with the same component switched off |
| `memory_kernel.py`, `entanglement_analysis.py` | damping rates, memory kernel, negativity and purity |
| `composition_floor.py`, `composition_decoding.py`, `composition_link.py` | composition references and the base composition readable from a register |
| `scaling_analysis.py` | four against six qubits |
| `holdout_kmer31.py` | 31-mers shared between the holdout and the benchmark |
| `make_figures.py` | figures 2 to 5, drawn from `results/` alone into `figures/` |

`make_results.py` is run first; the other scripts read its tables, the run directories and the cached features under `runs/features`.

## Result files

`results/` holds the files produced by the scripts above for the runs of the paper. `all_metrics_long.csv` has one row per condition, seed, task, evaluation set and readout; `primary_contrasts.csv`, `dose_placement_contrasts.csv`, `ablation_contrasts.csv`, `allpos_contrasts.csv`, `budget_contrasts.csv`, `cross_budget_contrasts.csv`, `classical_frame.csv`, `readout_contrasts.csv` and `matched_reference_contrasts.csv` hold the paired differences over seeds with 95% intervals, paired t-tests and exact sign-flip permutation tests.

The per-seed prediction files of every evaluation run (probabilities, labels and thresholds for each evaluation set and readout) are attached to the release as `predictions.zip`. Unpacked into `runs/`, they are found by `scripts/bootstrap_primary.py`.

## Promoter holdout

`data/holdout_promoter_chr9_chr10.npz` contains 764 promoters without a TATA box from EPDnew release 006 and 641 gene-body windows on chromosomes 9 and 10 of hg38, each 251 bases long. No window overlaps or duplicates a sequence of the `human_nontata_promoters` benchmark.

| Field | Meaning |
|---|---|
| `seq` | sequence on the strand of the promoter or gene |
| `y` | 1 for promoters, 0 for gene-body windows |
| `chrom`, `start`, `end`, `strand` | hg38 coordinates, 0-based start and exclusive end |
| `source` | EPDnew promoter name, or gene and transcript of a negative window |
| `isolated10kb` | promoter with no benchmark promoter within 10 kb (always true for negatives) |
| `cluster` | index shared by overlapping windows, used as the unit of the bootstrap |

The manifest next to the file lists the construction rules, the source files with their SHA-256 hashes and the checks against the benchmark.

## Citation

```bibtex
@article{kong2026qssmdna,
  author  = {Kong, Hyunseung},
  title   = {{QSSM-DNA}: {DNA} pretraining and transfer in a four-qubit quantum state-space model},
  journal = {Physica Scripta},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
