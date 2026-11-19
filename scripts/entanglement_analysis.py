"""Entanglement and purity of terminal states (CPU).

  python scripts/entanglement_analysis.py

Negativity across the three chain cuts (q1|q2q3q4, q1q2|q3q4, q1q2q3|q4) and
purity of the terminal state of the first 400 promoter benchmark test
sequences (forward read, complex128), for each seed's matched
initialization and its endpoint, dense-tail-8, all-position and no-coupling
dense-tail-8 encoders.

Inputs : the pretraining runs under the run roots (jobs.ROOTS), the promoter
         benchmark (datasets.load)
Output : results/entanglement.csv
"""
import sys

import pandas as pd
import torch

sys.path.insert(0, "src")
from qssm_dna.config import load_config
from qssm_dna.data.tokenizer import encode, pad_tokens
from qssm_dna.experiments import datasets, encoders, jobs

SEEDS = [22, 33, 44, 55, 66, 77, 88, 99, 110, 121]
CONDS = ["qssm_random_init", "qssm_endpoint", "qssm_dense8", "qssm_allpos", "qssm_dense8_norzz"]

def negativity(rho, nA):
    """rho [B,16,16] complex128; partial transpose on the first nA qubits."""
    b = rho.shape[0]; dA = 2 ** nA; dB = 16 // dA
    r = rho.reshape(b, dA, dB, dA, dB).permute(0, 3, 2, 1, 4).reshape(b, 16, 16)
    ev = torch.linalg.eigvalsh(r)
    return (ev.clamp(max=0).abs().sum(-1)).numpy()

def main():
    refs = encoders.scan_runs(jobs.ROOTS)
    base = load_config("configs/endpoint.yaml")
    gb, _ = datasets.load("gb_promoter")
    seqs = gb["test"]["seqs"][:400]
    t, m = pad_tokens([encode(s) for s in seqs])
    t, m = torch.from_numpy(t), torch.from_numpy(m)
    rows = []
    for s in SEEDS:
        for c in CONDS:
            try:
                enc = (encoders.matched_random_init(base, s, "cpu", [])[0] if c == "qssm_random_init"
                       else encoders.load_encoder(encoders.pick_run(refs, c, s), "cpu"))
            except LookupError:
                continue
            enc = enc.double(); enc.cdtype = torch.complex128
            with torch.no_grad():
                rho = enc.forward_state(t, m)
            pur = torch.einsum("bij,bji->b", rho, rho).real.numpy()
            rows.append({"seed": s, "condition": c,
                         "neg_1_234": float(negativity(rho, 1).mean()),
                         "neg_12_34": float(negativity(rho, 2).mean()),
                         "neg_123_4": float(negativity(rho, 3).mean()),
                         "frac_states_entangled_12_34": float((negativity(rho, 2) > 1e-9).mean()),
                         "purity": float(pur.mean())})
    df = pd.DataFrame(rows)
    df.to_csv("results/entanglement.csv", index=False)
    print(df.groupby("condition")[["neg_1_234", "neg_12_34", "neg_123_4", "frac_states_entangled_12_34", "purity"]].agg(["mean", "std"]).round(4).to_string())

if __name__ == "__main__":
    main()
