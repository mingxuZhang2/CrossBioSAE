"""
End-to-end self-interpretable model: jointly optimize SAE + linear prediction head.

Architecture:
  x = concat(prot_proj, dna_proj)  # 1536-dim
  z = TopK(ReLU(W_enc @ x + b_enc), k)  # sparse concepts
  score = w_pred @ z + Σ(w_human * human_feats) + bias  # interpretable prediction

Loss = prediction_loss + λ * reconstruction_loss
  prediction_loss: MSE(score, fitness_z)
  reconstruction_loss: MSE(W_dec @ z, x)  # keeps SAE features meaningful

Training: per-assay leave-out CV, report Spearman.
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold


HYDROPHOBIC = set('AVILMFWP')
CHARGE_MAP = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}
AA_VOLUME = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
             'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
             'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
AA_HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
            'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
            'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def parse_mutant(mutant_str):
    mutant_str = str(mutant_str).strip()
    if ':' in mutant_str:
        mutant_str = mutant_str.split(':')[0]
    mutant_str = mutant_str.replace('p.', '')
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutant_str)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def compute_human_features(ref_aa, alt_aa):
    feats = np.zeros(9)
    rc = CHARGE_MAP.get(ref_aa, 0)
    ac = CHARGE_MAP.get(alt_aa, 0)
    feats[0] = ac - rc
    feats[1] = AA_VOLUME.get(alt_aa, 140) - AA_VOLUME.get(ref_aa, 140)
    feats[2] = AA_HYDRO.get(alt_aa, 0) - AA_HYDRO.get(ref_aa, 0)
    feats[3] = float(ref_aa == 'C')
    feats[4] = float(alt_aa == 'P')
    feats[5] = float(ref_aa == 'P')
    feats[6] = float(ref_aa == 'G')
    feats[7] = float(ref_aa == 'W')
    feats[8] = float(rc * ac < 0)
    return feats


class SelfExplainModel(nn.Module):
    """Self-explanatory model: SAE encoder → TopK → linear prediction."""

    def __init__(self, d_input=1536, n_features=12288, k=64, n_human=9):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_input, n_features)
        self.decoder = nn.Linear(n_features, d_input, bias=False)
        self.pred_sae = nn.Linear(n_features, 1, bias=False)
        self.pred_human = nn.Linear(n_human, 1, bias=False)
        self.pred_bias = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        z = torch.relu(self.encoder(x))
        if self.k < z.shape[1]:
            topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, topk_idx, topk_vals)
            return z_sparse
        return z

    def forward(self, x, human_feats):
        z = self.encode(x)
        x_recon = self.decoder(z)
        score = self.pred_sae(z) + self.pred_human(human_feats) + self.pred_bias
        return score.squeeze(-1), x_recon, z


def train_e2e_per_assay(model, X, hf, y, n_epochs=50, lr=1e-3, lam_recon=0.1,
                         batch_size=256, device='cpu'):
    """Train end-to-end model on a single assay."""
    model.train()
    dataset = TensorDataset(
        torch.FloatTensor(X).to(device),
        torch.FloatTensor(hf).to(device),
        torch.FloatTensor(y).to(device)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(n_epochs):
        for xb, hfb, yb in loader:
            score, x_recon, z = model(xb, hfb)
            pred_loss = nn.functional.mse_loss(score, yb)
            recon_loss = nn.functional.mse_loss(x_recon, xb)
            loss = pred_loss + lam_recon * recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def eval_model(model, X, hf, device='cpu'):
    model.eval()
    with torch.no_grad():
        score, _, z = model(
            torch.FloatTensor(X).to(device),
            torch.FloatTensor(hf).to(device)
        )
    return score.cpu().numpy(), z.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_e2e")
    ap.add_argument("--n_features", type=int, default=12288)
    ap.add_argument("--k_values", type=str, default="32,64,128,256")
    ap.add_argument("--n_epochs", type=int, default=30)
    ap.add_argument("--lam_recon", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    k_values = [int(k) for k in args.k_values.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: %s" % device, flush=True)

    # ── Load SAE model for initialization ──
    print("Loading SAE model ...", flush=True)
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    enc_w_init = sd["encoder.weight"]
    enc_b_init = sd["encoder.bias"]
    dec_w_init = sd["decoder.weight"]

    # ── Load data ──
    print("Loading representations ...", flush=True)
    sae_files = sorted(glob.glob(os.path.join(args.sae_dir, "*_sae.npz")))

    assay_data = []
    for sf in sae_files:
        assay_name = os.path.basename(sf).replace("_sae.npz", "")
        d = np.load(sf, allow_pickle=True)
        fitness = d["y_score"]
        mutants = d["mutants"] if "mutants" in d else np.array([])
        prot_reps = d["prot_reps"] if "prot_reps" in d else None
        dna_reps = d["dna_reps"] if "dna_reps" in d else None
        if prot_reps is None or dna_reps is None:
            continue
        if len(mutants) == 0 or len(mutants) != len(fitness):
            continue

        x = np.concatenate([prot_reps, dna_reps], axis=1).astype(np.float32)

        n = len(mutants)
        hf = np.zeros((n, 9), dtype=np.float32)
        for i in range(n):
            ref_aa, pos, alt_aa = parse_mutant(mutants[i])
            if ref_aa and alt_aa and ref_aa in AA_TO_IDX and alt_aa in AA_TO_IDX:
                hf[i] = compute_human_features(ref_aa, alt_aa)

        assay_data.append({
            'name': assay_name,
            'x': x,
            'human_feats': hf,
            'fitness': fitness.astype(np.float32),
        })

    print("Loaded %d assays" % len(assay_data), flush=True)

    # ── Per-assay 5-fold CV for each k ──
    print("\n" + "=" * 70, flush=True)
    print("END-TO-END PER-ASSAY BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    for ad_idx, ad in enumerate(assay_data):
        y = ad['fitness']
        if len(y) < 100:
            continue

        y_z = (y - y.mean()) / (y.std() + 1e-8)

        result = {'assay': ad['name'], 'n': len(y)}

        for k in k_values:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rhos = []
            for train_idx, test_idx in kf.split(ad['x']):
                model = SelfExplainModel(
                    d_input=1536, n_features=args.n_features, k=k, n_human=9
                ).to(device)

                # Initialize from pretrained SAE
                with torch.no_grad():
                    model.encoder.weight.copy_(enc_w_init)
                    model.encoder.bias.copy_(enc_b_init)
                    model.decoder.weight.copy_(dec_w_init)

                train_e2e_per_assay(
                    model, ad['x'][train_idx], ad['human_feats'][train_idx],
                    y_z[train_idx], n_epochs=args.n_epochs, lr=args.lr,
                    lam_recon=args.lam_recon, device=device
                )

                scores, _ = eval_model(model, ad['x'][test_idx],
                                        ad['human_feats'][test_idx], device=device)
                if np.std(scores) > 1e-8 and np.std(y[test_idx]) > 1e-8:
                    rho = stats.spearmanr(scores, y[test_idx]).statistic
                    if not np.isnan(rho):
                        rhos.append(rho)

            result['e2e_k%d' % k] = np.mean(rhos) if rhos else 0

        all_results.append(result)
        if (len(all_results)) % 20 == 0:
            # Print running average
            tmp_df = pd.DataFrame(all_results)
            for k in k_values:
                col = 'e2e_k%d' % k
                if col in tmp_df.columns:
                    print("  [%d assays] k=%d: mean=%.4f" % (
                        len(all_results), k, tmp_df[col].mean()), flush=True)

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(os.path.join(args.out_dir, "e2e_results.csv"), index=False)

    # ── Summary ──
    print("\n" + "=" * 70, flush=True)
    print("END-TO-END SUMMARY", flush=True)
    print("=" * 70, flush=True)

    for k in k_values:
        col = 'e2e_k%d' % k
        if col in res_df.columns:
            print("k=%-5d: mean=%.4f  median=%.4f" % (
                k, res_df[col].mean(), res_df[col].median()), flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
