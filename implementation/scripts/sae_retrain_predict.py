"""
Retrain SAE with joint reconstruction + prediction loss.

Architecture:
  x = concat(prot_proj, dna_proj)  # 1536-dim
  z = TopK(ReLU(W_enc @ x + b_enc), k)  # sparse concepts
  x_recon = W_dec @ z  # reconstruction
  score = w_pred @ z + w_human @ human_feats + bias  # prediction

Loss = pred_loss + λ_recon * recon_loss + λ_aux * aux_loss
  pred_loss: MSE(score, fitness_z) per-assay z-scored
  recon_loss: MSE(x_recon, x)
  aux_loss: L1 sparsity on z (encourage sparse activation)

Training: global pooled across all DMS assays.
Evaluation: leave-assay-out CV (train on N-1 assays, test on held-out).
Also: per-assay 5-fold CV for comparison with baselines.
"""

import argparse, glob, os, re, random
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset


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
    feats = np.zeros(9, dtype=np.float32)
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


class PredictiveSAE(nn.Module):
    def __init__(self, d_input=1536, n_features=12288, k=128, n_human=9):
        super().__init__()
        self.k = k
        self.n_features = n_features
        self.encoder = nn.Linear(d_input, n_features)
        self.decoder = nn.Linear(n_features, d_input, bias=False)
        self.pred_w = nn.Linear(n_features, 1, bias=False)
        self.pred_human = nn.Linear(n_human, 1, bias=False)
        self.pred_bias = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        z = torch.relu(self.encoder(x))
        if self.training or self.k >= self.n_features:
            if self.k < self.n_features:
                topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
                mask = torch.zeros_like(z)
                mask.scatter_(1, topk_idx, 1.0)
                z = z * mask
        else:
            topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, topk_idx, topk_vals)
            z = z_sparse
        return z

    def forward(self, x, human_feats):
        z = self.encode(x)
        x_recon = self.decoder(z)
        score = self.pred_w(z).squeeze(-1) + self.pred_human(human_feats).squeeze(-1) + self.pred_bias
        return score, x_recon, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_retrain")
    ap.add_argument("--n_features", type=int, default=12288)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lam_recon", type=float, default=0.5)
    ap.add_argument("--lam_sparse", type=float, default=0.01)
    ap.add_argument("--n_cv_folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: %s, k=%d" % (device, args.k), flush=True)

    # ── Load pretrained SAE for initialization ──
    print("Loading pretrained SAE ...", flush=True)
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))

    # ── Load all data ──
    print("Loading data ...", flush=True)
    sae_files = sorted(glob.glob(os.path.join(args.sae_dir, "*_sae.npz")))

    all_x = []
    all_hf = []
    all_y = []
    all_assay_idx = []
    assay_names = []

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
        if len(fitness) < 50:
            continue

        x = np.concatenate([prot_reps, dna_reps], axis=1).astype(np.float32)

        # Z-score fitness within assay
        y_z = ((fitness - fitness.mean()) / (fitness.std() + 1e-8)).astype(np.float32)

        # Human features
        n = len(mutants)
        hf = np.zeros((n, 9), dtype=np.float32)
        for i in range(n):
            ref_aa, pos, alt_aa = parse_mutant(mutants[i])
            if ref_aa and alt_aa and ref_aa in AA_TO_IDX and alt_aa in AA_TO_IDX:
                hf[i] = compute_human_features(ref_aa, alt_aa)

        assay_id = len(assay_names)
        assay_names.append(assay_name)
        all_x.append(x)
        all_hf.append(hf)
        all_y.append(y_z)
        all_assay_idx.append(np.full(n, assay_id, dtype=np.int64))

    all_x = np.vstack(all_x)
    all_hf = np.vstack(all_hf)
    all_y = np.concatenate(all_y)
    all_assay_idx = np.concatenate(all_assay_idx)

    n_assays = len(assay_names)
    print("Loaded %d variants from %d assays" % (len(all_x), n_assays), flush=True)

    # Convert to tensors
    X_tensor = torch.FloatTensor(all_x)
    HF_tensor = torch.FloatTensor(all_hf)
    Y_tensor = torch.FloatTensor(all_y)

    # ===================================================================
    # APPROACH 1: Global training with per-assay CV evaluation
    # Train on all data, evaluate per-assay Spearman
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("APPROACH 1: Global training, per-assay evaluation", flush=True)
    print("=" * 70, flush=True)

    # Split assays into folds for leave-group-out CV
    assay_ids = list(range(n_assays))
    random.seed(42)
    random.shuffle(assay_ids)
    fold_size = n_assays // args.n_cv_folds

    fold_results = []
    for fold in range(args.n_cv_folds):
        # Test assays for this fold
        test_assays = set(assay_ids[fold * fold_size: (fold + 1) * fold_size])
        train_mask = np.array([a not in test_assays for a in all_assay_idx])
        test_mask = ~train_mask

        print("\nFold %d: train=%d variants, test=%d variants (%d assays)" % (
            fold, train_mask.sum(), test_mask.sum(), len(test_assays)), flush=True)

        # Build model
        model = PredictiveSAE(d_input=1536, n_features=args.n_features,
                              k=args.k, n_human=9).to(device)

        # Initialize from pretrained
        with torch.no_grad():
            model.encoder.weight.copy_(sd["encoder.weight"])
            model.encoder.bias.copy_(sd["encoder.bias"])
            model.decoder.weight.copy_(sd["decoder.weight"])

        # Training
        train_dataset = TensorDataset(
            X_tensor[train_mask].to(device),
            HF_tensor[train_mask].to(device),
            Y_tensor[train_mask].to(device)
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

        for epoch in range(args.n_epochs):
            model.train()
            epoch_pred_loss = 0
            epoch_recon_loss = 0
            n_batches = 0
            for xb, hfb, yb in train_loader:
                score, x_recon, z = model(xb, hfb)
                pred_loss = nn.functional.mse_loss(score, yb)
                recon_loss = nn.functional.mse_loss(x_recon, xb)
                sparse_loss = z.abs().mean()
                loss = pred_loss + args.lam_recon * recon_loss + args.lam_sparse * sparse_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_pred_loss += pred_loss.item()
                epoch_recon_loss += recon_loss.item()
                n_batches += 1

            scheduler.step()
            if (epoch + 1) % 5 == 0:
                print("  Epoch %d: pred=%.4f recon=%.4f" % (
                    epoch + 1, epoch_pred_loss / n_batches, epoch_recon_loss / n_batches),
                    flush=True)

        # Evaluate on test assays
        model.eval()
        with torch.no_grad():
            test_scores, _, test_z = model(
                X_tensor[test_mask].to(device),
                HF_tensor[test_mask].to(device)
            )
            test_scores = test_scores.cpu().numpy()

        test_y = all_y[test_mask]
        test_assay = all_assay_idx[test_mask]

        for aid in test_assays:
            amask = test_assay == aid
            if amask.sum() < 20:
                continue
            rho = stats.spearmanr(test_scores[amask], test_y[amask]).statistic
            if not np.isnan(rho):
                fold_results.append({
                    'fold': fold,
                    'assay': assay_names[aid],
                    'n': amask.sum(),
                    'spearman': rho,
                })

        # Sparsity stats
        with torch.no_grad():
            sample_z = model.encode(X_tensor[:1000].to(device))
            alive = (sample_z > 0).float().mean(dim=0)
            n_alive = (alive > 0.01).sum().item()
            mean_active = (sample_z > 0).float().sum(dim=1).mean().item()
            print("  Alive features: %d, mean active per variant: %.1f" % (
                n_alive, mean_active), flush=True)

    cv_df = pd.DataFrame(fold_results)
    cv_df.to_csv(os.path.join(args.out_dir, "cv_results.csv"), index=False)

    mean_rho = cv_df['spearman'].mean()
    median_rho = cv_df['spearman'].median()
    print("\n" + "=" * 70, flush=True)
    print("LEAVE-ASSAY-OUT CV RESULTS (k=%d)" % args.k, flush=True)
    print("=" * 70, flush=True)
    print("Mean Spearman:   %.4f" % mean_rho, flush=True)
    print("Median Spearman: %.4f" % median_rho, flush=True)
    print("Assays evaluated: %d" % len(cv_df), flush=True)

    # ===================================================================
    # APPROACH 2: Train on ALL data, report per-assay Spearman (in-sample)
    # For comparison with per-assay ridge baselines
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("APPROACH 2: Train on all, per-assay in-sample Spearman", flush=True)
    print("=" * 70, flush=True)

    model_full = PredictiveSAE(d_input=1536, n_features=args.n_features,
                                k=args.k, n_human=9).to(device)
    with torch.no_grad():
        model_full.encoder.weight.copy_(sd["encoder.weight"])
        model_full.encoder.bias.copy_(sd["encoder.bias"])
        model_full.decoder.weight.copy_(sd["decoder.weight"])

    full_dataset = TensorDataset(X_tensor.to(device), HF_tensor.to(device), Y_tensor.to(device))
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(model_full.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    for epoch in range(args.n_epochs):
        model_full.train()
        epoch_loss = 0
        n_batches = 0
        for xb, hfb, yb in full_loader:
            score, x_recon, z = model_full(xb, hfb)
            pred_loss = nn.functional.mse_loss(score, yb)
            recon_loss = nn.functional.mse_loss(x_recon, xb)
            sparse_loss = z.abs().mean()
            loss = pred_loss + args.lam_recon * recon_loss + args.lam_sparse * sparse_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_full.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print("  Epoch %d: loss=%.4f" % (epoch + 1, epoch_loss / n_batches), flush=True)

    # Per-assay in-sample evaluation
    model_full.eval()
    with torch.no_grad():
        all_scores, _, all_z = model_full(X_tensor.to(device), HF_tensor.to(device))
        all_scores = all_scores.cpu().numpy()

    insample_results = []
    for aid in range(n_assays):
        amask = all_assay_idx == aid
        if amask.sum() < 20:
            continue
        rho = stats.spearmanr(all_scores[amask], all_y[amask]).statistic
        if not np.isnan(rho):
            insample_results.append({
                'assay': assay_names[aid],
                'n': amask.sum(),
                'spearman': rho,
            })

    insample_df = pd.DataFrame(insample_results)
    insample_df.to_csv(os.path.join(args.out_dir, "insample_results.csv"), index=False)
    print("In-sample: mean=%.4f, median=%.4f (%d assays)" % (
        insample_df['spearman'].mean(), insample_df['spearman'].median(),
        len(insample_df)), flush=True)

    # Save model
    torch.save({
        'state_dict': model_full.state_dict(),
        'k': args.k,
        'n_features': args.n_features,
    }, os.path.join(args.out_dir, "predictive_sae.pt"))

    # ===================================================================
    # Concept analysis on retrained model
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("CONCEPT ANALYSIS (retrained)", flush=True)
    print("=" * 70, flush=True)

    pred_w = model_full.pred_w.weight.data.cpu().numpy().squeeze()
    human_w = model_full.pred_human.weight.data.cpu().numpy().squeeze()

    # Modality from encoder weights
    enc_w = model_full.encoder.weight.data.cpu().numpy()
    prot_norm = np.linalg.norm(enc_w[:, :768], axis=1)
    dna_norm = np.linalg.norm(enc_w[:, 768:], axis=1)
    mod_score = prot_norm / (prot_norm + dna_norm + 1e-8)

    # Top concepts by |prediction weight|
    abs_pred_w = np.abs(pred_w)
    top_idx = np.argsort(-abs_pred_w)[:50]

    print("\nTop 30 predictive concepts:", flush=True)
    print("%-5s %-8s %10s %8s %8s" % ("Rank", "Feature", "Weight", "Modality", "ModScore"),
          flush=True)
    for rank, fi in enumerate(top_idx[:30]):
        mod = 'PROT' if mod_score[fi] > 0.6 else ('DNA' if mod_score[fi] < 0.4 else 'CROSS')
        direction = 'damaging' if pred_w[fi] < 0 else 'protect'
        print("%-5d F%05d %+10.5f %8s %8.3f  %s" % (
            rank + 1, fi, pred_w[fi], mod, mod_score[fi], direction), flush=True)

    print("\nHuman feature weights:", flush=True)
    human_names = ['charge_chg', 'volume_chg', 'hydro_chg', 'cys_loss',
                   'pro_intro', 'pro_loss', 'gly_loss', 'trp_loss', 'charge_rev']
    for i, name in enumerate(human_names):
        print("  %-15s %+.5f" % (name, human_w[i]), flush=True)

    # Weight mass by modality
    prot_mass = np.abs(pred_w[mod_score > 0.6]).sum()
    dna_mass = np.abs(pred_w[mod_score < 0.4]).sum()
    cross_mass = np.abs(pred_w[(mod_score >= 0.4) & (mod_score <= 0.6)]).sum()
    total = prot_mass + dna_mass + cross_mass
    print("\nPrediction weight mass: PROT=%.1f%% DNA=%.1f%% CROSS=%.1f%%" % (
        100 * prot_mass / total, 100 * dna_mass / total, 100 * cross_mass / total), flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
