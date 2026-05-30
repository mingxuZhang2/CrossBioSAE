"""
Phase 1: Contrastive pretraining of the cross-modal shared embedding.

Trains a CLIP-style dual-encoder on ~16k matched protein-DNA gene pairs.
Self-supervised — no labels, the pairing is the signal. After training, the
two encoders define a shared space where protein and DNA representations of
the same gene are aligned.

Large batch is critical for contrastive learning (more negatives per step).
On H100 80GB with 16k genes × (1280+1024) floats = ~150MB → entire dataset
fits in GPU RAM. We use the full dataset as one batch (perfect for InfoNCE).
"""

import argparse
import json
import logging
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_activations(h5_path):
    with h5py.File(h5_path, "r") as h:
        acts = h["activations"][:]
        names = [n.decode() if isinstance(n, bytes) else n for n in h["gene_names"][:]]
    return acts, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prot_h5", default="data/full/protein_activations.h5")
    ap.add_argument("--dna_h5", default="data/full/dna_activations.h5")
    ap.add_argument("--output_dir", default="results/pretrain")
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--d_shared", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=0,
                    help="0 = full dataset as one batch (best for contrastive)")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prot_acts, prot_names = load_activations(args.prot_h5)
    dna_acts, dna_names = load_activations(args.dna_h5)
    if prot_names != dna_names:
        logger.info("gene order differs between h5 files; aligning by name ...")
        common = sorted(set(prot_names) & set(dna_names))
        p_idx = {n: i for i, n in enumerate(prot_names)}
        d_idx = {n: i for i, n in enumerate(dna_names)}
        prot_acts = prot_acts[[p_idx[n] for n in common]]
        dna_acts = dna_acts[[d_idx[n] for n in common]]
        prot_names = dna_names = common
        logger.info(f"aligned to {len(common)} shared genes")
    n, d_prot = prot_acts.shape
    _, d_dna = dna_acts.shape
    logger.info(f"loaded {n} genes: protein {d_prot}-d, DNA {d_dna}-d")

    # train/val split by gene (random)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    logger.info(f"train={len(train_idx)} val={len(val_idx)}")

    # standardize per-feature (fit on train)
    p_mean, p_std = prot_acts[train_idx].mean(0), prot_acts[train_idx].std(0) + 1e-8
    d_mean, d_std = dna_acts[train_idx].mean(0), dna_acts[train_idx].std(0) + 1e-8
    prot_acts = (prot_acts - p_mean) / p_std
    dna_acts = (dna_acts - d_mean) / d_std

    dev = args.device if torch.cuda.is_available() else "cpu"
    Xp_tr = torch.tensor(prot_acts[train_idx], dtype=torch.float32, device=dev)
    Xd_tr = torch.tensor(dna_acts[train_idx], dtype=torch.float32, device=dev)
    Xp_va = torch.tensor(prot_acts[val_idx], dtype=torch.float32, device=dev)
    Xd_va = torch.tensor(dna_acts[val_idx], dtype=torch.float32, device=dev)
    logger.info(f"GPU tensors: train {Xp_tr.shape}, val {Xp_va.shape}")

    model = CrossModalCLIP(
        d_prot=d_prot, d_dna=d_dna, d_hidden=args.d_hidden,
        d_shared=args.d_shared, n_layers=args.n_layers,
        temperature=args.temperature,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    bs = args.batch_size if args.batch_size > 0 else len(train_idx)

    best_val_loss, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        if bs >= len(train_idx):
            zp, zd = model(Xp_tr, Xd_tr)
            loss = model.contrastive_loss(zp, zd)
            opt.zero_grad(); loss.backward(); opt.step()
        else:
            perm = torch.randperm(len(train_idx), device=dev)
            epoch_loss = 0.0
            for i in range(0, len(train_idx), bs):
                idx_b = perm[i:i+bs]
                zp, zd = model(Xp_tr[idx_b], Xd_tr[idx_b])
                loss = model.contrastive_loss(zp, zd)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()

        sched.step()
        model.eval()
        with torch.no_grad():
            zp_v, zd_v = model(Xp_va, Xd_va)
            val_loss = model.contrastive_loss(zp_v, zd_v).item()
            acc = model.retrieval_accuracy(zp_v, zd_v)

        if ep % 20 == 0 or ep <= 5:
            zp_t, zd_t = model(Xp_tr[:min(2000, len(Xp_tr))],
                               Xd_tr[:min(2000, len(Xd_tr))])
            tr_acc = model.retrieval_accuracy(zp_t, zd_t)
            logger.info(
                f"ep {ep:4d}  loss={val_loss:.4f}  "
                f"val_R@1 p2d={acc['p2d_acc']:.3f} d2p={acc['d2p_acc']:.3f}  "
                f"train_R@1 p2d={tr_acc['p2d_acc']:.3f}  "
                f"temp={model.log_temp.exp().item():.1f}  "
                f"lr={sched.get_last_lr()[0]:.1e}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                logger.info(f"early stopping at ep {ep}")
                break

    elapsed = time.time() - t0
    logger.info(f"training done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()

    # final eval
    with torch.no_grad():
        zp_v, zd_v = model(Xp_va, Xd_va)
        zp_t, zd_t = model(Xp_tr, Xd_tr)
        val_acc = model.retrieval_accuracy(zp_v, zd_v)
        tr_acc = model.retrieval_accuracy(zp_t, zd_t)
    print(f"\n{'='*60}")
    print(f"Pretraining done ({n} genes, {ep} epochs, {elapsed:.0f}s)")
    print(f"  Val R@1:   p2d={val_acc['p2d_acc']:.4f}  d2p={val_acc['d2p_acc']:.4f}")
    print(f"  Train R@1: p2d={tr_acc['p2d_acc']:.4f}  d2p={tr_acc['d2p_acc']:.4f}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    # save
    ckpt = os.path.join(args.output_dir, "crossmodal_clip.pt")
    torch.save({
        "model_state": best_state,
        "config": {"d_prot": d_prot, "d_dna": d_dna, "d_hidden": args.d_hidden,
                   "d_shared": args.d_shared, "n_layers": args.n_layers},
        "norm": {"p_mean": p_mean, "p_std": p_std, "d_mean": d_mean, "d_std": d_std},
        "val_acc": val_acc, "train_acc": tr_acc,
        "gene_names": prot_names,
    }, ckpt)
    logger.info(f"saved {ckpt}")


if __name__ == "__main__":
    main()
