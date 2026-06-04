#!/usr/bin/env python3
"""
Train SAE on v6 model's internal representations for concept discovery.

Pipeline:
1. Pretrain v6 projections (1280→768, 4096→768) on 260k variants
2. Extract 1536-d cross-modal representations
3. Train TopK SAE (1536 → K*1536 → 1536)
4. Concept analysis with rich annotations (GO, gene constraint, AA properties)
"""

import argparse, glob, gzip, json, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── v6 projection layers (replicated from train_variant_mlp_v6.py) ────

class ProjectionLayers(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_proj=768, dropout=0.2):
        super().__init__()
        self.proj_prot = nn.Sequential(
            nn.Linear(d_prot, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        self.proj_dna = nn.Sequential(
            nn.Linear(d_dna, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))

    def forward(self, xp, xd):
        return torch.cat([self.proj_prot(xp), self.proj_dna(xd)], dim=-1)


class PretrainHead(nn.Module):
    def __init__(self, d_in=1536, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


# ── TopK SAE ──────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, d_input, n_features, k=32):
        super().__init__()
        self.encoder = nn.Linear(d_input, n_features)
        self.decoder = nn.Linear(n_features, d_input, bias=True)
        self.k = k
        self.n_features = n_features
        nn.init.kaiming_uniform_(self.encoder.weight)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def encode(self, x):
        z = self.encoder(x)
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return sparse

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_sae(sae, data, device, epochs=50, batch=2048, lr=1e-3):
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = len(data)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0; nb = 0
        for b in range(0, n, batch):
            idx = perm[b:b+batch].numpy()
            x = torch.tensor(data[idx], dtype=torch.float32, device=device)
            x_hat, z = sae(x)
            recon = ((x - x_hat) ** 2).mean()
            l1 = z.abs().mean() * 0.01
            loss = recon + l1
            opt.zero_grad(); loss.backward(); opt.step()
            # Normalize decoder columns
            with torch.no_grad():
                norms = sae.decoder.weight.norm(dim=0, keepdim=True)
                sae.decoder.weight.data /= norms.clamp(min=1e-8)
            total_loss += loss.item(); nb += 1
        if (ep + 1) % 10 == 0:
            alive = ((sae.encode(torch.tensor(data[:5000], dtype=torch.float32, device=device)) > 0).any(0).sum().item())
            print(f"  SAE ep {ep+1}: loss={total_loss/nb:.6f} alive={alive}/{sae.n_features}", flush=True)


# ── AA property tables ────────────────────────────────────────────────

HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
         'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'D':-1,'E':-1,'K':1,'R':1,'H':0.5}
VOLUME = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,
          'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
AROMATIC = set('FWY')
POLAR = set('STNQYHKRDE')
ALIPHATIC = set('AILV')
TINY = set('AGST')
SS_BREAKER = set('PG')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--llr_path", default="results/sae_pretrain_emb/esm2_llr.npz")
    ap.add_argument("--constraint", default="data/full/labels/gnomad_constraint.txt.bgz")
    ap.add_argument("--go_json", default="data/full/go_annotations.json")
    ap.add_argument("--out_dir", default="results/variant_sae_v6")
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--sae_epochs", type=int, default=80)
    ap.add_argument("--pretrain_epochs", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──
    print("Loading data ...", flush=True)
    df = pd.read_csv(args.annotated_csv, low_memory=False)
    dual_idx = np.load(args.dual_idx)
    sub = df.iloc[dual_idx].reset_index(drop=True)
    g2l = {int(g): l for l, g in enumerate(dual_idx)}
    n = len(dual_idx)

    prot = np.zeros((n, 1280), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(args.emb_dir, "esm2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l: prot[g2l[int(gi)]] = row

    dna = np.zeros((n, 4096), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(args.emb_dir, "clinvar_evo2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l: dna[g2l[int(gi)]] = row

    llr_all = np.load(args.llr_path)["llr"].astype(np.float32)
    prot_mean, prot_std = prot.mean(0), prot.std(0) + 1e-8
    dna_mean, dna_std = dna.mean(0), dna.std(0) + 1e-8
    prot_z = (prot - prot_mean) / prot_std
    dna_z = (dna - dna_mean) / dna_std
    llr_mean, llr_std = llr_all.mean(), llr_all.std() + 1e-8
    llr_z = (llr_all - llr_mean) / llr_std
    print(f"  Loaded {n} variants, prot={prot.shape}, dna={dna.shape}", flush=True)

    # Labels
    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_vus = sub["clin_sig"].str.contains("Uncertain", case=False, na=False)
    label_arr = np.full(n, -1)  # -1=unlabeled, 0=benign, 1=pathogenic
    label_arr[is_path.values] = 1
    label_arr[is_ben.values] = 0

    # ── Step 1: Pretrain projections (same as v6) ──
    print("\n[1] Pretraining projections on 260k variants ...", flush=True)
    torch.manual_seed(42); np.random.seed(42)
    proj = ProjectionLayers().to(device)
    pt_head = PretrainHead().to(device)

    params = list(proj.parameters()) + list(pt_head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    yt = torch.tensor(llr_z, dtype=torch.float32)
    spe = max(n // 4096, 1)

    for ep in range(args.pretrain_epochs):
        proj.train(); pt_head.train()
        perm = torch.randperm(n)
        ep_loss = 0; nb = 0
        for b in range(0, n, 4096):
            idx = perm[b:b+4096].numpy()
            xp = torch.tensor(prot_z[idx], dtype=torch.float32, device=device)
            xd = torch.tensor(dna_z[idx], dtype=torch.float32, device=device)
            yb = yt[idx].to(device)
            h = proj(xp, xd)
            loss = loss_fn(pt_head(h), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        if (ep + 1) % 10 == 0:
            print(f"  pretrain ep {ep+1}: loss={ep_loss/nb:.6f}", flush=True)
    del pt_head

    # ── Step 2: Extract representations ──
    print("\n[2] Extracting 1536-d representations ...", flush=True)
    proj.eval()
    reps = np.zeros((n, 1536), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n, 4096):
            xp = torch.tensor(prot_z[b:b+4096], dtype=torch.float32, device=device)
            xd = torch.tensor(dna_z[b:b+4096], dtype=torch.float32, device=device)
            h = proj(xp, xd).cpu().numpy()
            reps[b:b+4096] = h
    print(f"  Representations: {reps.shape}, norm: mean={np.linalg.norm(reps, axis=1).mean():.2f}", flush=True)

    # Also get protein-only and dna-only projections for modality attribution
    prot_proj = np.zeros((n, 768), dtype=np.float32)
    dna_proj = np.zeros((n, 768), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n, 4096):
            xp = torch.tensor(prot_z[b:b+4096], dtype=torch.float32, device=device)
            xd = torch.tensor(dna_z[b:b+4096], dtype=torch.float32, device=device)
            prot_proj[b:b+4096] = proj.proj_prot(xp).cpu().numpy()
            dna_proj[b:b+4096] = proj.proj_dna(xd).cpu().numpy()

    # ── Step 3: Train SAE ──
    d_rep = 1536
    n_features = d_rep * args.expansion
    print(f"\n[3] Training TopK SAE ({d_rep} → {n_features}, k={args.topk}) ...", flush=True)
    sae = TopKSAE(d_rep, n_features, k=args.topk).to(device)
    train_sae(sae, reps, device, epochs=args.sae_epochs)

    # Extract SAE activations for all variants
    print("\n[4] Extracting SAE activations ...", flush=True)
    sae.eval()
    sae_acts = np.zeros((n, n_features), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n, 4096):
            x = torch.tensor(reps[b:b+4096], dtype=torch.float32, device=device)
            z = sae.encode(x).cpu().numpy()
            sae_acts[b:b+4096] = z

    alive_mask = (sae_acts > 0).any(0)
    n_alive = alive_mask.sum()
    print(f"  Alive features: {n_alive}/{n_features}", flush=True)

    # ── Step 4: Modality attribution per feature ──
    print("\n[5] Computing modality attribution ...", flush=True)
    # For each feature, measure how much protein vs DNA projection contributes
    # via encoder weight decomposition
    W = sae.encoder.weight.detach().cpu().numpy()  # (n_features, 1536)
    W_prot = W[:, :768]
    W_dna = W[:, 768:]
    prot_norm = np.linalg.norm(W_prot, axis=1)
    dna_norm = np.linalg.norm(W_dna, axis=1)
    modality_ratio = prot_norm / (prot_norm + dna_norm + 1e-8)
    # 0 = pure DNA, 0.5 = balanced, 1 = pure protein

    modality_labels = np.array(['balanced'] * n_features)
    modality_labels[modality_ratio > 0.65] = 'protein-driven'
    modality_labels[modality_ratio < 0.35] = 'dna-driven'
    modality_labels[(modality_ratio >= 0.35) & (modality_ratio <= 0.65)] = 'cross-modal'

    print(f"  Modality distribution (alive):")
    for m in ['protein-driven', 'dna-driven', 'cross-modal']:
        c = ((modality_labels == m) & alive_mask).sum()
        print(f"    {m}: {c}")

    # ── Step 5: Load annotations ──
    print("\n[6] Loading annotations ...", flush=True)
    genes = sub["gene"].values
    from_aas = sub["from_aa"].values.astype(str)
    to_aas = sub["to_aa"].values.astype(str)
    prot_pos = sub["prot_pos"].values

    # Gene constraint
    gc_map = {}
    if os.path.exists(args.constraint):
        with gzip.open(args.constraint, "rt") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                vals = line.strip().split("\t")
                rec = dict(zip(header, vals))
                try:
                    gc_map[rec["gene"]] = {
                        "pLI": float(rec["pLI"]) if rec["pLI"] not in ("NA","") else 0.5,
                        "oe_mis": float(rec["oe_mis"]) if rec["oe_mis"] not in ("NA","") else 1.0,
                    }
                except: pass

    # GO annotations
    go_data = {}
    if os.path.exists(args.go_json):
        go_data = json.load(open(args.go_json))
    print(f"  GO annotations: {len(go_data)} genes", flush=True)

    # ── Step 6: Deep concept analysis ──
    print(f"\n[7] Deep concept analysis on {n_alive} alive features ...", flush=True)

    concept_records = []
    alive_indices = np.where(alive_mask)[0]

    for fi in alive_indices:
        col = sae_acts[:, fi]
        active = col > 0
        n_act = active.sum()
        if n_act < 10:
            continue

        # Top 10% activating variants
        top_k = max(20, int(n_act * 0.05))
        top_idx = np.argsort(col)[-top_k:]

        t_genes = genes[top_idx]
        t_from = from_aas[top_idx]
        t_to = to_aas[top_idx]
        t_labels = label_arr[top_idx]
        t_pos = prot_pos[top_idx]

        # Basic stats
        labeled_mask = t_labels >= 0
        path_rate = t_labels[labeled_mask].mean() if labeled_mask.sum() > 5 else -1

        gene_counts = Counter(g for g in t_genes if g and g != 'nan')
        from_counts = Counter(f for f in t_from if f and f != 'nan')
        to_counts = Counter(t for t in t_to if t and t != 'nan')
        sub_counts = Counter((f, t) for f, t in zip(t_from, t_to) if f != 'nan' and t != 'nan')

        # Gene family enrichment
        gene_families = defaultdict(int)
        for g in t_genes:
            g = str(g)
            if g.startswith('COL') and len(g) > 3 and g[3:4].isdigit(): gene_families['collagen'] += 1
            elif g.startswith('SCN') or g.startswith('KCN') or g.startswith('CACN'): gene_families['ion_channel'] += 1
            elif g.startswith('SLC'): gene_families['transporter'] += 1
            elif g.startswith('MYH') or g.startswith('TTN') or g in ('MYBPC3','TNNT2','TNNI3','ACTC1','MYL2','MYL3','TPM1'): gene_families['cardiac_sarcomere'] += 1
            elif g.startswith('FBN'): gene_families['fibrillin'] += 1
            elif g.startswith('KRT'): gene_families['keratin'] += 1
            elif 'RAS' in g or g in ('BRAF','KRAS','NRAS','HRAS','RAF1','MAP2K1','MAP2K2','PTPN11','SOS1'): gene_families['RAS_MAPK'] += 1
            elif g.startswith('HB') and len(g) <= 4: gene_families['hemoglobin'] += 1
            elif g.startswith('F') and g[1:].isdigit(): gene_families['coagulation'] += 1

        top_family = max(gene_families.items(), key=lambda x: x[1]) if gene_families else ('other', 0)
        top_family_frac = top_family[1] / len(t_genes) if len(t_genes) > 0 else 0

        # GO enrichment (which GO terms are overrepresented?)
        go_terms_all = []
        for g in t_genes:
            if str(g) in go_data:
                go_terms_all.extend(go_data[str(g)])
        go_counts = Counter(go_terms_all)
        top_go = go_counts.most_common(3)

        # Position clustering (are mutations in a narrow region?)
        valid_pos = [int(p) for p in t_pos if str(p) not in ('nan', 'None', '') and str(p).isdigit()]
        pos_range = max(valid_pos) - min(valid_pos) if len(valid_pos) > 5 else -1
        pos_clustered = pos_range < 100 if pos_range > 0 else False

        # AA property patterns
        n_valid = sum(1 for f in t_from if f in HYDRO)
        if n_valid > 5:
            hydro_changes = [abs(HYDRO.get(t,0) - HYDRO.get(f,0)) for f,t in zip(t_from, t_to) if f in HYDRO and t in HYDRO]
            charge_changes = [abs(CHARGE.get(t,0) - CHARGE.get(f,0)) for f,t in zip(t_from, t_to)]
            vol_changes = [abs(VOLUME.get(t,0) - VOLUME.get(f,0)) for f,t in zip(t_from, t_to) if f in VOLUME and t in VOLUME]
            mean_hydro = np.mean(hydro_changes) if hydro_changes else 0
            mean_charge = np.mean(charge_changes) if charge_changes else 0
            mean_vol = np.mean(vol_changes) if vol_changes else 0
        else:
            mean_hydro = mean_charge = mean_vol = 0

        # Concept assignment (hierarchical, most specific first)
        concept = "unresolved"
        specificity = 0

        # Gene-specific concept
        top_gene = gene_counts.most_common(1)[0] if gene_counts else ('', 0)
        top_gene_frac = top_gene[1] / len(t_genes) if len(t_genes) > 0 else 0
        if top_gene_frac > 0.6 and n_act < 500:
            concept = f"gene-specific:{top_gene[0]}"
            specificity = 5

        # Gene family concept
        if top_family_frac > 0.4 and specificity < 4:
            concept = f"gene-family:{top_family[0]}"
            specificity = 4

        # Position-clustered (domain-like)
        if pos_clustered and top_gene_frac > 0.3 and specificity < 3:
            concept = f"domain-specific:{top_gene[0]}:pos{min(valid_pos)}-{max(valid_pos)}"
            specificity = 3.5

        # Specific AA patterns
        top_sub = sub_counts.most_common(1)[0] if sub_counts else (('',''), 0)
        top_sub_frac = top_sub[1] / max(n_valid, 1)

        gly_from = from_counts.get('G', 0) / max(n_valid, 1)
        cys_from = from_counts.get('C', 0) / max(n_valid, 1)
        cys_to = to_counts.get('C', 0) / max(n_valid, 1)
        pro_to = to_counts.get('P', 0) / max(n_valid, 1)
        gly_to = to_counts.get('G', 0) / max(n_valid, 1)

        if gly_from > 0.5 and top_family[0] == 'collagen' and specificity < 5:
            concept = "mechanism:collagen-Gly-XY-disruption"
            specificity = 5
        elif cys_from > 0.4 and path_rate > 0.7 and specificity < 4:
            concept = "mechanism:disulfide-bond-loss"
            specificity = 4
        elif cys_to > 0.4 and specificity < 4:
            concept = "mechanism:cysteine-gain"
            specificity = 4
        elif pro_to > 0.4 and path_rate > 0.6 and specificity < 3:
            concept = "mechanism:proline-introduction"
            specificity = 3
        elif mean_charge > 0.8 and path_rate > 0.6 and specificity < 2:
            concept = "mechanism:charge-reversal"
            specificity = 2
        elif mean_hydro > 5 and specificity < 2:
            concept = "mechanism:hydrophobicity-switch"
            specificity = 2
        elif gly_from > 0.5 and specificity < 2:
            concept = "mechanism:glycine-loss"
            specificity = 2

        # Pathogenicity-related
        if path_rate > 0.95 and n_act >= 50 and specificity < 1:
            concept = "pathogenic-general"
            specificity = 1
        elif path_rate < 0.1 and n_act >= 50 and specificity < 1:
            concept = "benign-general"
            specificity = 1

        # Constraint-related
        gene_pli = [gc_map.get(str(g), {}).get("pLI", 0.5) for g in t_genes if str(g) in gc_map]
        mean_pli = np.mean(gene_pli) if gene_pli else 0.5
        if mean_pli > 0.9 and specificity < 1.5:
            concept = f"constraint:high-pLI-genes (mean={mean_pli:.2f})"
            specificity = 1.5

        concept_records.append({
            'feature': fi,
            'n_active': int(n_act),
            'path_rate': round(float(path_rate), 3),
            'modality': modality_labels[fi],
            'modality_ratio': round(float(modality_ratio[fi]), 3),
            'concept': concept,
            'specificity': specificity,
            'top_gene': top_gene[0],
            'top_gene_frac': round(top_gene_frac, 3),
            'top_family': top_family[0],
            'top_family_frac': round(top_family_frac, 3),
            'top_sub': f"{top_sub[0][0]}>{top_sub[0][1]}" if top_sub[0][0] else '',
            'top_sub_frac': round(top_sub_frac, 3),
            'gly_from': round(gly_from, 3),
            'cys_from': round(cys_from, 3),
            'cys_to': round(cys_to, 3),
            'pro_to': round(pro_to, 3),
            'mean_hydro_change': round(mean_hydro, 2),
            'mean_charge_change': round(mean_charge, 2),
            'mean_vol_change': round(mean_vol, 1),
            'pos_clustered': pos_clustered,
            'mean_pLI': round(mean_pli, 3),
            'top_go_1': top_go[0][0] if len(top_go) > 0 else '',
            'top_go_2': top_go[1][0] if len(top_go) > 1 else '',
            'top_go_3': top_go[2][0] if len(top_go) > 2 else '',
        })

    cdf = pd.DataFrame(concept_records)
    print(f"\n{'='*90}")
    print(f"RESULTS: {len(cdf)} features analyzed")
    print(f"{'='*90}")

    print(f"\nModality distribution:")
    print(cdf['modality'].value_counts().to_string())

    print(f"\nConcept distribution (top 20):")
    concept_types = cdf['concept'].apply(lambda x: x.split(':')[0] if ':' in x else x)
    print(concept_types.value_counts().head(20).to_string())

    print(f"\nDetailed concept counts:")
    for c, grp in cdf.groupby('concept'):
        if len(grp) >= 3 or grp['specificity'].max() >= 3:
            print(f"  {c:55s} n={len(grp):3d} path={grp['path_rate'].mean():.2f} mod={','.join(grp['modality'].value_counts().index[:2])}")

    # Highlight the most interesting features
    print(f"\n{'='*90}")
    print("TOP DISCOVERY CANDIDATES (high specificity, novel concepts)")
    print("="*90)
    top = cdf[cdf['specificity'] >= 3].sort_values('specificity', ascending=False)
    for _, r in top.head(30).iterrows():
        print(f"  F{r['feature']:5d} [{r['concept']:50s}] mod={r['modality']:14s} n={r['n_active']:5d} path={r['path_rate']:.2f} top={r['top_gene']}({r['top_sub']})")

    # Save everything
    cdf.to_csv(os.path.join(args.out_dir, "sae_concepts.csv"), index=False)
    np.savez_compressed(os.path.join(args.out_dir, "sae_acts.npz"), acts=sae_acts)
    torch.save({
        'state_dict': sae.state_dict(),
        'config': {'d_input': d_rep, 'n_features': n_features, 'k': args.topk},
        'proj_state_dict': proj.state_dict(),
    }, os.path.join(args.out_dir, "sae_model.pt"))
    print(f"\nSaved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
