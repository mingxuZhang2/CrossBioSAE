"""
Merge human + multi-species gene embeddings into unified pretraining h5 files.
Output: data/pretrain/protein_all.h5, data/pretrain/dna_all.h5
"""

import h5py
import numpy as np
import os
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pretrain"
os.makedirs(out_dir, exist_ok=True)

sources = {
    "prot": [
        ("data/full/protein_activations.h5", "human"),
        ("data/multispecies/protein_activations_mouse.h5", "mouse"),
        ("data/multispecies/protein_activations_zebrafish.h5", "zebrafish"),
        ("data/multispecies/protein_activations_rat.h5", "rat"),
    ],
    "dna": [
        ("data/full/evo2_activations.h5", "human"),
        ("data/multispecies/evo2_activations_mouse.h5", "mouse"),
        ("data/multispecies/evo2_activations_zebrafish.h5", "zebrafish"),
        ("data/multispecies/evo2_activations_rat.h5", "rat"),
    ],
}

for modality, files in sources.items():
    acts_all, names_all = [], []
    for path, species in files:
        if not os.path.exists(path):
            print(f"MISSING {path} — skipping {species}")
            continue
        with h5py.File(path, "r") as h:
            a = h["activations"][:]
            n = [x.decode() if isinstance(x, bytes) else x for x in h["gene_names"][:]]
        # prefix gene names with species to avoid collisions
        n = [f"{species}:{g}" for g in n]
        acts_all.append(a)
        names_all.extend(n)
        print(f"  {species} {modality}: {a.shape}")

    acts_all = np.concatenate(acts_all, axis=0)
    out = os.path.join(out_dir, f"{'protein' if modality == 'prot' else 'dna'}_all.h5")
    with h5py.File(out, "w") as h:
        h.create_dataset("activations", data=acts_all.astype(np.float32))
        h.create_dataset("gene_names", data=np.array(names_all, dtype="S"))
    print(f"merged {modality}: {acts_all.shape} -> {out}")
