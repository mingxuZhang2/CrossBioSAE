"""Merge 8 Evo2 gene embedding shards into one h5 (same format as protein_activations.h5)."""
import h5py, numpy as np, os, sys

out = sys.argv[1] if len(sys.argv) > 1 else "data/full/evo2_activations.h5"
base = out.replace(".h5", "")
acts, names = [], []
for s in range(8):
    f = f"{base}_shard{s}.h5"
    if not os.path.exists(f):
        print(f"MISSING {f}"); continue
    with h5py.File(f, "r") as h:
        acts.append(h["activations"][:])
        names.extend([n.decode() if isinstance(n, bytes) else n for n in h["gene_names"][:]])
acts = np.concatenate(acts, axis=0)
print(f"merged: {acts.shape} genes, {len(names)} names")
with h5py.File(out, "w") as h:
    h.create_dataset("activations", data=acts)
    h.create_dataset("gene_names", data=np.array(names, dtype="S"))
print(f"saved {out}")
