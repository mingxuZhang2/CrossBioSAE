#!/usr/bin/env python3
"""
Test the CrossBioSAE pipeline with synthetic data on CPU.
Run this to verify the code works before submitting to HPC.

Usage:
    python scripts/test_pipeline.py

This does NOT require GPU or pre-trained models.
It creates tiny synthetic data and runs all components.
"""

import logging
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def test_model():
    """Test CrossBioSAE model forward pass and loss computation."""
    from src.model import CrossBioSAE, CrossBioSAEConfig, CrossBioSAELoss, count_parameters

    logger.info("Testing model...")

    config = CrossBioSAEConfig(
        dim_dna=128,           # Tiny for testing
        dim_protein=64,
        dim_shared=96,
        expansion_factor=4,    # 384 features
        sparsity_type="topk",
        topk_k=16,
        crossmodal_weight=0.1,
    )

    model = CrossBioSAE(config)
    loss_fn = CrossBioSAELoss(config)

    # Count parameters
    params = count_parameters(model)
    logger.info(f"Parameters: {params}")

    # Test forward pass with both modalities
    batch_size = 8
    dna_acts = torch.randn(batch_size, config.dim_dna)
    protein_acts = torch.randn(batch_size, config.dim_protein)

    output = model(dna_acts=dna_acts, protein_acts=protein_acts)

    assert "recon_dna" in output, "Missing recon_dna"
    assert "recon_protein" in output, "Missing recon_protein"
    assert "features_dna" in output, "Missing features_dna"
    assert "features_protein" in output, "Missing features_protein"
    assert output["recon_dna"].shape == (batch_size, config.dim_dna)
    assert output["recon_protein"].shape == (batch_size, config.dim_protein)
    assert output["features_dna"].shape == (batch_size, config.n_features)
    assert output["features_protein"].shape == (batch_size, config.n_features)

    # Check TopK sparsity
    active_dna = (output["features_dna"] > 0).sum(dim=-1)
    assert (active_dna == config.topk_k).all(), f"TopK should give exactly {config.topk_k} active features"

    # Test loss computation
    losses = loss_fn(
        dna_acts=dna_acts,
        protein_acts=protein_acts,
        model_output=output,
        paired=True,
    )

    assert "total" in losses
    assert "recon_dna" in losses
    assert "recon_protein" in losses
    assert "crossmodal" in losses
    assert losses["total"].requires_grad

    # Test single-modality forward
    output_dna = model(dna_acts=dna_acts)
    assert "recon_dna" in output_dna
    assert "recon_protein" not in output_dna

    output_prot = model(protein_acts=protein_acts)
    assert "recon_protein" in output_prot
    assert "recon_dna" not in output_prot

    # Test backward pass
    losses["total"].backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    logger.info("Model tests PASSED")
    return True


def test_data_pipeline():
    """Test data preparation and dataset classes."""
    from src.data import GenePairDataset, PairedActivationDataset

    logger.info("Testing data pipeline...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic gene pairs
        dataset = GenePairDataset(data_dir=tmpdir, split="test")
        pairs = dataset._generate_synthetic_pairs(
            n_genes=50, max_protein_length=200, max_cds_length=600
        )
        assert len(pairs) == 50

        # Create synthetic activation HDF5 files
        n_genes = 50
        dim_dna = 128
        dim_protein = 64
        gene_names = [f"GENE_{i:05d}" for i in range(n_genes)]

        dna_h5 = Path(tmpdir) / "dna_acts.h5"
        prot_h5 = Path(tmpdir) / "prot_acts.h5"

        with h5py.File(dna_h5, "w") as f:
            f.create_dataset("activations", data=np.random.randn(n_genes, dim_dna).astype(np.float32))
            f.create_dataset("gene_names", data=np.array(gene_names, dtype=h5py.string_dtype()))

        with h5py.File(prot_h5, "w") as f:
            f.create_dataset("activations", data=np.random.randn(n_genes, dim_protein).astype(np.float32))
            f.create_dataset("gene_names", data=np.array(gene_names, dtype=h5py.string_dtype()))

        # Test PairedActivationDataset
        paired_ds = PairedActivationDataset(str(dna_h5), str(prot_h5))
        assert len(paired_ds) == n_genes

        sample = paired_ds[0]
        assert "dna_acts" in sample
        assert "protein_acts" in sample
        assert "gene_name" in sample
        assert sample["dna_acts"].shape == (dim_dna,)
        assert sample["protein_acts"].shape == (dim_protein,)

        logger.info("Data pipeline tests PASSED")
        return True


def test_training_loop():
    """Test training loop for a few steps."""
    from src.model import CrossBioSAEConfig
    from src.data import PairedActivationDataset, create_dataloaders
    from src.trainer import CrossBioSAETrainer

    logger.info("Testing training loop...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic data
        n_genes = 100
        dim_dna = 128
        dim_protein = 64
        gene_names = [f"GENE_{i:05d}" for i in range(n_genes)]

        dna_h5 = Path(tmpdir) / "dna_acts.h5"
        prot_h5 = Path(tmpdir) / "prot_acts.h5"

        with h5py.File(dna_h5, "w") as f:
            f.create_dataset("activations", data=np.random.randn(n_genes, dim_dna).astype(np.float32))
            f.create_dataset("gene_names", data=np.array(gene_names, dtype=h5py.string_dtype()))

        with h5py.File(prot_h5, "w") as f:
            f.create_dataset("activations", data=np.random.randn(n_genes, dim_protein).astype(np.float32))
            f.create_dataset("gene_names", data=np.array(gene_names, dtype=h5py.string_dtype()))

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            str(dna_h5), str(prot_h5),
            batch_size=16, train_ratio=0.8, num_workers=0,
        )

        # Config
        config = CrossBioSAEConfig(
            dim_dna=dim_dna,
            dim_protein=dim_protein,
            dim_shared=96,
            expansion_factor=4,
            sparsity_type="topk",
            topk_k=16,
            crossmodal_weight=0.1,
        )

        # Trainer
        trainer = CrossBioSAETrainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-3,
            max_steps=10,
            max_epochs=100,
            log_interval=5,
            eval_interval=5,
            save_interval=10,
            save_dir=str(Path(tmpdir) / "checkpoints"),
            device="cpu",
            use_wandb=False,
            warmup_steps=2,
            crossmodal_warmup_steps=3,
        )

        # Train for a few steps
        trainer.train()

        assert trainer.global_step == 10
        assert (Path(tmpdir) / "checkpoints" / "checkpoint_final.pt").exists()

        # Test checkpoint loading
        trainer.load_checkpoint(str(Path(tmpdir) / "checkpoints" / "checkpoint_final.pt"))
        assert trainer.global_step == 10

        logger.info("Training loop tests PASSED")
        return True


def test_evaluation():
    """Test evaluation functions."""
    from src.model import CrossBioSAE, CrossBioSAEConfig
    from src.evaluation import compute_feature_statistics, run_pilot_evaluation

    logger.info("Testing evaluation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model
        config = CrossBioSAEConfig(
            dim_dna=128, dim_protein=64, dim_shared=96,
            expansion_factor=4, sparsity_type="topk", topk_k=16,
            crossmodal_weight=0.1,
        )
        model = CrossBioSAE(config)

        # Create synthetic data
        n_genes = 50
        gene_names = [f"GENE_{i:05d}" for i in range(n_genes)]
        dna_h5 = Path(tmpdir) / "dna_acts.h5"
        prot_h5 = Path(tmpdir) / "prot_acts.h5"

        with h5py.File(dna_h5, "w") as f:
            f.create_dataset("activations", data=np.random.randn(n_genes, 128).astype(np.float32))
            f.create_dataset("gene_names", data=np.array(gene_names, dtype=h5py.string_dtype()))

        with h5py.File(prot_h5, "w") as f:
            f.create_dataset("activations", data=np.random.randn(n_genes, 64).astype(np.float32))
            f.create_dataset("gene_names", data=np.array(gene_names, dtype=h5py.string_dtype()))

        # Test feature statistics
        stats = compute_feature_statistics(
            model, str(dna_h5), str(prot_h5), device="cpu"
        )
        assert "n_shared" in stats
        assert "mean_crossmodal_sim" in stats
        assert stats["n_genes"] == n_genes

        # Test pilot evaluation
        results = run_pilot_evaluation(
            model, str(dna_h5), str(prot_h5),
            output_dir=str(Path(tmpdir) / "results"),
            device="cpu",
        )
        assert "feasible" in results
        assert "z_score" in results

        logger.info("Evaluation tests PASSED")
        return True


def test_sparsity_types():
    """Test different sparsity types."""
    from src.model import CrossBioSAE, CrossBioSAEConfig

    logger.info("Testing sparsity types...")

    for sparsity in ["topk", "l1", "jumprelu"]:
        config = CrossBioSAEConfig(
            dim_dna=64, dim_protein=32, dim_shared=48,
            expansion_factor=4, sparsity_type=sparsity,
            topk_k=8, l1_coeff=1e-3,
        )
        model = CrossBioSAE(config)

        dna = torch.randn(4, 64)
        prot = torch.randn(4, 32)

        output = model(dna_acts=dna, protein_acts=prot)
        assert output["features_dna"].shape == (4, config.n_features)

        if sparsity == "topk":
            active = (output["features_dna"] > 0).sum(dim=-1)
            assert (active == 8).all(), f"TopK: expected 8 active, got {active}"

        logger.info(f"  {sparsity}: OK")

    logger.info("Sparsity type tests PASSED")
    return True


def main():
    logger.info("=" * 60)
    logger.info("CrossBioSAE Pipeline Tests")
    logger.info("=" * 60)

    tests = [
        ("Model", test_model),
        ("Data Pipeline", test_data_pipeline),
        ("Training Loop", test_training_loop),
        ("Evaluation", test_evaluation),
        ("Sparsity Types", test_sparsity_types),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"{name} FAILED: {e}", exc_info=True)
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED. Pipeline is ready for HPC submission.")
    else:
        print("Some tests FAILED. Fix issues before submitting to HPC.")
        sys.exit(1)


if __name__ == "__main__":
    main()
