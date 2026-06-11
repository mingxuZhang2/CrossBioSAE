# Repository Agent Notes

## HPC GPU Environments

- Local/default environment is HKUST-GZ HPC2.
  - Login/current hosts are typically `mgmt-*`, currently observed as `mgmt-4`.
  - Filesystem/project path: `/hpc2hdd/home/mzhang630/data/bioinfo/implementation`.
  - GPU SLURM partition: `i64m1tga800u` (A800 GPUs).
  - Common environment for Evo2 jobs: `module load cuda/12.4` then `conda activate evo2x`.
  - Submit jobs locally with `sbatch scripts/<slurm_script>.sh`; monitor with `squeue -u mzhang630`; logs are under `implementation/logs/`.
  - Repo scripts using this environment include `slurm_clinvar_evo2.sh`, `slurm_clinvar_evo2_emb.sh`, `slurm_finetune_variant.sh`, `slurm_brca1_evo2.sh`, `slurm_gene_evo2.sh`, and VUS/multispecies Evo2 scripts.

- Remote H100 environment is HKUST-GZ HPC3.
  - SSH command:
    `ssh -i /hpc2hdd/home/mzhang630/data/id_rsa -o StrictHostKeyChecking=no mzhang630@hpc3login.hpc.hkust-gz.edu.cn`
  - Verified login host: `ACD-Manage-3`.
  - Filesystem/project path: `/data/user/mzhang630/data/bioinfo/implementation`.
  - GPU SLURM partition/account: `acd_u`, account `d_yings_team` (H100 GPUs).
  - Common environment: `source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh` then `conda activate sake`.
  - Submit jobs after SSH and `cd /data/user/mzhang630/data/bioinfo/implementation` with `sbatch scripts/<slurm_script>.sh`.
  - Monitor with `sinfo -p acd_u` and `squeue -u mzhang630`.
  - Repo scripts using this environment include `slurm_pretrain_hpc3.sh`, `slurm_benchmark.sh`, `slurm_applications.sh`, `slurm_crossmodal_tasks.sh`, `slurm_spvae.sh`, and `slurm_full_run.sh`.

When a script already declares one of these partitions, preserve its cluster-specific paths and conda environment unless the user explicitly asks to migrate it.
