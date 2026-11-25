#!/bin/bash
#SBATCH -A c3se2026-1-12
#SBATCH -J timegan
#SBATCH -t 40:00:00
#SBATCH --gpus-per-node=H100:1
#SBATCH -N 1
#SBATCH --output=output/timegan_%j.log
#SBATCH --error=output/timegan_%j.err

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

cd /cephyr/users/soniyaa/Vera/timegan_new/apptainer_run/

echo "=== STARTING TRAINING ==="
apptainer exec --nv timegan.sif python main_timegan.py

if [ $? -eq 0 ]; then
    echo "=== TRAINING SUCCESSFUL - STARTING EVALUATION ==="
    apptainer exec --nv timegan.sif python evaluate.py
    echo "=== EVALUATION COMPLETE ==="
else
    echo "=== TRAINING FAILED - SKIPPING EVALUATION ==="
fi

echo "Job finished: $(date)"
