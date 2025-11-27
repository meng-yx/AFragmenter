#!/bin/bash
#SBATCH --job-name=afdb_to_domain
#SBATCH --output=/scratch/%u/logs/afdb_to_domain_%A_%a.out
#SBATCH --error=/scratch/%u/logs/afdb_to_domain_%A_%a.out
#SBATCH --time=00:20:00
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G

# Usage:
#   sbatch slurm/afdb_to_domain.sh --array 0-499 /path/to/subset_dir /path/to/domainome_root /path/to/out_dir
#
# subset_dir should contain files like subset_0.csv, subset_1.csv, ...

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 /path/to/subset_dir /path/to/domainome_root /path/to/out_dir"
    exit 1
fi

SUBSET_DIR="$1"        # dir containing input subset csv files
DOMAINOME_DIR="$2"     # dir to save domainome .pdb and domains .csv files
OUT_DIR="$3"           # dir to save subset out csv files (for tracking status of each subset)

# Derive the subset file from the array index
IDX="${SLURM_ARRAY_TASK_ID:-0}"
SUBSET_FILE="${SUBSET_DIR}/subset_${IDX}.csv"

if [ ! -f "$SUBSET_FILE" ]; then
    echo "Subset file not found: $SUBSET_FILE"
    exit 1
fi

echo "SLURM_ARRAY_TASK_ID=${IDX}"
echo "Using subset file: $SUBSET_FILE"
echo "Domainome root: $DOMAINOME_DIR"
echo "Output directory: $OUT_DIR"

# Activate environment (temporarily disable -u for conda's shell hooks)
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MaSIF
set -u

# Run the AFDB-to-domain pipeline
cd /scratch/ymeng/splitdomain/AFragmenter
mkdir -p "$DOMAINOME_DIR"
mkdir -p "$OUT_DIR"

SUBSET_BASENAME="$(basename "$SUBSET_FILE")"
SUBSET_BASE="${SUBSET_BASENAME%.csv}"
OUT_CSV="${OUT_DIR}/${SUBSET_BASE}_out.csv"

python ./afdb_to_domain.py \
    --input_csv "$SUBSET_FILE" \
    --output_csv "$OUT_CSV" \
    --out_root "$DOMAINOME_DIR"