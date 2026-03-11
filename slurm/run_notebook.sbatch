#!/usr/bin/env bash
#SBATCH --partition=gpu-batch
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=notebook-batch
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100,GPU_MEM:80GB
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch slurm/run_notebook.sbatch <notebook.ipynb>

Example:
  sbatch slurm/run_notebook.sbatch notebooks/01_backbones.ipynb

Optional environment overrides:
  CONDA_ENV=vs-gpu2   Conda environment to activate
  NB_KERNEL=python3  Jupyter kernel name
  NB_TIMEOUT=0       Cell timeout in seconds, 0 = no timeout
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
fi
NOTEBOOK_ARG="$1"

if [[ "${NOTEBOOK_ARG}" = /* ]]; then
  NOTEBOOK_PATH="${NOTEBOOK_ARG}"
else
  NOTEBOOK_PATH="${REPO_ROOT}/${NOTEBOOK_ARG}"
fi

if [[ ! -f "${NOTEBOOK_PATH}" ]]; then
  echo "Notebook not found: ${NOTEBOOK_PATH}" >&2
  exit 1
fi

case "${NOTEBOOK_PATH}" in
  "${REPO_ROOT}"/*) ;;
  *)
    echo "Notebook must be inside the repository: ${NOTEBOOK_PATH}" >&2
    exit 1
    ;;
esac

if [[ -f "${HOME}/.bashrc" ]]; then
  # Batch shells often do not inherit the interactive conda setup.
  source "${HOME}/.bashrc"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-vs-gpu2}"

NOTEBOOK_DIR="$(dirname -- "${NOTEBOOK_PATH}")"
NOTEBOOK_NAME="$(basename -- "${NOTEBOOK_PATH}" .ipynb)"
OUTPUT_NOTEBOOK="${NOTEBOOK_DIR}/${NOTEBOOK_NAME}__executed_${SLURM_JOB_ID:-manual}.ipynb"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "Running notebook: ${NOTEBOOK_PATH}"
echo "Writing executed copy to: ${OUTPUT_NOTEBOOK}"
echo "Using conda env: ${CONDA_ENV:-vs-gpu2}"
echo "Using kernel: ${NB_KERNEL:-python3}"
echo "Working directory: ${REPO_ROOT}"
echo "Slurm submit dir: ${SLURM_SUBMIT_DIR:-<unset>}"
echo "Script dir: ${SCRIPT_DIR}"

nvidia-smi || true
python --version
which python

cp "${NOTEBOOK_PATH}" "${OUTPUT_NOTEBOOK}"

jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.kernel_name="${NB_KERNEL:-python3}" \
  --ExecutePreprocessor.timeout="${NB_TIMEOUT:-0}" \
  "${OUTPUT_NOTEBOOK}"

echo "Notebook execution finished: ${OUTPUT_NOTEBOOK}"
