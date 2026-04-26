#!/usr/bin/env bash
# =============================================================================
# setup_datagen_env.sh
# Creates an isolated conda environment for vLLM-based data generation.
# Does NOT touch the main training environment (base/respailab).
#
# Usage (on server):
#   bash scripts/setup_datagen_env.sh
#
# After setup:
#   conda activate gnosis_datagen
#   python src/data_preprocess/data_generation.py ...
#
# Environment spec:
#   - Python 3.11 (safest for vLLM; avoids 3.12 edge cases)
#   - vLLM latest stable (pip install vllm)
#   - pandas, datasets, rich, transformers (HF, not local)
#   - NO local transformers branch (data generation uses standard HF)
# =============================================================================

set -euo pipefail

ENV_NAME="gnosis_datagen"
PYTHON_VERSION="3.11"

echo "============================================================"
echo " Gnosis DataGen — Isolated Environment Setup"
echo " Env name  : $ENV_NAME"
echo " Python    : $PYTHON_VERSION"
echo " CUDA      : $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'unknown')"
echo "============================================================"

# ── 1. Create conda env ───────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "[1/5] Env '$ENV_NAME' already exists — skipping creation."
    echo "      To recreate: conda env remove -n $ENV_NAME"
else
    echo "[1/5] Creating conda env: $ENV_NAME (Python $PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# ── 2. Install vLLM (brings its own torch compatible with CUDA 12.8) ──────────
echo "[2/5] Installing vLLM (isolated; will install its own torch)..."
conda run -n "$ENV_NAME" pip install vllm --upgrade

# ── 3. Install data pipeline dependencies ─────────────────────────────────────
echo "[3/5] Installing data generation dependencies..."
conda run -n "$ENV_NAME" pip install \
    datasets \
    pandas \
    pyarrow \
    rich \
    huggingface_hub \
    transformers \
    accelerate

# ── 4. Verify installation ────────────────────────────────────────────────────
echo "[4/5] Verifying installation..."
conda run -n "$ENV_NAME" python -c "
import vllm, torch, datasets, pandas, rich
print(f'  vLLM      : {vllm.__version__}')
print(f'  torch     : {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
print(f'  GPU count : {torch.cuda.device_count()}')
print(f'  datasets  : {datasets.__version__}')
print(f'  pandas    : {pandas.__version__}')
"

# ── 5. Print usage ────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Setup complete. To generate data:"
echo ""
echo "  conda activate $ENV_NAME"
echo "  cd /mnt/vaultb/Aditya_Manik/Gnosis"
echo ""
echo "  # TriviaQA:"
echo "  python src/data_preprocess/data_generation.py \\"
echo "    --data_mode hf --dataset_id mandarjoshi/trivia_qa \\"
echo "    --dataset_config rc --dataset_split train \\"
echo '    --system_prompt "This is a trivia question. Put your final answer within \boxed{}." \\'
echo "    --model_id Qwen/Qwen3-8B \\"
echo "    --save_dir data/train/Qwen3_8B_trivia_qa40k-6k"
echo ""
echo "  # DAPO-Math:"
echo "  python src/data_preprocess/data_generation.py \\"
echo "    --data_mode hf --dataset_id open-r1/DAPO-Math-17k-Processed \\"
echo "    --dataset_config en --dataset_split train \\"
echo '    --system_prompt "Please reason step by step, and put your final answer within \boxed{}." \\'
echo "    --model_id Qwen/Qwen3-8B \\"
echo "    --save_dir data/train/Qwen3_8B_DAPO_Math_9k3k_2gen"
echo ""
echo "  # After generation, deactivate and label in the MAIN env:"
echo "  conda deactivate"
echo "  python src/data_preprocess/Label_for_SFT.py ..."
echo "============================================================"
