#!/bin/bash
# Launch DDPM Training with 4 GPUs
# Run from /workspace/ directory on RunPod

echo "=========================================="
echo "DDPM Training Launcher"
echo "=========================================="
echo ""

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/workspace"

# Check if VQ-GAN checkpoint exists
VQGAN_CKPT="${WORKSPACE}/checkpoint/vqgan_encoder_decoder.ckpt"
if [ ! -f "$VQGAN_CKPT" ]; then
    echo "ERROR: VQ-GAN checkpoint not found at: $VQGAN_CKPT"
    exit 1
fi
echo "✓ VQ-GAN checkpoint found"

# Check if data directory exists
DATA_DIR="${WORKSPACE}/data/final_patient_data"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found at: $DATA_DIR"
    exit 1
fi
echo "✓ Data directory found"

# Create results directory
RESULTS_DIR="${WORKSPACE}/results/ddpm_dual_drr"
mkdir -p "$RESULTS_DIR"
echo "✓ Results directory: $RESULTS_DIR"

# Create logs directory
LOG_DIR="${WORKSPACE}/logs"
mkdir -p "$LOG_DIR"
echo "✓ Logs directory: $LOG_DIR"

echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Configuration:"
echo "  - GPUs: 4"
echo "  - Batch size per GPU: 1"
echo "  - Gradient accumulation: 4"
echo "  - Effective batch size: 16"
echo "  - Mixed precision: Enabled"
echo "=========================================="
echo ""

# Launch training with torchrun (distributed)
cd "${SCRIPT_DIR}"
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_ddpm_dual_drr.py \
    2>&1 | tee "${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "Training Complete"
echo "=========================================="
