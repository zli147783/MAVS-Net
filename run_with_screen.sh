#!/bin/bash

# MAVS-Net Training with Screen
# This script starts training in a screen session for persistent background execution

SESSION_NAME="mavs_net_train"

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo "Error: screen is not installed. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install screen"
    echo "  CentOS/RHEL: sudo yum install screen"
    echo "  macOS: brew install screen"
    exit 1
fi

# Check if session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Screen session '$SESSION_NAME' already exists."
    echo "To attach: screen -r $SESSION_NAME"
    echo "To kill existing session: screen -S $SESSION_NAME -X quit"
    exit 1
fi

echo "Starting MAVS-Net training in screen session '$SESSION_NAME'..."
echo "To detach: Ctrl+A, then D"
echo "To reattach: screen -r $SESSION_NAME"
echo "To kill session: screen -S $SESSION_NAME -X quit"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start screen session with training command
if [ "$1" = "multi" ]; then
    # Multi-GPU training
    screen -S $SESSION_NAME -d -m bash -c "
        echo 'Starting multi-GPU training...'
        CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu.py >./logs/mavs_net_multi_$(date +%Y%m%d_%H%M%S).txt 2>&1
        echo 'Training completed.'
    "
    echo "Multi-GPU training started in screen session '$SESSION_NAME'"
else
    # Single-GPU training (default)
    screen -S $SESSION_NAME -d -m bash -c "
        echo 'Starting single-GPU training...'
        python -u train_single_gpu.py >./logs/mavs_net_single_$(date +%Y%m%d_%H%M%S).txt 2>&1
        echo 'Training completed.'
    "
    echo "Single-GPU training started in screen session '$SESSION_NAME'"
fi

echo ""
echo "Session management commands:"
echo "  Attach: screen -r $SESSION_NAME"
echo "  Detach: Ctrl+A, then D (when attached)"
echo "  Kill: screen -S $SESSION_NAME -X quit"
echo "  List all: screen -list"