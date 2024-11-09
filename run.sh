#!/bin/bash

# Path to your virtual environment
VENV_PATH="./venv"

# Check if a device argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <device>"
  exit 1
fi

DEVICE=$1

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run the Python script with the specified device and customized output path
python main.py \
  --gens_per_prompt 2 \
  --input all_variations_${DEVICE}.json \
  --output "./output-${DEVICE}" \
  --model musicgen \
  --batch_size 15 \
  --device "$DEVICE" \
  --duration 10
