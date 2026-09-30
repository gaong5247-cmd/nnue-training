#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-data/train.epd}"
EPOCHS="${EPOCHS:-1000}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-16384}"
THREADS="${THREADS:-2}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2

./build/nnue_train preprocess --input "$DATA" --output data/train.nnueb
./build/nnue_train train --data data/train.nnueb --checkpoint out/checkpoint.bin --export out/net.nnue --metrics out/metrics.csv --epochs "$EPOCHS" --epoch-samples "$EPOCH_SAMPLES" --threads "$THREADS" --minutes 330
