param(
  [string]$Data = "data/train.epd",
  [int]$Epochs = 1000,
  [int]$EpochSamples = 16384,
  [int]$Threads = 2
)

$ErrorActionPreference = "Stop"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 2

$exe = if (Test-Path "build/Release/nnue_train.exe") { "build/Release/nnue_train.exe" } else { "build/nnue_train.exe" }

& $exe preprocess --input $Data --output data/train.nnueb
& $exe train --data data/train.nnueb --checkpoint out/checkpoint.bin --export out/net.nnue --metrics out/metrics.csv --epochs $Epochs --epoch-samples $EpochSamples --threads $Threads --minutes 330
