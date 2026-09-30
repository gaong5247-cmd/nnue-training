# FastNNUE CPU Trainer

A tiny C++17 NNUE-style trainer designed for **CPU-only** training from EPD-like chess data.

Input format:

```text
rnbqkb1r/ppp1pppp/3p1n2/8/3P4/2N1P3/PPP2PPP/R1BQKBNR b KQk - 1 3 | score=-64 | move=b8c6 | ply=5 | result=1
```

The trainer converts text EPD once into a compact fixed-size binary file and then trains from that binary data. The network is intentionally small so it can train quickly on CPUs and be easy to embed in a chess engine later.

## Network

- Input: 768 sparse features
  - own/enemy × 6 piece types × 64 squares
  - the board is mirrored when Black is to move so the side to move always sees itself from the same perspective
- Hidden: 64 by default
- Activation: clipped ReLU `[0, 1]`
- Output: scalar evaluation
- Score target: `tanh(score_cp / 400)` blended with the game result
- Optimizer: Adam with sparse NNUE-style feature accumulation

This is a deliberately lightweight first NNUE. It is not Stockfish's HalfKP architecture yet; the repo is structured so HalfKP/king buckets can be added after the CPU pipeline is stable.

## About “1000 epochs”

For a multi-million-position dataset, 1000 **full passes** over the whole dataset cannot realistically be “instant” on CPU. So this trainer supports two modes:

- FastEpoch mode (default): `--epoch-samples 16384` randomly samples 16,384 positions per epoch. This makes 1000 epochs practical on CPU.
- Strict full epoch: `--epoch-samples 0` uses every training position once per epoch.

## Windows quick start

Requirements: CMake + Visual Studio C++ build tools.

```powershell
.\scripts\run-local.ps1 -Data "D:\datasets\train.epd" -Epochs 1000 -EpochSamples 16384 -Threads 2
```

Outputs:

```text
out/checkpoint.bin   resumable optimizer/model state
out/net.nnue         lightweight exported network
out/metrics.csv      epoch, loss, MAE, throughput
```

## Linux quick start

```bash
./scripts/run-local.sh /path/to/train.epd
```

## GitHub Actions

- `.github/workflows/ci.yml`: compiles and runs a tiny sample training test on a GitHub-hosted runner.
- `.github/workflows/day-train-self-hosted.yml`: manual **Windows self-hosted runner** workflow, capped at about 23 hours, default 2 CPU threads.

The heavy day-long training intentionally stays on your own PC while GitHub Actions orchestrates it.

### Self-hosted setup

1. Repository → **Settings → Actions → Runners → New self-hosted runner**
2. Choose **Windows x64** and follow GitHub's commands.
3. Start the runner with `run.cmd`.
4. Repository → **Actions → NNUE Day Train (Self Hosted) → Run workflow**
5. Set `dataset_path`, e.g. `D:\datasets\train.epd`.

The checkout uses `clean: false`, so `out/checkpoint.bin` can be resumed on the same runner workspace.

## EPD fields used

- FEN: required
- `score=`: required, centipawns from side-to-move perspective
- `result=`: optional, `-1`, `0`, or `1` from side-to-move perspective
- `ply=`: optional metadata
- `move=`: currently ignored by the value trainer; reserved for a later policy head

## Next upgrades

- HalfKP / king-bucket sparse feature transformer
- SIMD int16 inference export
- policy head using `move=`
- WDL/value multi-head target
- engine-side incremental accumulator updates
