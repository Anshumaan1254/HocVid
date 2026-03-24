# EvoIR Stage 1 — AFLB + EOS + Frame Cache

Implementation of Stage 1 from the [EvoIR paper](https://arxiv.org/abs/2512.05104) for video super-resolution.

## Architecture

```
Degraded Frame → Frame Cache Check (64×64 cosine similarity)
    ├── Cache Hit  → Return cached result (skip processing)
    └── Cache Miss → AFLB → Restored Frame → Store in cache
```

**AFLB**: Adaptive Frequency Learning Block
- **FD**: Learnable low-pass filter → low/high frequency decomposition
- **FMgM**: Spectral gating (low-freq via FFT) + spatial masking (high-freq via depthwise conv)
- **FMoM**: Bidirectional cross-frequency modulation (SpatialGate + ChannelGate)

**EOS**: Evolutionary Optimization Strategy
- Teacher-Student EMA (momentum=0.999)
- Population-based loss weight search every 500 iterations
- Crossover + mutation on simplex for `(λ_L1, λ_SSIM)`

## Project Structure

```
evoIR_aflb/
├── models/
│   ├── aflb.ipynb           # AFLB module (FD + FMgM + FMoM)
│   ├── eos.ipynb            # EOS evolutionary optimizer + EMA Teacher
│   ├── frame_cache.ipynb    # Frame similarity caching (LRU)
│   └── res_fftb.ipynb       # RES-FFTB blocks + AdaIR encoder-decoder
├── utils/
│   └── load_hf_model.ipynb  # Load pretrained from HuggingFace
├── train.ipynb              # Training with EOS triggers + CUDA AMP
├── test.ipynb               # Inference demo with frame cache
└── README.md
```

## Requirements

```bash
pip install torch torchvision einops torchmetrics matplotlib huggingface_hub
```

**Hardware**: NVIDIA GPU with CUDA support (optimized for RTX 5070)

## Quick Start

### Training (synthetic demo)
1. Open `train.ipynb` in Jupyter
2. Run all cells — will train on synthetic data with small model for demo
3. For full training: set `cfg.degraded_dir` and `cfg.clean_dir` to your data

### Inference
1. Open `test.ipynb`
2. Run all cells — processes synthetic video frames with frame cache
3. To load pretrained: first run `utils/load_hf_model.ipynb`

### Exploring Components
- `models/aflb.ipynb` — Detailed AFLB architecture with diagrams and tests
- `models/eos.ipynb` — EOS with population evolution visualization
- `models/frame_cache.ipynb` — Cache simulation with video sequences
- `models/res_fftb.ipynb` — Full encoder-decoder verification

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 48 | Base feature dimension |
| `num_blocks` | [4,6,6,8] | Transformer blocks per level |
| `num_fft_blocks` | [4,2,2,1] | FFT blocks per level |
| `eos_interval` | 500 | Steps between weight evolution |
| `ema_momentum` | 0.999 | Teacher EMA decay |
| `cache_threshold` | 0.95 | Cosine similarity for cache hit |

## References

- **Paper**: [EvoIR: Evolutionary Frequency Modulation for All-in-One Image Restoration](https://arxiv.org/abs/2512.05104)
- **GitHub**: [leonmakise/EvoIR](https://github.com/leonmakise/EvoIR)
- **HuggingFace**: [leonmakise/EvoIR](https://huggingface.co/leonmakise/EvoIR)
- **Restormer**: Backbone architecture reference
