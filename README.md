# Chess D3PM: Discrete Diffusion Models for Chess Position Generation

A PyTorch implementation of D3PM (Structured Denoising Diffusion Models in Discrete State-Spaces) adapted for unconditional chess position generation. This project uses an absorbing state diffusion process paired with a Diffusion Transformer (DiT) to learn and generate realistic chess positions.

## Overview

This implementation explores discrete diffusion models for chess position generation using two different board representations:

1. **Square-based representation**: Each of the 64 squares is a token (vocabulary size: 14)
2. **Piece-centric representation**: Each of the 32 canonical pieces is a token (vocabulary size: 66)

The model uses an absorbing state diffusion process where pieces gradually transition to an absorbing state during the forward process, and the reverse process learns to reconstruct valid chess positions.

## Architecture

- **Diffusion Engine**: D3PM with absorbing state transitions and cosine noise schedule
- **Backbone Model**: Diffusion Transformer (DiT) with adaptive layer normalization
- **Training Objective**: Cross-entropy loss with optional variational bound term
- **Sampling**: Supports full generation, partial completion, and top-k filtered sampling

## Key Features

- Unconditional chess position generation from pure noise
- Partial board completion (inpainting) for masked positions
- Top-k sampling for improved generation quality
- PyTorch Lightning training framework with W&B logging
- Comprehensive utilities for FEN ↔ tensor conversion and visualization

## Generated Samples

*[Space reserved for generated chess position images]*

## Dataset

The model is trained on a dataset of ~1M chess positions extracted from grandmaster games. The dataset creation pipeline:

1. Downloads chess.com GM games from 2022 via Kaggle
2. Filters for blitz games with standard rules
3. Extracts 3-5 positions per game from moves 8-70
4. Converts positions to tensor format for training

## Installation

```bash
pip install torch pytorch-lightning wandb
pip install python-chess cairosvg pillow
```

For Kaggle dataset download:
```bash
pip install kaggle kagglehub
```

## Quick Start

### Training

```bash
# Generate dataset (or download pre-processed version)
python create_chess_dataset.py

# Train the model
python train.py
```

### Generation

```python
import torch
from train import D3PMLightning

# Load trained model
model = D3PMLightning.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Generate new positions
with torch.no_grad():
    generated_boards = model.sample(num_samples=4)

# Visualize results
from visualize import display_board_from_tensor
display_board_from_tensor(generated_boards[0])
```

### Partial Completion

```python
# Mask part of an existing position
from chess_utils import mask_half, fen_to_tensor

fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
masked_tensor = mask_half(fen, ratio=0.6)

# Complete the position
completed = model.d3pm.sample(
    initial_state=masked_tensor.unsqueeze(0),
    partial_generate=True,
    use_absorbing=True,
    sampling_strategy="top5"
)
```

## Configuration

Key hyperparameters in `train.py`:

```python
# Model architecture
MODEL_DIM = 256      # Hidden dimension
MODEL_DEPTH = 8      # Number of transformer layers
MODEL_HEADS = 8      # Number of attention heads

# Diffusion process
NUM_TIMESTEPS = 16   # Number of diffusion steps

# Training
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
```

## File Structure

```
chess-d3pm/
├── chess_d3pm.py          # Core D3PM diffusion engine
├── dit.py                 # Diffusion Transformer model
├── chess_utils.py         # FEN/tensor conversion utilities
├── create_chess_dataset.py # Dataset generation pipeline
├── train.py               # Training script with Lightning
├── visualize.py           # Board visualization utilities
├── test.py                # Unit tests for sampling logic
└── gm_dataset.ipynb       # Kaggle dataset extraction notebook
```

## Representation Formats

### Square-based (Current)
- 64 tokens representing board squares
- Vocabulary: 12 pieces + empty + absorbing (size 14)
- Direct spatial correspondence to chess board

### Piece-centric (Experimental)
- 32 tokens representing canonical piece instances
- Vocabulary: 64 squares + off-board + absorbing (size 66)
- Each token encodes piece location (1-64) or off-board (0)

## Acknowledgments

This implementation builds upon the D3PM framework. Core diffusion algorithms and mathematical formulations are adapted from the repository by Simo Ryu.
