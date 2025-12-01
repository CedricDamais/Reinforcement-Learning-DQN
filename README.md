# Deep Q-Network (DQN) for Atari Breakout
## Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Model Implementation](#model-implementation)
- [Optimizations](#optimizations)
- [Training Details](#training-details)
- [Evaluation Results](#evaluation-results)
- [Gameplay Video](#gameplay-video)
- [Installation](#installation)
- [Usage](#usage)

## Overview

This project implements a Deep Q-Network agent trained to play Atari Breakout from raw pixel inputs, based on the seminal work by Mnih et al. (2013). The agent learns optimal policies through experience replay and Q-learning, achieving an average score of 187.67 over 30 evaluation episodes.

**Key Features:**
- DQN architecture from 2013 NIPS paper
- Efficient memory-optimized replay buffer (75% memory reduction)
- Experience replay for breaking sample correlation
- Target network for stable Q-learning

## Project Architecture

The project is organized into modular components:

```
├── agent.py              # DQN Agent with training logic
├── model.py              # Neural network architecture
├── environment.py        # Environment wrapper
├── config.py             # Hyperparameters and configuration
├── main.py              # Training script
├── evaluate.py          # Evaluation script
├── play.py              # Interactive gameplay with recording
├── visualize.py         # Result visualization
├── data/
│   ├── efficient_memory.py  # Optimized replay buffer
│   └── replay_memory.py     # Standard replay buffer
└── results/
    └── breakout/
        ├── policy_net.pth          # Trained model weights
        ├── evaluation_results.txt  # Evaluation metrics
        └── videos/                 # Gameplay recordings
```

## Model Implementation

### Network Architecture

We use the **2013 NIPS DQN architecture**, which is more lightweight and faster than the 2015 Nature version:

```python
Input: 4 × 84 × 84 (4 stacked grayscale frames)
    ↓
Conv1: 16 filters, 8×8 kernel, stride 4 → ReLU
    ↓
Conv2: 32 filters, 4×4 kernel, stride 2 → ReLU
    ↓
Flatten: 32 × 9 × 9 = 2,592 features
    ↓
FC1: 256 units → ReLU
    ↓
FC2: 4 units (Q-values for each action)
```

**Why this architecture?**

We chose the 2013 NIPS architecture (677K parameters) rather than the 2015 Nature version (1.7M parameters) as it provides sufficient capacity for Breakout while being more efficient to train.

### Frame Preprocessing

Following the DQN paper, we preprocess frames by converting to grayscale, resizing to 84×84, stacking 4 consecutive frames to capture temporal information, and normalizing pixel values to [0, 1].

## Implementation Details

### Efficient Replay Memory

Standard replay buffers store complete 4-frame stacks, resulting in 4× memory redundancy. Our `EfficientReplayMemory` stores only individual frames and reconstructs stacks on-the-fly, achieving **75% memory reduction** (11.2 GB → 3.5 GB for 500K capacity). This enables larger replay buffers for better sample diversity, a key factor in DQN's success as described in the original paper.

### Experience Replay

Following Mnih et al., we use experience replay to break temporal correlations in the training data. Transitions are stored in a replay buffer and sampled uniformly during training, which stabilizes learning and improves data efficiency.

## Training Details

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Batch Size** | 32 | Balances gradient stability and speed |
| **Learning Rate** | 0.00025 | RMSprop with α=0.95, ε=0.01 |
| **Discount (γ)** | 0.99 | Long-term reward consideration |
| **Replay Memory** | 500K | Large buffer for diversity |
| **ε-greedy** | 1.0 → 0.1 | Linear decay over 1M frames |
| **Target Update** | 10K steps | DQN stability |
| **Learning Frequency** | Every 4 steps | Computational efficiency |
| **Total Frames** | 10M | Sufficient for convergence |

### Training Strategy

We follow the DQN training procedure: an initial exploration phase (50K frames) populates the replay buffer with random experiences, followed by learning with ε-greedy policy (linear decay from 1.0 to 0.1). The agent optimizes the MSE loss between predicted Q-values and targets computed using a separate target network, updated every 10K steps for stability.

## Training Results

### Training Metrics Visualization

![DQN Key Metrics](results/breakout/dqn_key_metrics.png)

The training curves show steady improvement in average reward (~0.2 to 2.0+) and Q-value estimates (~0.04 to 1.5) over 10 million frames, with epsilon linearly decaying from 1.0 to 0.1 and loss stabilizing after the initial learning phase.

### Model Architecture Metrics

| Metric | Value |
|--------|-------|
| **Total Parameters** | 677,172 |
| **Trainable Parameters** | 677,172 |
| **Model Size** | 2.59 MB |
| **Architecture** | 2013 NIPS DQN |
| **Training Time** | ~5-6 hours (single GPU) |

**Layer-wise breakdown:**
- Conv1: 16 filters (8×8, stride 4) → 4,096 params (+ 16 bias)
- Conv2: 32 filters (4×4, stride 2) → 8,192 params (+ 32 bias)
- FC1: 256 units → 663,552 params (+ 256 bias)
- FC2: 4 units (Q-values) → 1,024 params (+ 4 bias)

## Evaluation Results

Our trained DQN agent was evaluated over **30 episodes** with ε=0.05 (5% random exploration):

### Performance Metrics

```
Average Score:  187.67 ± 94.64
Maximum Score:  367.0
Minimum Score:  19.0
Median Score:   184.5
```

### Score Distribution

```
Episode Scores:
[197, 84, 19, 336, 222, 350, 257, 91, 26, 182, 367, 260, 269, 198, 299,
 104, 89, 137, 259, 106, 150, 103, 262, 154, 191, 105, 311, 91, 254, 157]
```

### Q-Value Analysis

To understand the model's decision-making, we analyzed Q-values across 1000 random states:

**Q-Value Statistics:**

| Metric | Value |
|--------|-------|
| **Mean Q-value** | 2.36 |
| **Std Q-value** | 2.24 |
| **Max Q-value** | 5.99 |
| **Min Q-value** | -1.66 |
| **Mean Max-Q** | 2.44 |

**Per-Action Q-Values:**

| Action | Mean | Std | Range |
|--------|------|-----|-------|
| **NOOP** | 2.38 | 2.27 | [-1.59, 5.97] |
| **FIRE** | 2.31 | 2.29 | [-1.66, 5.92] |
| **RIGHT** | 2.34 | 2.15 | [-1.43, 5.81] |
| **LEFT** | 2.43 | 2.24 | [-1.53, 5.99] |

**Action Distribution:** From 1000 random states, the agent prefers LEFT (55.9%), RIGHT (29.3%), and NOOP (14.8%), with FIRE being context-dependent. Q-values are relatively uniform across actions (~2.3-2.4), indicating balanced value estimation with an average Max-Q of 2.44.

### Analysis

The agent achieves consistent performance (average 187.67) well above random baseline (~2-5 points), with a maximum of 367 demonstrating strong learned strategies. The variance (σ=94.64) reflects game stochasticity, exploration (ε=0.05), and the difficulty of Breakout's precise timing requirements.

**Comparison to Baselines:**
- Random agent: ~2-5 points
- Human amateur: ~30-50 points
- Our DQN: **187.67 average** ✓
- Human expert: ~300-400 points

## Gameplay Video

Watch our trained DQN agent play Breakout:

### Video Demonstration

<div align="center">
  <video src="https://github.com/user-attachments/assets/breakout_full_gameplay.mp4" controls width="640">
    Your browser does not support the video tag.
  </video>
</div>

Alternatively, view the video directly: [breakout_full_gameplay.mp4](results/breakout/videos/breakout_full_gameplay.mp4)

*Full gameplay video showing 5 consecutive games. Individual episode videos and the concatenated version are available in `results/breakout/videos/`*

**To generate and combine gameplay videos:**

```bash
# 1. Play and record 5 games
uv run play.py

# 2. Concatenate all episodes into a single video
python concatenate_videos.py

# Videos saved to: results/breakout/videos/
# - Individual episodes: breakout_gameplay-episode-0.mp4 through episode-4.mp4
# - Combined video: breakout_full_gameplay.mp4
```

The `play.py` script records gameplay using Gymnasium's `RecordVideo` wrapper, and `concatenate_videos.py` uses ffmpeg to stitch all episodes together into a single comprehensive video.

### What to observe in the video:

1. **Ball Tracking**: Agent adjusts paddle position to intercept the ball with high accuracy
2. **Strategic Positioning**: Centers paddle for maximum brick coverage and optimal return angles
3. **Recovery**: Handles difficult angles, fast rebounds, and edge cases effectively
4. **Score Progression**: Consistent brick clearing patterns with scores ranging from 100-350+
5. **Consistency**: Demonstrates learned strategy across multiple games with minimal failures

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- Gymnasium (Atari environments)
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/CedricDamais/Reinforcement-Learning-DQN.git
cd Reinforcement-Learning-DQN

# Install dependencies
pip install -r requirements.txt

# Or with uv (recommended)
uv pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train from scratch
python main.py

# Monitor with TensorBoard
tensorboard --logdir=runs
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py

# Results saved to: results/breakout/evaluation_results.txt
```

### Playing

```bash
# Watch agent play (with video recording)
python play.py

# Videos: results/breakout/videos/
```

### Visualization

```bash
# Plot training curves
python visualize.py
```

## Key Takeaways

### What Worked Well

1. **Efficient Memory**: 75% memory reduction enabled larger buffer → better learning
2. **Lightweight Architecture**: 2013 NIPS DQN sufficient for Breakout
3. **Hyperparameter Tuning**: Standard DQN hyperparameters worked well
4. **Frame Stacking**: Temporal information crucial for motion prediction

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Memory constraints | Efficient replay buffer (single-frame storage) |
| Slow training | GPU acceleration, batch operations |
| Exploration vs exploitation | ε-greedy with linear decay |
| Target instability | Periodic target network updates (10K steps) |

### Future Improvements

- **Double DQN**: Reduce Q-value overestimation
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Focus on important transitions
- **Rainbow DQN**: Combine multiple improvements
- **Longer Training**: Extend to 20M+ frames for further improvement

## References

- Mnih et al. (2013). "Playing Atari with Deep Reinforcement Learning" [[arXiv:1312.5602](https://arxiv.org/abs/1312.5602)]
- Mnih et al. (2015). "Human-level control through deep reinforcement learning" [[Nature](https://www.nature.com/articles/nature14236)]

## License

MIT License - see LICENSE file for details.

## Author

Cédric Damais - [GitHub](https://github.com/CedricDamais)

---
