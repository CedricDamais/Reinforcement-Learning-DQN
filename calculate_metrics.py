"""
Calculate additional metrics for the trained DQN model.
This script computes various performance and model analysis metrics.
"""

import torch
import numpy as np
from pathlib import Path
import json

from model import DQN
from environment import make_env
from config import Config
from agent import DQNAgent


def calculate_model_metrics(model_path="results/breakout/policy_net.pth"):
    """Calculate various metrics about the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    policy_net = DQN((4, 84, 84), 4).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    metrics = {}

    # 1. Model Size Metrics
    total_params = sum(p.numel() for p in policy_net.parameters())
    trainable_params = sum(
        p.numel() for p in policy_net.parameters() if p.requires_grad
    )
    model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)

    metrics["model_architecture"] = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "architecture": "2013 NIPS DQN",
    }

    print("=" * 60)
    print("MODEL ARCHITECTURE METRICS")
    print("=" * 60)
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Model File Size:       {model_size_mb:.2f} MB")
    print()

    # 2. Layer-wise Parameter Count
    print("Layer-wise Parameter Breakdown:")
    print("-" * 60)
    layer_params = {}
    for name, param in policy_net.named_parameters():
        params = param.numel()
        layer_params[name] = params
        print(f"  {name:<20} {params:>12,} params")
    print()

    metrics["layer_parameters"] = layer_params

    # 3. Q-Value Statistics on Random States
    print("=" * 60)
    print("Q-VALUE STATISTICS (1000 random states)")
    print("=" * 60)

    env = make_env(Config.ENV_NAME, render_mode=None)

    q_values_list = []
    max_q_values = []
    action_distribution = {i: 0 for i in range(4)}

    for _ in range(1000):
        state, _ = env.reset()
        state_tensor = torch.tensor(state, device=device).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            q_values = policy_net(state_tensor).cpu().numpy()[0]
            q_values_list.append(q_values)
            max_q_values.append(q_values.max())
            best_action = q_values.argmax()
            action_distribution[best_action] += 1

    env.close()

    q_values_array = np.array(q_values_list)

    metrics["q_value_statistics"] = {
        "mean_q_value": float(np.mean(q_values_array)),
        "std_q_value": float(np.std(q_values_array)),
        "max_q_value": float(np.max(q_values_array)),
        "min_q_value": float(np.min(q_values_array)),
        "mean_max_q": float(np.mean(max_q_values)),
        "action_distribution": action_distribution,
    }

    print(f"Mean Q-value:          {np.mean(q_values_array):.4f}")
    print(f"Std Q-value:           {np.std(q_values_array):.4f}")
    print(f"Max Q-value:           {np.max(q_values_array):.4f}")
    print(f"Min Q-value:           {np.min(q_values_array):.4f}")
    print(f"Mean Max-Q:            {np.mean(max_q_values):.4f}")
    print()
    print("Action Distribution (from 1000 random states):")
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]
    for action_id, count in action_distribution.items():
        percentage = (count / 1000) * 100
        print(f"  {action_names[action_id]:<10} {count:>4} ({percentage:>5.1f}%)")
    print()

    # 4. Per-Action Q-Value Analysis
    print("=" * 60)
    print("PER-ACTION Q-VALUE ANALYSIS")
    print("=" * 60)

    per_action_stats = {}
    for action_idx in range(4):
        action_q_values = q_values_array[:, action_idx]
        per_action_stats[action_names[action_idx]] = {
            "mean": float(np.mean(action_q_values)),
            "std": float(np.std(action_q_values)),
            "max": float(np.max(action_q_values)),
            "min": float(np.min(action_q_values)),
        }
        print(
            f"{action_names[action_idx]:<10} Mean: {np.mean(action_q_values):>7.4f}  "
            f"Std: {np.std(action_q_values):>7.4f}  "
            f"Range: [{np.min(action_q_values):>7.4f}, {np.max(action_q_values):>7.4f}]"
        )

    metrics["per_action_statistics"] = per_action_stats
    print()

    # 5. Weight Statistics
    print("=" * 60)
    print("WEIGHT STATISTICS")
    print("=" * 60)

    weight_stats = {}
    for name, param in policy_net.named_parameters():
        if "weight" in name:
            weights = param.detach().cpu().numpy()
            weight_stats[name] = {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "max": float(np.max(weights)),
                "min": float(np.min(weights)),
            }
            print(
                f"{name:<25} Mean: {np.mean(weights):>8.5f}  Std: {np.std(weights):>8.5f}"
            )

    metrics["weight_statistics"] = weight_stats
    print()

    # Save metrics to JSON
    output_path = Path(model_path).parent / "additional_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 60)
    print(f"✓ Metrics saved to: {output_path}")
    print("=" * 60)

    return metrics


def analyze_value_estimation():
    """Analyze how well the model estimates values across different game states."""
    print("\n" + "=" * 60)
    print("VALUE ESTIMATION ANALYSIS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "results/breakout/policy_net.pth"

    policy_net = DQN((4, 84, 84), 4).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    env = make_env(Config.ENV_NAME, render_mode=None)

    # Collect Q-values throughout a game
    state, _ = env.reset()
    game_q_values = []
    game_actions = []
    game_rewards = []

    done = False
    step = 0
    while not done and step < 1000:
        state_tensor = torch.tensor(state, device=device).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            q_values = policy_net(state_tensor).cpu().numpy()[0]
            action = q_values.argmax()

        game_q_values.append(q_values)
        game_actions.append(action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        game_rewards.append(reward)
        state = next_state
        step += 1

    env.close()

    game_q_values = np.array(game_q_values)
    max_q_per_step = game_q_values.max(axis=1)

    print(f"Game Length:           {step} steps")
    print(f"Total Reward:          {sum(game_rewards):.0f}")
    print(f"Initial Max-Q:         {max_q_per_step[0]:.4f}")
    print(f"Final Max-Q:           {max_q_per_step[-1]:.4f}")
    print(f"Average Max-Q:         {np.mean(max_q_per_step):.4f}")
    print(f"Max-Q Std Dev:         {np.std(max_q_per_step):.4f}")
    print()

    # Action distribution in this game
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]
    print("Action Distribution in Sample Game:")
    for i, name in enumerate(action_names):
        count = game_actions.count(i)
        percentage = (count / len(game_actions)) * 100
        print(f"  {name:<10} {count:>4} ({percentage:>5.1f}%)")
    print()


if __name__ == "__main__":
    metrics = calculate_model_metrics()
    analyze_value_estimation()
    print("\n✓ Analysis complete!")
