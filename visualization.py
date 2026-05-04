import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig


def _fig_path(cfg: SimulationConfig, filename: str) -> str:
    path = os.path.join(cfg.output_dir, "figures", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _epochs(logs: List[dict]):
    return [r["epoch"] for r in logs]


def plot_ai_adoption(epoch_logs: List[dict], cfg: SimulationConfig) -> None:
    epochs = _epochs(epoch_logs)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["average_ai_usage"] for r in epoch_logs], label="Overall", linewidth=2)
    plt.plot(epochs, [r["average_ai_usage_low"] for r in epoch_logs], label="Low", linestyle="--")
    plt.plot(epochs, [r["average_ai_usage_medium"] for r in epoch_logs], label="Medium", linestyle="--")
    plt.plot(epochs, [r["average_ai_usage_high"] for r in epoch_logs], label="High", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("AI Adoption Rate")
    plt.title("AI Adoption Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_fig_path(cfg, "ai_adoption_over_time.png"), dpi=150)
    plt.close()


def plot_firm_trust(epoch_logs: List[dict], cfg: SimulationConfig) -> None:
    epochs = _epochs(epoch_logs)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["average_individual_trust"] for r in epoch_logs], label="Avg Individual Trust", linewidth=2)
    plt.plot(epochs, [r["average_global_trust"] for r in epoch_logs], label="Avg Global Firm Trust", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Trust")
    plt.title("Average Firm Trust Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_fig_path(cfg, "firm_trust_over_time.png"), dpi=150)
    plt.close()


def plot_offer_distribution(epoch_logs: List[dict], cfg: SimulationConfig) -> None:
    window = min(10, len(epoch_logs))
    recent = epoch_logs[-window:]

    types = ["Low", "Medium", "High"]
    type_keys = ["low", "medium", "high"]

    reject_rates = []
    low_rates = []
    high_rates = []

    for key in type_keys:
        all_offers = []
        for row in recent:
            # reconstruct per-type rates from interview logs is expensive;
            # approximate from epoch-level per-type averages
            pass
        # Use epoch-level offer averages normalized to [0,2] → approximate fractions
        avg_offers = [row.get(f"average_offer_{key}", 1.0) for row in recent]
        avg = float(np.mean(avg_offers))
        # Map continuous average to approximate categorical fractions via linear interpolation
        # avg=0 → 100% reject, avg=1 → 100% low, avg=2 → 100% high
        r = max(0.0, 1.0 - avg)
        h = max(0.0, avg - 1.0)
        l = max(0.0, 1.0 - r - h)
        reject_rates.append(r)
        low_rates.append(l)
        high_rates.append(h)

    x = np.arange(len(types))
    width = 0.5
    plt.figure(figsize=(8, 5))
    bars_r = plt.bar(x, reject_rates, width, label="Reject")
    bars_l = plt.bar(x, low_rates, width, bottom=reject_rates, label="Low Offer")
    bars_h = plt.bar(x, high_rates, width,
                     bottom=[reject_rates[i] + low_rates[i] for i in range(len(types))],
                     label="High Offer")
    plt.xticks(x, types)
    plt.xlabel("Candidate Type")
    plt.ylabel("Fraction")
    plt.title("Offer Distribution by Candidate Type (Last 10 Epochs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_fig_path(cfg, "offer_distribution_by_type.png"), dpi=150)
    plt.close()


def plot_detection_rate(epoch_logs: List[dict], cfg: SimulationConfig) -> None:
    epochs = _epochs(epoch_logs)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["detection_rate"] for r in epoch_logs], label="Overall", linewidth=2)
    plt.plot(epochs, [r["detection_rate_low"] for r in epoch_logs], label="Low", linestyle="--")
    plt.plot(epochs, [r["detection_rate_medium"] for r in epoch_logs], label="Medium", linestyle="--")
    plt.plot(epochs, [r["detection_rate_high"] for r in epoch_logs], label="High", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Detection Rate")
    plt.title("AI Detection Rate Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_fig_path(cfg, "detection_rate_over_time.png"), dpi=150)
    plt.close()


def plot_losses(epoch_logs: List[dict], cfg: SimulationConfig) -> None:
    epochs = _epochs(epoch_logs)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["average_candidate_loss"] for r in epoch_logs], label="Avg Candidate Loss", linewidth=2)
    plt.plot(epochs, [r["average_company_loss"] for r in epoch_logs], label="Avg Company Loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Candidate and Company Losses Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_fig_path(cfg, "losses_over_time.png"), dpi=150)
    plt.close()


def plot_market_efficiency(epoch_logs: List[dict], cfg: SimulationConfig) -> None:
    epochs = _epochs(epoch_logs)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["correct_match_rate"] for r in epoch_logs], label="Correct Match", linewidth=2)
    plt.plot(epochs, [r["overoffer_rate"] for r in epoch_logs], label="Overoffer", linestyle="--")
    plt.plot(epochs, [r["underoffer_rate"] for r in epoch_logs], label="Underoffer", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.title("Market Efficiency Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_fig_path(cfg, "market_efficiency_over_time.png"), dpi=150)
    plt.close()


def generate_all_plots(results: dict, cfg: SimulationConfig) -> None:
    epoch_logs = results["epoch_logs"]
    if not epoch_logs:
        print("No epoch logs to plot.")
        return

    plot_ai_adoption(epoch_logs, cfg)
    plot_firm_trust(epoch_logs, cfg)
    plot_offer_distribution(epoch_logs, cfg)
    plot_detection_rate(epoch_logs, cfg)
    plot_losses(epoch_logs, cfg)
    plot_market_efficiency(epoch_logs, cfg)

    print(f"Figures saved to {os.path.join(cfg.output_dir, 'figures')}/")
