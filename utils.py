import csv
import json
import os
from typing import List

from config import SimulationConfig


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_epoch_metrics(epoch_logs: List[dict], cfg: SimulationConfig) -> None:
    results_dir = os.path.join(cfg.output_dir, "results")
    _ensure_dir(results_dir)
    path = os.path.join(results_dir, "epoch_metrics.csv")

    if not epoch_logs:
        return

    fieldnames = list(epoch_logs[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_logs)

    print(f"Epoch metrics saved to {path}")


def save_interview_logs(interview_logs: List[dict], cfg: SimulationConfig) -> None:
    results_dir = os.path.join(cfg.output_dir, "results")
    _ensure_dir(results_dir)
    path = os.path.join(results_dir, "interview_logs.csv")

    if not interview_logs:
        return

    fieldnames = list(interview_logs[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interview_logs)

    print(f"Interview logs saved to {path}")


def save_final_summary(results: dict, cfg: SimulationConfig) -> None:
    results_dir = os.path.join(cfg.output_dir, "results")
    _ensure_dir(results_dir)
    path = os.path.join(results_dir, "final_summary.json")

    summary = {
        "num_epochs_run": results["num_epochs_run"],
        "converged": results["converged"],
        "convergence_epoch": results["convergence_epoch"],
        "final_average_ai_usage": results["final_average_ai_usage"],
        "final_average_global_trust": results["final_average_global_trust"],
        "final_average_offer": results["final_average_offer"],
        "final_detection_rate": results["final_detection_rate"],
        "final_correct_match_rate": results["final_correct_match_rate"],
        "final_overoffer_rate": results["final_overoffer_rate"],
        "final_underoffer_rate": results["final_underoffer_rate"],
        "final_average_candidate_loss": results["final_average_candidate_loss"],
        "final_average_company_loss": results["final_average_company_loss"],
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Final summary saved to {path}")


def save_results(results: dict, cfg: SimulationConfig) -> None:
    save_epoch_metrics(results["epoch_logs"], cfg)
    save_interview_logs(results["interview_logs"], cfg)
    save_final_summary(results, cfg)
