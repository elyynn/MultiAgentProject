import argparse

from config import get_default_config
from simulation import run_simulation
from visualization import generate_all_plots
from utils import save_results


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-Assisted Hiring Market Simulation")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--initial_ai_rate", type=float, default=None)
    parser.add_argument("--early_stop", action="store_true", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_default_config()

    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.initial_ai_rate is not None:
        cfg.initial_ai_rate = args.initial_ai_rate
    if args.early_stop:
        cfg.early_stop = True
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    print(f"Running simulation: {cfg.num_epochs} epochs, seed={cfg.seed}")
    results = run_simulation(cfg)

    print(f"Epochs completed: {results['num_epochs_run']}")
    print(f"Converged: {results['converged']} (epoch {results['convergence_epoch']})")
    print(f"Final AI usage: {results['final_average_ai_usage']:.3f}")
    print(f"Final global trust: {results['final_average_global_trust']:.3f}")
    print(f"Final correct match rate: {results['final_correct_match_rate']:.3f}")

    if cfg.save_results:
        save_results(results, cfg)

    generate_all_plots(results, cfg)


if __name__ == "__main__":
    main()
