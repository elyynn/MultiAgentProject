# Project Map — LLM-Assisted Hiring Market Simulation

A multi-agent simulation studying candidate AI use, firm detection, and trust dynamics in a hiring market. Pure-Python, NumPy + Matplotlib only; no ML framework.

## Layout at a glance

```
MultiAgentProject/
├── main.py                 # CLI entrypoint
├── config.py               # SimulationConfig dataclass + defaults
├── simulation.py           # Top-level run loop
├── agents.py               # CandidateAgent / CompanyAgent + initializers
├── environment.py          # Signals, detection, offers, trust updates, grouping
├── learning.py             # Candidate Q-learning + epsilon decay
├── losses.py               # candidate_loss / company_loss
├── metrics.py              # MetricsLogger (per-interview + per-epoch)
├── visualization.py        # Matplotlib plotting (6 figures)
├── utils.py                # CSV/JSON result persistence
├── outputs/
│   ├── figures/            # PNGs (6 plots)
│   └── results/            # epoch_metrics.csv, interview_logs.csv, final_summary.json
├── .claude/agents/         # Project subagents (methodology/results auditors, runner, etc.)
├── .git/, .venv/           # tooling
└── reports/                # This file
```

## Main simulation entrypoints

- **`main.py`** — single CLI entrypoint. Parses args (`--num_epochs`, `--seed`, `--initial_ai_rate`, `--early_stop`, `--output_dir`), calls `get_default_config()`, runs `run_simulation`, prints summary, persists results, generates plots. (`main.py:19`)
- **`simulation.py:run_simulation`** — the actual training/eval loop. Iterates epochs → rounds → groups → candidates; orchestrates env step, learning update, and metrics logging. (`simulation.py:19`)

## Config files

- **`config.py`** — sole config source.
  - `SimulationConfig` dataclass (`config.py:5`) defines all hyperparameters: market size (100 candidates, 10 companies, group size 10), time (10 rounds × 100 epochs), candidate-type distribution, AI signal/detection params, trust dynamics, learning rates, loss weights, offer thresholds, convergence criteria, output dir.
  - `get_default_config()` (`config.py:89`) fills in `type_distribution` (30/50/20 L/M/H) and `detection_prob_by_type` (0.35/0.20/0.10).
- No YAML/TOML; CLI flags override a handful of fields in `main.py`.

## Loss functions

- **`losses.py:candidate_loss`** (`losses.py:6`) — `-offer_value + detection_penalty + reputation_penalty`. Reputation term sums trust drops across all firms.
- **`losses.py:company_loss`** (`losses.py:44`) — sum of mismatch (`(offer - true_type)^2`), AI deception (overoffer × ai_action), missed talent (offer < true_type), and detection cost.
- Used in `simulation.py:70-71` to compute per-interview losses; candidate loss feeds Q-learning via `learning.update_candidate_q_value`.

## Experiment scripts

- **None as standalone scripts.** Experiments are invoked via `main.py` + CLI flags. No sweep runner, no seed-loop wrapper, no batch driver. A single run is the unit of experimentation today. (Worth flagging if multi-seed / sweep results are needed.)

## Output folders

- **`outputs/`** — root output dir, set in `cfg.output_dir` (`config.py:86`).
- **`outputs/results/`** — written by `utils.py`:
  - `epoch_metrics.csv` (per-epoch aggregate metrics; ~30 KB)
  - `interview_logs.csv` (per-interview rows; ~7 MB)
  - `final_summary.json` (final-epoch summary stats)
- **`outputs/figures/`** — written by `visualization.py`:
  - `ai_adoption_over_time.png`
  - `detection_rate_over_time.png`
  - `firm_trust_over_time.png`
  - `losses_over_time.png`
  - `market_efficiency_over_time.png`
  - `offer_distribution_by_type.png`
- All artifacts in `outputs/` are from a single run (no per-seed/per-config subfolders).

## Figure-generation scripts

- **`visualization.py`** — sole plotting module. Six `plot_*` functions (`visualization.py:22-141`) plus `generate_all_plots(results, cfg)` (`visualization.py:144`) called from `main.py:46`. Uses `matplotlib` Agg backend; saves PNGs at 150 DPI to `outputs/figures/`.
- Note: `plot_offer_distribution` (`visualization.py:52`) reconstructs per-type offer distributions by *linearly interpolating* the per-type mean offer rather than reading the actual categorical distribution from `interview_logs` — a known approximation worth flagging to the methodology/results auditors.

## Paper / docs folders

- **None present.** No `paper/`, `manuscript/`, `docs/`, `slides/`, `*.tex`, or `README.md` in the repo. The `.claude/agents/` subagents (paper-writer, slide-writer) are scaffolded but have no target files yet.
- `reports/` (this file) is the only prose artifact.

## Key cross-file flow (one interview)

`simulation.py:42` (per candidate) →
`learning.choose_candidate_action` →
`environment.compute_observed_signal` →
`environment.estimate_candidate_ability` →
`environment.choose_offer` →
`environment.detect_ai_use` →
`environment.update_trust_after_interview` →
`losses.candidate_loss` + `losses.company_loss` →
`learning.update_candidate_q_value` →
`metrics.MetricsLogger.log_interview`.

End of epoch: `environment.update_company_global_trust` for each firm, `learning.decay_epsilon`, `metrics_logger.log_epoch`, optional convergence check.
