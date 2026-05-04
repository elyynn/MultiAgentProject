import numpy as np

from agents import CandidateAgent
from config import SimulationConfig


def choose_candidate_action(candidate: CandidateAgent, rng: np.random.Generator, epsilon: float) -> int:
    if rng.random() < epsilon:
        return int(rng.choice([0, 1]))

    q_no_ai = candidate.q_values[0]
    q_ai = candidate.q_values[1]

    if q_ai > q_no_ai:
        return 1
    elif q_no_ai > q_ai:
        return 0
    else:
        return int(rng.choice([0, 1]))


def update_candidate_q_value(
    candidate: CandidateAgent,
    action: int,
    candidate_loss_value: float,
    cfg: SimulationConfig,
) -> None:
    utility = -candidate_loss_value
    old_q = candidate.q_values[action]
    new_q = old_q + cfg.candidate_learning_rate * (utility - old_q)
    candidate.q_values[action] = new_q


def decay_epsilon(epsilon: float, cfg: SimulationConfig) -> float:
    return max(cfg.min_epsilon, epsilon * cfg.epsilon_decay)
