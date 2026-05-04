from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from config import SimulationConfig


@dataclass
class CandidateAgent:
    candidate_id: int
    true_type: int
    uses_ai_initially: bool
    q_values: Dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0})
    current_ai_action: int = 0
    history: List[dict] = field(default_factory=list)


@dataclass
class CompanyAgent:
    company_id: int
    global_trust: float = 1.0
    history: List[dict] = field(default_factory=list)
    detected_count: int = 0
    interview_count: int = 0


def initialize_candidates(cfg: SimulationConfig, rng: np.random.Generator) -> List[CandidateAgent]:
    types = list(cfg.type_distribution.keys())
    probs = list(cfg.type_distribution.values())

    true_types = rng.choice(types, size=cfg.num_candidates, p=probs)
    uses_ai = rng.random(cfg.num_candidates) < cfg.initial_ai_rate

    candidates = []
    for i in range(cfg.num_candidates):
        c = CandidateAgent(
            candidate_id=i,
            true_type=int(true_types[i]),
            uses_ai_initially=bool(uses_ai[i]),
        )
        # Warm-start Q-values based on initial AI preference
        if uses_ai[i]:
            c.q_values[1] = 0.1
        else:
            c.q_values[0] = 0.1
        candidates.append(c)

    return candidates


def initialize_companies(cfg: SimulationConfig) -> List[CompanyAgent]:
    return [
        CompanyAgent(company_id=j, global_trust=cfg.initial_trust)
        for j in range(cfg.num_companies)
    ]


def initialize_trust_matrix(cfg: SimulationConfig) -> np.ndarray:
    return np.full(
        (cfg.num_candidates, cfg.num_companies),
        cfg.initial_trust,
        dtype=float,
    )
