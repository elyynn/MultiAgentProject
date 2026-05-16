from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SimulationConfig:
    # Reproducibility
    seed: int = 42

    # Market size
    num_candidates: int = 100
    num_companies: int = 10
    group_size: int = 10

    # Time structure
    rounds_per_epoch: int = 100
    num_epochs: int = 200

    # Candidate type distribution: 0=Low, 1=Medium, 2=High
    type_distribution: Dict[int, float] = None

    # Initial AI usage
    initial_ai_rate: float = 0.30

    # AI effect
    ai_signal_boost: int = 1
    max_ability_level: int = 2

    # Detection probability by candidate type
    detection_prob_by_type: Dict[int, float] = None

    # Trust parameters
    initial_trust: float = 1.0
    min_trust: float = 0.0
    max_trust: float = 1.0

    # Trust update after AI detection
    direct_trust_penalty: float = 0.50
    spillover_trust_penalty: float = 0.20
    num_spillover_companies: int = 1

    # Optional trust recovery if no detection
    trust_recovery_rate: float = 0.01


    # Firm global trust learning
    firm_global_trust_learning_rate: float = 0.05
    min_global_trust: float = 0.0
    max_global_trust: float = 1.0

    # Candidate learning
    candidate_learning_rate: float = 0.10
    candidate_epsilon: float = 0.10
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.02

    # Offer values
    reject_value: float = 0.0
    low_offer_value: float = 1.0
    high_offer_value: float = 2.0

    # Candidate loss weights
    candidate_detection_penalty: float = 1.0
    candidate_reputation_penalty_weight: float = 1.0

    # Company loss weights
    company_mismatch_weight: float = 1.0
    company_ai_deception_weight: float = 1.0
    company_missed_talent_weight: float = 1.0
    company_detection_cost_weight: float = 0.0
    company_detection_cost: float = 0.0

    # Company decision thresholds
    reject_threshold: float = 0.70
    low_offer_threshold: float = 1.50

    # Prior mean ability
    prior_mean_ability: float = 1.0

    # Convergence
    convergence_window: int = 10
    convergence_tolerance: float = 0.01
    early_stop: bool = False

    # Output
    save_results: bool = True
    output_dir: str = "outputs"


def get_default_config() -> SimulationConfig:
    cfg = SimulationConfig()

    cfg.type_distribution = {
        0: 0.30,  # Low
        1: 0.50,  # Medium
        2: 0.20,  # High
    }

    cfg.detection_prob_by_type = {
        0: 0.35,  # Low-type AI user is easier to detect
        1: 0.20,
        2: 0.10,
    }

    return cfg
