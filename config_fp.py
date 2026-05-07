from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FPConfig:
    seed: int = 42

    # Fictitious play
    num_fp_iterations: int = 2000
    fp_stability_window: int = 200
    fp_stability_tol: float = 0.002
    tie_tol: float = 1e-8

    # Types
    types: List[int] = field(default_factory=lambda: [0, 1, 2])
    type_prior: Dict[int, float] = field(default_factory=lambda: {
        0: 0.30,
        1: 0.50,
        2: 0.20,
    })

    # Candidate actions: 0=No AI, 1=Use AI
    candidate_actions: List[int] = field(default_factory=lambda: [0, 1])

    # Firm policies
    firm_actions: List[str] = field(default_factory=lambda: [
        "LowVerify",
        "BaseVerify",
        "HighVerify",
    ])

    detection_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "LowVerify": 0.50,
        "BaseVerify": 1.00,
        "HighVerify": 2.00,
    })

    verification_cost: Dict[str, float] = field(default_factory=lambda: {
        "LowVerify": 0.02,
        "BaseVerify": 0.08,
        "HighVerify": 0.20,
    })

    false_positive_rate: Dict[str, float] = field(default_factory=lambda: {
        "LowVerify": 0.005,
        "BaseVerify": 0.010,
        "HighVerify": 0.020,
    })

    # Signal model: s ~ N(theta + Delta_theta * a, sigma_s^2)
    signal_sigma: float = 0.25
    ai_signal_boost: Dict[int, float] = field(default_factory=lambda: {
        0: 0.70,
        1: 0.50,
        2: 0.20,
    })

    # Detection: P(d=1 | theta, a=1, m) = min(1, q_m * pi_theta)
    base_detection_prob: Dict[int, float] = field(default_factory=lambda: {
        0: 0.35,
        1: 0.20,
        2: 0.10,
    })

    # Candidate utility components
    offer_value: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0,
        1: 1.0,
        2: 2.0,
    })

    ai_effort_benefit: Dict[int, float] = field(default_factory=lambda: {
        0: 0.10,
        1: 0.08,
        2: 0.05,
    })

    detection_penalty: float = 1.0
    reputation_penalty_weight: float = 1.0
    direct_reputation_penalty: float = 0.60
    spillover_reputation_penalty: float = 0.30
    expected_spillover_count: int = 2

    # Payoff integration
    num_payoff_samples: int = 2000
    common_random_numbers: bool = True

    # Held-out evaluation
    num_eval_samples: int = 100_000
    num_eval_seeds: int = 30

    # Output
    output_dir: str = "outputs_fp"

    def reputation_damage(self) -> float:
        return (
            self.direct_reputation_penalty
            + self.expected_spillover_count * self.spillover_reputation_penalty
        )
