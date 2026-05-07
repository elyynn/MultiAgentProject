from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FPConfig:
    """
    Configuration for the Bayesian best-response dynamics with FP-on-(verification, ai_suspicion).

    Note: this is *not* canonical fictitious play in the Brown/Robinson sense — it is
    closer to "best-response dynamics on a finite, stationary stage game". The firm's
    joint action is (verification_level, ai_suspicion); both are part of the FP
    empirical-frequency object that the candidate best-responds to. The offer rule is a
    deterministic function of the joint action plus (s, d), so the stage game is
    stationary across iterations (unlike the prior version where the offer rule moved
    with sigma_c). See `payoffs.firm_offer_belief` for the offer rule.
    """

    # ------------------------------------------------------------------
    # Seeds
    # ------------------------------------------------------------------
    # Base seed; per-training-seed runs are reseeded to base_seed + k, k in 0..num_training_seeds-1.
    seed: int = 42
    num_training_seeds: int = 5

    # ------------------------------------------------------------------
    # Fictitious play
    # ------------------------------------------------------------------
    num_fp_iterations: int = 750
    fp_stability_window: int = 75            # 10% of iterations
    fp_stability_max_delta: float = 0.002    # max single-step empirical-mixture change
    fp_regret_threshold: float = 0.01        # r_max threshold for "converged"

    # Tie-breaking. tie_tol is auto-scaled to MC-SE at runtime (see effective_tie_tol).
    # mc_tie_tol_scale: how many MC standard errors of slack constitute a "tie".
    mc_tie_tol_scale: float = 1.0
    tie_break_prefer_lower: bool = True      # candidate ties go to lower index (No-AI)

    # ------------------------------------------------------------------
    # Fixed-firm mode (used by fp_fixed_firm arm). When set, the firm's joint action
    # is held fixed for all iterations and only candidates run FP.
    # Format: (verification_level, ai_suspicion_index).
    # ------------------------------------------------------------------
    fixed_firm: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Game primitives
    # ------------------------------------------------------------------
    types: List[int] = field(default_factory=lambda: [0, 1, 2])
    type_prior: Dict[int, float] = field(default_factory=lambda: {
        0: 0.30,
        1: 0.50,
        2: 0.20,
    })

    # Candidate actions: 0=No AI, 1=Use AI
    candidate_actions: List[int] = field(default_factory=lambda: [0, 1])

    # Firm verification policies
    firm_actions: List[str] = field(default_factory=lambda: [
        "LowVerify",
        "BaseVerify",
        "HighVerify",
    ])

    # Firm "ai_suspicion" — a committed belief P(a=1) used in the offer rule.
    # The firm's full FP action is (verification_level, ai_suspicion). The offer rule
    # is Bayes-optimal under this *fixed* belief, making the stage game stationary.
    firm_ai_suspicion_levels: List[float] = field(default_factory=lambda: [0.10, 0.50, 0.90])

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

    # ------------------------------------------------------------------
    # Candidate utility components
    # ------------------------------------------------------------------
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
    # NOTE: "reputation" here is a static cost coefficient, NOT a stateful network.
    # The previous bandit version had a per-firm trust matrix and graph-mediated spillovers;
    # those mechanisms are stateful and incompatible with FP's stationary-stage-game
    # assumption, so they were dropped. See reports/methodology_audit_fp.md (I1-FP, Q4).
    reputation_penalty_weight: float = 1.0
    direct_reputation_penalty: float = 0.60
    spillover_reputation_penalty: float = 0.30
    expected_spillover_count: int = 2

    # ------------------------------------------------------------------
    # Payoff integration
    # ------------------------------------------------------------------
    num_payoff_samples: int = 2000
    common_random_numbers: bool = True

    # ------------------------------------------------------------------
    # Held-out evaluation
    # ------------------------------------------------------------------
    num_eval_samples: int = 100_000

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: str = "outputs_fp"

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def reputation_damage(self) -> float:
        """Static reputation cost coefficient (not a network)."""
        return (
            self.direct_reputation_penalty
            + self.expected_spillover_count * self.spillover_reputation_penalty
        )

    def effective_tie_tol(self) -> float:
        """
        Tie tolerance scaled to MC standard error.

        With num_payoff_samples=N and signal_sigma=σ, the dominant source of MC noise in
        per-action expected utility is the offer (~σ-ish scale) ÷ √N. We use σ/√N as the
        unit and scale by mc_tie_tol_scale.
        """
        import math
        return self.mc_tie_tol_scale * self.signal_sigma / math.sqrt(self.num_payoff_samples)

    @property
    def num_firm_joint_actions(self) -> int:
        return len(self.firm_actions) * len(self.firm_ai_suspicion_levels)

    def firm_joint_actions(self):
        """Yield (verification_level, suspicion_index) for the firm's joint action space."""
        for m in self.firm_actions:
            for k in range(len(self.firm_ai_suspicion_levels)):
                yield (m, k)

    def firm_action_label(self, verification: str, suspicion_idx: int) -> str:
        p = self.firm_ai_suspicion_levels[suspicion_idx]
        return f"{verification}|sus={p:.2f}"
