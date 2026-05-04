import numpy as np

from config import SimulationConfig
from agents import CompanyAgent


def compute_observed_signal(true_type: int, ai_action: int, cfg: SimulationConfig) -> int:
    return min(true_type + ai_action * cfg.ai_signal_boost, cfg.max_ability_level)


def detect_ai_use(true_type: int, ai_action: int, rng: np.random.Generator, cfg: SimulationConfig) -> bool:
    if ai_action == 0:
        return False
    p_detect = cfg.detection_prob_by_type[true_type]
    return bool(rng.random() < p_detect)


def estimate_candidate_ability(
    observed_signal: int,
    individual_trust: float,
    global_trust: float,
    cfg: SimulationConfig,
) -> float:
    effective_trust = individual_trust * global_trust
    estimated_ability = (
        effective_trust * observed_signal
        + (1.0 - effective_trust) * cfg.prior_mean_ability
    )
    return estimated_ability


def choose_offer(estimated_ability: float, cfg: SimulationConfig) -> int:
    if estimated_ability < cfg.reject_threshold:
        return 0
    elif estimated_ability < cfg.low_offer_threshold:
        return 1
    else:
        return 2


def update_trust_after_interview(
    candidate_id: int,
    company_id: int,
    detected: bool,
    trust_matrix: np.ndarray,
    rng: np.random.Generator,
    cfg: SimulationConfig,
) -> np.ndarray:
    """
    Update individual trust scores after one interview.

    Returns:
        updated trust_matrix (modified in-place, also returned for clarity).
    """
    if detected:
        trust_matrix[candidate_id, company_id] = max(
            cfg.min_trust,
            trust_matrix[candidate_id, company_id] - cfg.direct_trust_penalty,
        )

        other_companies = [c for c in range(cfg.num_companies) if c != company_id]
        spillover_companies = rng.choice(
            other_companies,
            size=cfg.num_spillover_companies,
            replace=False,
        )

        for c in spillover_companies:
            trust_matrix[candidate_id, c] = max(
                cfg.min_trust,
                trust_matrix[candidate_id, c] - cfg.spillover_trust_penalty,
            )
    else:
        trust_matrix[candidate_id, company_id] = min(
            cfg.max_trust,
            trust_matrix[candidate_id, company_id] + cfg.trust_recovery_rate,
        )

    return trust_matrix


def update_company_global_trust(company: CompanyAgent, cfg: SimulationConfig) -> None:
    if company.interview_count == 0:
        return

    detected_rate = company.detected_count / company.interview_count

    if detected_rate == 0:
        company.global_trust = min(
            cfg.max_global_trust,
            company.global_trust + cfg.trust_recovery_rate,
        )
    else:
        company.global_trust = max(
            cfg.min_global_trust,
            company.global_trust - cfg.firm_global_trust_learning_rate * detected_rate,
        )

    company.detected_count = 0
    company.interview_count = 0


def create_candidate_groups(candidate_ids, rng: np.random.Generator, cfg: SimulationConfig):
    shuffled = list(candidate_ids)
    rng.shuffle(shuffled)
    groups = [
        shuffled[i: i + cfg.group_size]
        for i in range(0, cfg.num_candidates, cfg.group_size)
    ]
    return groups
