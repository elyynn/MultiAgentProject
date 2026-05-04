import numpy as np

from config import SimulationConfig


def candidate_loss(
    offer: int,
    detected: bool,
    trust_before: np.ndarray,
    trust_after: np.ndarray,
    cfg: SimulationConfig,
) -> float:
    """
    Compute candidate loss for a single interview.

    Args:
        offer: 0 reject, 1 low offer, 2 high offer.
        detected: whether AI use was detected.
        trust_before: 1D array of candidate's trust scores across all companies before update.
        trust_after: 1D array of candidate's trust scores across all companies after update.
        cfg: SimulationConfig.

    Returns:
        float candidate loss (lower is better for the candidate).
    """
    offer_value_map = {
        0: cfg.reject_value,
        1: cfg.low_offer_value,
        2: cfg.high_offer_value,
    }

    offer_value = offer_value_map[offer]

    detection_penalty = cfg.candidate_detection_penalty * float(detected)

    reputation_loss = float(np.sum(trust_before - trust_after))
    reputation_penalty = cfg.candidate_reputation_penalty_weight * reputation_loss

    loss = -offer_value + detection_penalty + reputation_penalty

    return loss


def company_loss(
    offer: int,
    true_type: int,
    ai_action: int,
    detected: bool,
    cfg: SimulationConfig,
) -> float:
    """
    Compute company loss for a single interview.

    Args:
        offer: 0 reject, 1 low offer, 2 high offer.
        true_type: 0 low, 1 medium, 2 high.
        ai_action: 0 no AI, 1 use AI.
        detected: whether AI use was detected.
        cfg: SimulationConfig.

    Returns:
        float company loss (lower is better for the company).
    """
    mismatch_loss = cfg.company_mismatch_weight * (offer - true_type) ** 2

    ai_deception_loss = (
        cfg.company_ai_deception_weight
        * ai_action
        * float(offer > true_type)
    )

    missed_talent_loss = (
        cfg.company_missed_talent_weight
        * float(offer < true_type)
    )

    detection_cost_loss = (
        cfg.company_detection_cost_weight
        * cfg.company_detection_cost
        * float(detected)
    )

    loss = mismatch_loss + ai_deception_loss + missed_talent_loss + detection_cost_loss

    return loss
