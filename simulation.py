import numpy as np

from config import SimulationConfig
from agents import initialize_candidates, initialize_companies, initialize_trust_matrix
from environment import (
    compute_observed_signal,
    detect_ai_use,
    estimate_candidate_ability,
    choose_offer,
    update_trust_after_interview,
    update_company_global_trust,
    create_candidate_groups,
)
from learning import choose_candidate_action, update_candidate_q_value, decay_epsilon
from losses import candidate_loss, company_loss
from metrics import MetricsLogger


def run_simulation(cfg: SimulationConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)

    candidates = initialize_candidates(cfg, rng)
    companies = initialize_companies(cfg)
    trust_matrix = initialize_trust_matrix(cfg)

    metrics_logger = MetricsLogger(cfg)

    epsilon = cfg.candidate_epsilon

    for epoch in range(cfg.num_epochs):
        groups = create_candidate_groups(
            candidate_ids=range(cfg.num_candidates),
            rng=rng,
            cfg=cfg,
        )

        for round_id in range(cfg.rounds_per_epoch):
            for group_id, group in enumerate(groups):
                company_id = (group_id + round_id) % cfg.num_companies
                company = companies[company_id]

                for candidate_id in group:
                    candidate = candidates[candidate_id]

                    ai_action = choose_candidate_action(candidate, rng, epsilon)

                    observed_signal = compute_observed_signal(
                        candidate.true_type, ai_action, cfg
                    )

                    individual_trust = trust_matrix[candidate_id, company_id]
                    global_trust = company.global_trust

                    estimated_ability = estimate_candidate_ability(
                        observed_signal, individual_trust, global_trust, cfg
                    )

                    offer = choose_offer(estimated_ability, cfg)

                    detected = detect_ai_use(candidate.true_type, ai_action, rng, cfg)

                    trust_before = trust_matrix[candidate_id, :].copy()

                    trust_matrix = update_trust_after_interview(
                        candidate_id, company_id, detected, trust_matrix, rng, cfg
                    )

                    trust_after = trust_matrix[candidate_id, :].copy()

                    c_loss = candidate_loss(offer, detected, trust_before, trust_after, cfg)
                    e_loss = company_loss(offer, candidate.true_type, ai_action, detected, cfg)

                    update_candidate_q_value(candidate, ai_action, c_loss, cfg)

                    company.interview_count += 1
                    if detected:
                        company.detected_count += 1

                    metrics_logger.log_interview(
                        epoch=epoch,
                        round_id=round_id,
                        candidate_id=candidate_id,
                        company_id=company_id,
                        true_type=candidate.true_type,
                        ai_action=ai_action,
                        observed_signal=observed_signal,
                        estimated_ability=estimated_ability,
                        offer=offer,
                        detected=detected,
                        candidate_loss=c_loss,
                        company_loss=e_loss,
                        individual_trust=individual_trust,
                        global_trust=global_trust,
                    )

        for company in companies:
            update_company_global_trust(company, cfg)

        epsilon = decay_epsilon(epsilon, cfg)

        metrics_logger.log_epoch(
            epoch=epoch,
            candidates=candidates,
            companies=companies,
            trust_matrix=trust_matrix,
            epsilon=epsilon,
        )

        if cfg.early_stop and metrics_logger.has_converged():
            print(f"Converged at epoch {epoch}")
            break

    results = metrics_logger.to_results()
    return results
