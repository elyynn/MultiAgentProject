from typing import List

import numpy as np

from config import SimulationConfig


TYPE_NAMES = {0: "low", 1: "medium", 2: "high"}


class MetricsLogger:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.interview_logs: List[dict] = []
        self.epoch_logs: List[dict] = []
        self._current_epoch_interviews: List[dict] = []

    def log_interview(
        self,
        epoch: int,
        round_id: int,
        candidate_id: int,
        company_id: int,
        true_type: int,
        ai_action: int,
        observed_signal: int,
        estimated_ability: float,
        offer: int,
        detected: bool,
        candidate_loss: float,
        company_loss: float,
        individual_trust: float,
        global_trust: float,
    ) -> None:
        record = dict(
            epoch=epoch,
            round_id=round_id,
            candidate_id=candidate_id,
            company_id=company_id,
            true_type=true_type,
            ai_action=ai_action,
            observed_signal=observed_signal,
            estimated_ability=estimated_ability,
            offer=offer,
            detected=int(detected),
            candidate_loss=candidate_loss,
            company_loss=company_loss,
            individual_trust=individual_trust,
            global_trust=global_trust,
        )
        self.interview_logs.append(record)
        self._current_epoch_interviews.append(record)

    def log_epoch(
        self,
        epoch: int,
        candidates,
        companies,
        trust_matrix: np.ndarray,
        epsilon: float,
    ) -> None:
        rows = self._current_epoch_interviews
        self._current_epoch_interviews = []

        if not rows:
            return

        def _mean(key):
            return float(np.mean([r[key] for r in rows]))

        def _mean_by_type(key, t):
            subset = [r[key] for r in rows if r["true_type"] == t]
            return float(np.mean(subset)) if subset else 0.0

        def _rate(condition):
            return float(np.mean([float(condition(r)) for r in rows]))

        def _rate_by_type(condition, t):
            subset = [float(condition(r)) for r in rows if r["true_type"] == t]
            return float(np.mean(subset)) if subset else 0.0

        record = dict(
            epoch=epoch,
            average_ai_usage=_mean("ai_action"),
            average_ai_usage_low=_mean_by_type("ai_action", 0),
            average_ai_usage_medium=_mean_by_type("ai_action", 1),
            average_ai_usage_high=_mean_by_type("ai_action", 2),
            average_offer=_mean("offer"),
            average_offer_low=_mean_by_type("offer", 0),
            average_offer_medium=_mean_by_type("offer", 1),
            average_offer_high=_mean_by_type("offer", 2),
            detection_rate=_mean("detected"),
            detection_rate_low=_mean_by_type("detected", 0),
            detection_rate_medium=_mean_by_type("detected", 1),
            detection_rate_high=_mean_by_type("detected", 2),
            average_individual_trust=float(np.mean(trust_matrix)),
            average_global_trust=float(np.mean([c.global_trust for c in companies])),
            average_candidate_loss=_mean("candidate_loss"),
            average_company_loss=_mean("company_loss"),
            reject_rate=_rate(lambda r: r["offer"] == 0),
            low_offer_rate=_rate(lambda r: r["offer"] == 1),
            high_offer_rate=_rate(lambda r: r["offer"] == 2),
            overoffer_rate=_rate(lambda r: r["offer"] > r["true_type"]),
            underoffer_rate=_rate(lambda r: r["offer"] < r["true_type"]),
            correct_match_rate=_rate(lambda r: r["offer"] == r["true_type"]),
            epsilon=epsilon,
        )
        self.epoch_logs.append(record)

    def has_converged(self) -> bool:
        window = self.cfg.convergence_window
        tol = self.cfg.convergence_tolerance

        if len(self.epoch_logs) < window + 1:
            return False

        recent = self.epoch_logs[-window:]
        keys = ["average_ai_usage", "average_global_trust", "average_offer"]

        for key in keys:
            values = [row[key] for row in recent]
            if max(values) - min(values) > tol:
                return False

        return True

    def to_results(self) -> dict:
        last = self.epoch_logs[-1] if self.epoch_logs else {}
        converged = self.has_converged()

        convergence_epoch = None
        if converged:
            window = self.cfg.convergence_window
            convergence_epoch = int(last.get("epoch", len(self.epoch_logs) - 1))

        return dict(
            epoch_logs=self.epoch_logs,
            interview_logs=self.interview_logs,
            num_epochs_run=len(self.epoch_logs),
            converged=converged,
            convergence_epoch=convergence_epoch,
            final_average_ai_usage=last.get("average_ai_usage"),
            final_average_global_trust=last.get("average_global_trust"),
            final_average_offer=last.get("average_offer"),
            final_detection_rate=last.get("detection_rate"),
            final_correct_match_rate=last.get("correct_match_rate"),
            final_overoffer_rate=last.get("overoffer_rate"),
            final_underoffer_rate=last.get("underoffer_rate"),
            final_average_candidate_loss=last.get("average_candidate_loss"),
            final_average_company_loss=last.get("average_company_loss"),
        )
