"""Driving-specific orchestration around the generic candidate scorer."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .model import CandidateDecisionModel
from .safety import driving_prior, evaluate_action
from .types import DecisionResult, DrivingState


class DrivingDecisionEngine:
    def __init__(
        self,
        confidence_threshold: float = 0.42,
        switch_margin: float = 0.08,
        model: CandidateDecisionModel | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.switch_margin = switch_margin
        self.model = model or CandidateDecisionModel()
        self._previous_choice: str | None = None

    def decide(
        self,
        state: DrivingState,
        question: str,
        candidates: Sequence[str],
    ) -> DecisionResult:
        checks = [evaluate_action(state, candidate) for candidate in candidates]
        feasible = [allowed for allowed, _ in checks]
        reasons = [reason for _, reason in checks]
        if not any(feasible):
            return DecisionResult(
                choice="request_human_review",
                probabilities={candidate: 0.0 for candidate in candidates},
                feasible=dict(zip(candidates, feasible)),
                reasons=dict(zip(candidates, reasons)),
                confidence=0.0,
                safe=False,
            )

        output = self.model.score(
            state.to_text(),
            question,
            candidates,
            priors=[driving_prior(state, candidate) for candidate in candidates],
            feasible=feasible,
        )
        probabilities = dict(zip(candidates, output.probabilities))
        choice = max(probabilities, key=probabilities.get)
        confidence = probabilities[choice]

        if self._previous_choice in probabilities and feasible[candidates.index(self._previous_choice)]:
            previous_probability = probabilities[self._previous_choice]
            if confidence - previous_probability < self.switch_margin:
                choice = self._previous_choice
                confidence = previous_probability

        if confidence < self.confidence_threshold:
            choice = "request_human_review"
        else:
            self._previous_choice = choice

        return DecisionResult(
            choice=choice,
            probabilities=probabilities,
            feasible=dict(zip(candidates, feasible)),
            reasons=dict(zip(candidates, reasons)),
            confidence=confidence,
            safe=choice != "request_human_review",
        )
