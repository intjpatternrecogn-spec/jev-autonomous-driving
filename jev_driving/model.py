"""Dependency-free implementation of the candidate-embedding architecture."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(_dot(vector, vector)) or 1.0
    return [value / norm for value in vector]


@dataclass(frozen=True)
class ScoreOutput:
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    attention: tuple[float, ...]


class HashTextEncoder:
    """Deterministic stand-in for a trainable text encoder.

    It makes the architecture runnable without model downloads. Production
    experiments can replace ``token_embedding`` while keeping the interfaces.
    """

    def __init__(self, dimension: int = 48) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    @staticmethod
    def tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()) or ["empty"]

    def token_embedding(self, token: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            block = hashlib.sha256(f"{token}:{counter}".encode()).digest()
            values.extend((byte / 127.5) - 1.0 for byte in block)
            counter += 1
        return _normalise(values[: self.dimension])

    def encode_tokens(self, text: str) -> list[list[float]]:
        return [self.token_embedding(token) for token in self.tokens(text)]

    def encode(self, text: str) -> list[float]:
        token_vectors = self.encode_tokens(text)
        pooled = [sum(column) / len(token_vectors) for column in zip(*token_vectors)]
        return _normalise(pooled)


class CandidateDecisionModel:
    """Shared state encoding plus query-conditioned parallel option scoring."""

    def __init__(self, dimension: int = 48, temperature: float = 0.22) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.encoder = HashTextEncoder(dimension)
        self.temperature = temperature

    def _decision_embedding(self, state_text: str, question: str) -> tuple[list[float], list[float]]:
        state_memory = self.encoder.encode_tokens(state_text)
        query = self.encoder.encode(question)
        attention = _softmax([_dot(query, token) * 2.0 for token in state_memory])
        context = [
            sum(weight * token[index] for weight, token in zip(attention, state_memory))
            for index in range(self.encoder.dimension)
        ]
        decision = _normalise([q + c for q, c in zip(query, context)])
        return decision, attention

    def score(
        self,
        state_text: str,
        question: str,
        candidates: Sequence[str],
        priors: Sequence[float] | None = None,
        feasible: Sequence[bool] | None = None,
    ) -> ScoreOutput:
        if not candidates:
            raise ValueError("at least one candidate is required")
        priors = list(priors or [0.0] * len(candidates))
        feasible = list(feasible or [True] * len(candidates))
        if len(priors) != len(candidates) or len(feasible) != len(candidates):
            raise ValueError("candidates, priors and feasible must have equal length")
        if not any(feasible):
            raise ValueError("at least one candidate must be feasible")

        decision, attention = self._decision_embedding(state_text, question)
        option_vectors = [self.encoder.encode(option) for option in candidates]
        logits = [
            (_dot(decision, option) + prior) / self.temperature if allowed else -1e9
            for option, prior, allowed in zip(option_vectors, priors, feasible)
        ]
        probabilities = _softmax(logits)
        return ScoreOutput(tuple(logits), tuple(probabilities), tuple(attention))
