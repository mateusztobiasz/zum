from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import entropy


class QueryStrategy(ABC):

    @abstractmethod
    def select(self, probs: np.ndarray, n: int) -> np.ndarray:
        pass


class RandomSampling(QueryStrategy):
    def select(self, probs, n):
        return np.random.choice(len(probs), size=n, replace=False)


class LeastConfidence(QueryStrategy):
    def select(self, probs, n):
        confidences = probs.max(axis=1)
        return np.argsort(confidences)[:n]  # lowest confidence = most uncertain


class MarginSampling(QueryStrategy):
    def select(self, probs, n):
        part = np.partition(-probs, 1, axis=1)
        margins = -(part[:, 0] - part[:, 1])
        return np.argsort(margins)[:n]  # smallest margin = most uncertain


class EntropySampling(QueryStrategy):
    def select(self, probs, n):
        entropies = entropy(probs, axis=1)  # FIXED here
        return np.argsort(entropies)[-n:]  # highest entropy = most uncertain
