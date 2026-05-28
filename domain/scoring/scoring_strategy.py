from abc import ABC, abstractmethod

from domain.models.load_evaluation import LoadEvaluation
from domain.models.score_result import ScoreResult


class ScoringStrategy(ABC):

    @abstractmethod
    def score_load(self, load: LoadEvaluation) -> ScoreResult:
        ...