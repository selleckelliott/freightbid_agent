from abc import ABC, abstractmethod

class ScoringStrategy(ABC):

    @abstractmethod
    def score_load(self, load) -> float:
        pass