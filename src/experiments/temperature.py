from difflib import SequenceMatcher
from src.experiments.base import BaseExperiment
from src.config import ExperimentResult


class TemperatureSweepExperiment(BaseExperiment):
    TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def run(self) -> list[ExperimentResult]:
        results = []
        for temp in self.TEMPERATURES:
            for i in range(self.config.num_runs):
                result = self.run_single(run_index=i + 1, temperature=temp)
                results.append(result)
        return results

    @staticmethod
    def similarity(texts: list[str]) -> float:
        if len(texts) < 2:
            return 1.0
        scores = [
            SequenceMatcher(None, texts[i], texts[j]).ratio()
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        ]
        return sum(scores) / len(scores)
