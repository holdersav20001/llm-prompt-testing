from src.experiments.base import BaseExperiment
from src.config import ExperimentResult

DEFAULT_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]


class ModelCompareExperiment(BaseExperiment):
    def __init__(self, client, storage, config, include_opus: bool = False):
        super().__init__(client, storage, config)
        self.models = DEFAULT_MODELS + (["claude-opus-4-6"] if include_opus else [])

    def run(self) -> list[ExperimentResult]:
        results = []
        for model in self.models:
            result = self.run_single(run_index=1, model=model, temperature=0.0)
            results.append(result)
        return results
