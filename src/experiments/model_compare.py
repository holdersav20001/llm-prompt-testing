from src.experiments.base import BaseExperiment
from src.config import ExperimentResult

DEFAULT_MODELS = ["anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-4.6"]
EXTRA_MODEL = "moonshotai/kimi-k2.5"


class ModelCompareExperiment(BaseExperiment):
    def __init__(self, client, storage, config, models: list[str] = None, include_opus: bool = False, run_id: str = ""):
        super().__init__(client, storage, config, run_id=run_id)
        if models:
            self.models = models
        else:
            self.models = DEFAULT_MODELS + ([EXTRA_MODEL] if include_opus else [])

    def run(self) -> list[ExperimentResult]:
        results = []
        for model in self.models:
            result = self.run_single(run_index=1, model=model, temperature=self.config.temperature or 0.0)
            results.append(result)
        return results
