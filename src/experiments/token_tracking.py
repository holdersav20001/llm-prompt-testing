from src.experiments.base import BaseExperiment
from src.config import ExperimentResult

PROMPT_VARIANTS = [
    ("short", "What is 2+2?"),
    ("medium", "Explain the difference between supervised and unsupervised machine learning in 2-3 sentences."),
    ("long", "Write a detailed explanation of how neural networks work, covering: 1) perceptrons, 2) layers, 3) activation functions, 4) backpropagation, and 5) gradient descent."),
]

DEFAULT_MODELS = ["anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-4.6"]


class TokenTrackingExperiment(BaseExperiment):
    def __init__(self, client, storage, config, models: list[str] = None, run_id: str = ""):
        super().__init__(client, storage, config, run_id=run_id)
        self.models = models or DEFAULT_MODELS

    def run(self) -> list[ExperimentResult]:
        results = []
        for prompt_name, prompt in PROMPT_VARIANTS:
            for model in self.models:
                result = self.run_single(
                    run_index=1,
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    extra_params={"prompt_length": prompt_name},
                )
                results.append(result)
        return results
