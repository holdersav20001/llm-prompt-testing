from src.client import ClaudeClient
from src.storage import SQLiteStorage
from src.config import ExperimentConfig, ExperimentResult


class BaseExperiment:
    def __init__(self, client: ClaudeClient, storage: SQLiteStorage, config: ExperimentConfig):
        self.client = client
        self.storage = storage
        self.config = config

    def run(self) -> list[ExperimentResult]:
        raise NotImplementedError

    def run_single(self, run_index: int = 1, extra_params: dict = None, **overrides) -> ExperimentResult:
        call_kwargs = {
            "model": self.config.model,
            "prompt": self.config.prompt,
            "system_prompt": self.config.system_prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        call_kwargs.update(overrides)

        raw = self.client.call(**call_kwargs)

        result = ExperimentResult(
            experiment=self.config.name,
            model=call_kwargs["model"],
            prompt=call_kwargs["prompt"],
            system_prompt=call_kwargs.get("system_prompt", ""),
            temperature=call_kwargs.get("temperature"),
            run_index=run_index,
            params=extra_params or {},
            **raw,
        )
        self.storage.save(result)
        return result
