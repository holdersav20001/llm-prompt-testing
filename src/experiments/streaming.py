from src.experiments.base import BaseExperiment
from src.config import ExperimentResult


class StreamingExperiment(BaseExperiment):
    def run(self) -> list[ExperimentResult]:
        results = []

        batch_raw = self.client.call(
            model=self.config.model,
            prompt=self.config.prompt,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        batch_result = ExperimentResult(
            run_id=self.run_id or None,
            experiment=self.config.name,
            model=self.config.model,
            prompt=self.config.prompt,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            run_index=1,
            params={"mode": "batch"},
            **batch_raw,
        )
        self.storage.save(batch_result)
        results.append(batch_result)

        stream_raw = self.client.stream(
            model=self.config.model,
            prompt=self.config.prompt,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        stream_result = ExperimentResult(
            run_id=self.run_id or None,
            experiment=self.config.name,
            model=self.config.model,
            prompt=self.config.prompt,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            run_index=2,
            params={"mode": "stream"},
            **stream_raw,
        )
        self.storage.save(stream_result)
        results.append(stream_result)

        return results
