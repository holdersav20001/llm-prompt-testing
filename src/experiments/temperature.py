from src.experiments.base import BaseExperiment
from src.config import ExperimentResult

TEMPERATURES = [0.0, 0.1, 0.5, 1.0]


class TemperatureSweepExperiment(BaseExperiment):
    def run(self) -> list[ExperimentResult]:
        results = []
        for i, temp in enumerate(TEMPERATURES):
            result = self.run_single(
                run_index=i + 1,
                extra_params={"temperature_value": temp},
                temperature=temp,
            )
            results.append(result)
        return results
