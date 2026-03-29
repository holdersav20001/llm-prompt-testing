from src.experiments.base import BaseExperiment
from src.config import ExperimentResult

SYSTEM_PROMPT_VARIANTS = [
    ("none", ""),
    ("short_role", "You are a helpful assistant."),
    ("detailed_role", "You are an expert technical writer who produces clear, well-structured, accurate responses."),
    ("format_constraints", "You are a helpful assistant. Always respond in bullet points. Keep responses under 100 words."),
    ("persona_constraints", "You are Alex, a senior software engineer with 10 years of experience. Respond in a friendly but direct tone. Structure your response with: 1) Direct answer 2) Brief explanation 3) One concrete example."),
]


class SystemPromptsExperiment(BaseExperiment):
    def run(self) -> list[ExperimentResult]:
        results = []
        for variant_name, system_prompt in SYSTEM_PROMPT_VARIANTS:
            result = self.run_single(
                run_index=1,
                extra_params={"variant": variant_name},
                system_prompt=system_prompt,
                temperature=0.0,
            )
            results.append(result)
        return results
