import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.experiments.base import BaseExperiment
from src.config import ExperimentResult


class MatrixExperiment(BaseExperiment):
    def __init__(
        self,
        client,
        storage,
        config,
        models: list[str],
        system_prompt_variants: list[dict],  # [{"name": str, "content": str}]
        prompt_variants: list[dict],          # [{"name": str, "content": str}]
        temperatures: list[float],
        streaming: bool = False,
        run_id: str = "",
    ):
        super().__init__(client, storage, config, run_id=run_id)
        self.models = models
        self.system_prompt_variants = system_prompt_variants
        self.prompt_variants = prompt_variants
        self.temperatures = temperatures
        self.streaming = streaming

    def _make_tasks(self) -> list[dict]:
        tasks = []
        index = 0
        for model in self.models:
            for sp in self.system_prompt_variants:
                for pv in self.prompt_variants:
                    for temp in self.temperatures:
                        index += 1
                        tasks.append({"index": index, "model": model, "sp": sp, "pv": pv, "temp": temp, "mode": "batch"})
                        if self.streaming:
                            index += 1
                            tasks.append({"index": index, "model": model, "sp": sp, "pv": pv, "temp": temp, "mode": "stream"})
        return tasks

    def _execute_task(self, task: dict) -> ExperimentResult:
        model, sp, pv, temp, mode = task["model"], task["sp"], task["pv"], task["temp"], task["mode"]
        params = {
            "model": model.split("/")[-1],
            "system_prompt": sp["name"],
            "prompt": pv["name"],
            "temperature": temp,
            "mode": mode,
        }
        if mode == "batch":
            return self.run_single(
                run_index=task["index"],
                model=model,
                prompt=pv["content"],
                system_prompt=sp["content"],
                temperature=temp,
                extra_params=params,
            )
        else:
            raw = self.client.stream(
                model=model,
                prompt=pv["content"],
                system_prompt=sp["content"],
                temperature=temp,
                max_tokens=self.config.max_tokens,
            )
            result = ExperimentResult(
                run_id=self.run_id or None,
                experiment=self.config.name,
                model=model,
                prompt=pv["content"],
                system_prompt=sp["content"],
                temperature=temp,
                run_index=task["index"],
                params=params,
                **raw,
            )
            self.storage.save(result)
            return result

    def run(self, on_update=None) -> list[ExperimentResult]:
        tasks = self._make_tasks()
        results = [None] * len(tasks)
        task_status = {t["index"]: "queued" for t in tasks}
        lock = threading.Lock()

        def _push_update():
            if on_update:
                rows = [
                    {
                        "#": t["index"],
                        "Model": t["model"].split("/")[-1],
                        "System Prompt": t["sp"]["name"],
                        "Prompt": t["pv"]["name"],
                        "Temp": t["temp"],
                        "Mode": t["mode"],
                        "Status": task_status[t["index"]],
                    }
                    for t in tasks
                ]
                on_update(rows)

        def _run_task(task):
            with lock:
                task_status[task["index"]] = "running"
            try:
                result = self._execute_task(task)
                with lock:
                    task_status[task["index"]] = "done"
                return result
            except Exception as e:
                with lock:
                    task_status[task["index"]] = f"error: {e}"
                raise

        with ThreadPoolExecutor(max_workers=min(len(tasks), 10)) as pool:
            futures = {pool.submit(_run_task, t): t["index"] - 1 for t in tasks}
            _push_update()  # initial queued state
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    pass
                _push_update()  # update after each completion

        return [r for r in results if r is not None]
