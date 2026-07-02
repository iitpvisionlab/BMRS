import json

from pathlib import Path
from typing import Optional


class ResultsDataManager:

    def __init__(self, input_results_path: Path, prompts_path: Path):
        self.input_path: Path = input_results_path
        self.prompts_path: Path = prompts_path

    def get_initial_data(self):
        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.input_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file: {e}")

    def get_clean_data(
        self, output_path: Optional[Path] = None
    ) -> list[dict[str, str | int]]:
        parsed_data: list[dict[str, str | int]] = []
        input_record = self.get_initial_data()

        modelname = input_record.get("model")
        end_time = input_record.get("end_time")
        seed = input_record.get("end_time")
        strategies = input_record["results"]

        for strat in strategies:
            response = strat["answers"]
            prompt = strat["prompts"]
            for ans in response:
                extracted: dict[str, str | int] = {
                    "username": (f"{modelname}, {strat['strategy']}"),
                    "timestamp": end_time,
                    "seed": seed,
                    "strategy": strat["strategy"],
                    "modelname": modelname,
                    "prompt_id": self._save_new_prompt(prompt),
                }
                image_path = f'{ans.get("problem")}.png'
                extracted["test_image"] = image_path
                extracted["response_type"] = "task_answer"
                extracted["left_answer"] = ans.get("answer", "")
                parsed_data.append(extracted)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)
                print(f"Clean data saved to {output_path}")
        return parsed_data

    def _save_new_prompt(self, new_prompt: list[str]):
        prompts_path = self.prompts_path
        if not prompts_path.parent.is_dir():
            prompts_path.parent.mkdir(parents=True)
        if prompts_path.suffix != ".json":
            raise FileNotFoundError("Wrong suffix of a prompts file")
        string_prompt = "\n".join(new_prompt)
        if not prompts_path.is_file():
            with prompts_path.open("w", encoding="utf-8") as f:
                json.dump({}, f)
        with prompts_path.open(encoding="utf-8") as f:
            prompts = json.load(f)
        if string_prompt not in prompts:
            prompts[string_prompt] = len(prompts)
            with prompts_path.open("w", encoding="utf-8") as f:
                json.dump(prompts, f, ensure_ascii=False, indent=2)
        return prompts[string_prompt]

    def get_sorted_data(
        self,
        output_path: Optional[Path] = None,
    ) -> dict[str, list[dict[str, str | int]]]:
        clean_data = self.get_clean_data()
        unique_users: dict[str, list[dict[str, str | int]]] = {}
        for item in clean_data:
            id = str(item["seed"]) + str(item["username"])
            if id not in unique_users:
                unique_users[id] = []
            unique_users[id].append(item)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(unique_users, f, ensure_ascii=False, indent=2)
                print(f"Sorted data saved to {output_path}")
        return unique_users
