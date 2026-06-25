import json
from pathlib import Path
from typing import Dict, List, Optional
from bmrs.evaluation.results_reader import ResultsDataManager


class ResultSet:
    def __init__(
        self,
        results_path: Path,
        evaluation_path: Path = Path("evaluation.json"),
        tasks_to_remove: list[str] = [],
        prompts_path: Optional[Path] = None,
    ):
        """
        results_path: results input file (json)
        evaluation_path: upcoming task evaluation output file path (json)
        tasks_to_remove: optional list of tasks not to include
        prompts_path: json file with VLM prompts list. Will be created if not exist (json)
        """

        if not prompts_path:
            prompts_path = Path("data/prompts.json")
        self.data = ResultsDataManager(
            results_path, prompts_path
        ).get_sorted_data()  # General data

        self.results_path = results_path
        self.evaluation_path = evaluation_path
        self.tasks_to_remove: list[str] = tasks_to_remove
        self.users: Dict[str, Dict[str, str]] = {}  # Users info
        self.prompts_path = prompts_path  # 
        self.user_tasks: Dict[str, List[Dict[str, str | int]]] = (
            {}
        )  # Tasks answers by users
        self.evaluations: Dict[str, Dict[str, Dict[str, str | int]]] = {}
        # Tasks with evaluations by tasks (tasks[user ids[task responses + evaluation]])
        self.load_results()
        self.load_evaluation()

    def load_results(self):
        if not self.results_path.exists():
            raise FileNotFoundError(f"file {self.results_path} not found!")

        self._exclude_tasks(self.tasks_to_remove)
        # data setup
        for user_id, records in self.data.items():
            self.user_tasks[user_id] = []

            for _, record in enumerate(records):
                if record.get("response_type") == "task_answer":
                    task_info = {
                        "test_image": record.get("test_image"),
                        "left_answer": record.get("left_answer"),
                    }
                    self.user_tasks[user_id].append(task_info)

    def load_evaluation(self):

        if self.evaluation_path.exists():
            with open(self.evaluation_path, "r", encoding="utf-8") as f:
                self.evaluations = json.load(f)
        evaluations = self.evaluations
        for user_id, records in self.data.items():
            for _, record in enumerate(records):
                if record["response_type"] == "task_answer":
                    img_name = record["test_image"]
                    if img_name not in evaluations:
                        evaluations[img_name] = {}
                    if user_id not in evaluations[img_name]:
                        empty_eval = {
                            "user_id": user_id,
                            "username": record["username"],
                            "modelname": record["modelname"],
                            "strategy": record["strategy"],
                            "left_answer": record["left_answer"],
                            "prompt_id": record.get("prompt_id", "N/A"),
                            "evaluation": "",
                        }
                        evaluations[img_name][user_id] = empty_eval

    def _exclude_tasks(self, tasks: list[str]):
        full_data = self.data
        for userid in full_data:
            to_del: list[int] = []
            if tasks:
                for i, task in enumerate(full_data[userid]):
                    if task["response_type"] == "task_answer":
                        if task["test_image"] in tasks:
                            to_del.append(i)
                full_data[userid] = [
                    item
                    for idx, item in enumerate(full_data[userid])
                    if idx not in to_del
                ]

    def save_evaluation(self):
        with open(self.evaluation_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluations, f, ensure_ascii=False, indent=2)
