import json

from pathlib import Path
from typing import Any

def clean_data(
    input_file: Path,
    dump: bool = True,
    output_file: Path = Path("parsed_data"),
    prompts_path: Path = Path("_delete_me.json")
) -> list[dict[str, str | int]]:
    """
    Parse experiment data from JSONL format, extract only the response data
    and relevant metadata, and save to a clean JSON file.
    """

    parsed_data: list[dict[str, str | int]] = []

    with open(input_file, "r", encoding="utf-8") as f:

        try:
            record = json.load(f)

            # Extract only the data we need
            modelname = record.get("model")
            end_time  = record.get("end_time")
            seed = record.get("end_time")

            # Extract response data
            strategies = record["results"]
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
                    "prompt_id": save_new_prompt(prompt, prompts_path)
                    }
                    image_path = (f'{ans.get("problem")}.png')
                    extracted["test_image"] = image_path
                    # Check if this is an answer to a task (has left_ans and right_ans)
                    extracted["response_type"] = "task_answer"
                    extracted["left_answer"] = ans.get("answer", "")

                    parsed_data.append(extracted)
        
        
        except json.JSONDecodeError as e:
            print(f"Error parsing file: {e}")
    # Save to JSON file
    if dump:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)

        print(f"Successfully parsed {len(parsed_data)} records")
        print(f"Data saved to {output_file}")
    return parsed_data

def save_new_prompt(new_prompt: list[str], prompts_path: Path):
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


def sort_data(
    clean_data: list[Any],
    full: bool = True,
    dump: bool = True,
    output_path: Path = Path("sorted_data.json"),
    entry_amount: int = 22
) -> dict[str, list[dict[str, str | int]]]:
    """
    Full: return only completed tests
    """
    unique_users: dict[str, list[dict[str, str | int]]] = {}
    for item in clean_data:
        id = str(item["seed"]) + item["username"]
        if id not in unique_users:
            unique_users[id] = []
        unique_users[id].append(item)
    if full:
        to_del: list[str] = []
        for key, value in unique_users.items():
            if len(value) < entry_amount:
                print(f"{len(value)} < {entry_amount}")
                to_del.append(key)
                print(f"{key} did not complete the experiment")
        [unique_users.pop(key) for key in to_del]
    if dump:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(unique_users, f, ensure_ascii=False, indent=2)
    return unique_users

def load_model_data(input_file: Path, prompts_path: Path) -> dict[str, list[dict[str, str | int]]]:

    parsed_data = clean_data(input_file, dump=False, prompts_path=prompts_path)

    sorted_data = sort_data(
        parsed_data, full=False, dump=False, output_path=Path("grouped_relevant.json")
    )
    return sorted_data

