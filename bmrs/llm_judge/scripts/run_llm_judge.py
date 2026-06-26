from pathlib import Path

from src.data_utils import load_json, save_json, load_text, make_run_dir
from src.prompt_builder import build_user_prompt

# from src.llm_utils import load_llm, ask_llm

# from src.qwen_utils import load_llm, ask_llm

from src.deepseek_utils import load_llm, ask_llm

import argparse


def convert_human_unclear_to_users_format(human_data: dict):
    users_data = {}

    for image, users in human_data.items():
        for user_id, record in users.items():
            users_data.setdefault(user_id, []).append(
                {
                    "response_type": "task_answer",
                    "test_image": image,
                    "left_answer": record.get("left_answer", ""),
                    "right_answer": record.get("right_answer", ""),
                    "evaluation": record.get("evaluation", ""),
                }
            )

    return users_data


def get_task_specific_prompt(
    test_image: str,
    tasks_metadata: dict,
    task_type_prompts: dict,
) -> tuple[str | None, int | None, str]:
    task_info = tasks_metadata.get(test_image)

    if task_info is None:
        return None, None, ""

    task_type = task_info.get("type")
    task_complexity = task_info.get("complexity")

    if task_type is None:
        return None, task_complexity, ""

    prompt_info = task_type_prompts.get(task_type, {})

    if isinstance(prompt_info, dict):
        task_specific_prompt = prompt_info.get("prompt", "")
    else:
        task_specific_prompt = str(prompt_info)

    return task_type, task_complexity, task_specific_prompt


def check_all_users(
    reference_path: str | Path,
    users_path: str | Path,
    system_prompt_path: str | Path,
    tokenizer,
    model,
    output_format: str,
    examples_path: str | Path | None = None,
    use_task_specific_prompts: bool = False,
    tasks_metadata: dict | None = None,
    task_type_prompts: dict | None = None,
):
    references = load_json(reference_path)
    users_data = load_json(users_path)

    users_data = convert_human_unclear_to_users_format(users_data)

    total_users = len(users_data)
    print(f"Всего пользователей: {total_users}")

    system_prompt = load_text(system_prompt_path)

    if examples_path is not None:
        examples = load_json(examples_path)
    else:
        examples = {}

    if tasks_metadata is None:
        tasks_metadata = {}

    if task_type_prompts is None:
        task_type_prompts = {}

    detailed_results = {}
    summary_results = {}

    for user_number, (user_id, records) in enumerate(users_data.items(), start=1):
        print(f"\nПользователь {user_number}/{total_users}: {user_id}")

        user_task_results = []

        correct_count = 0
        incorrect_count = 0
        unknown_count = 0
        missing_reference_count = 0

        task_number = 0

        for record in records:
            if record.get("response_type") != "task_answer":
                continue

            task_number += 1

            test_image = record.get("test_image")
            user_left = record.get("left_answer", "")
            user_right = record.get("right_answer", "")

            print(f"  Задача {task_number}: {test_image}")

            if test_image not in references:
                user_task_results.append(
                    {
                        "test_image": test_image,
                        "user_left": user_left,
                        "user_right": user_right,
                        "verdict": "unknown",
                        "error": "reference_not_found",
                    }
                )

                unknown_count += 1
                missing_reference_count += 1
                continue

            ref_left = references[test_image].get("left_answer", "")
            ref_right = references[test_image].get("right_answer", "")

            task_examples = examples.get(test_image, {})

            correct_example = task_examples.get("correct_examples")
            incorrect_example = task_examples.get("incorrect_examples")

            task_type = None
            task_complexity = None
            task_specific_prompt = ""

            if use_task_specific_prompts:
                task_type, task_complexity, task_specific_prompt = (
                    get_task_specific_prompt(
                        test_image=test_image,
                        tasks_metadata=tasks_metadata,
                        task_type_prompts=task_type_prompts,
                    )
                )

            user_prompt = build_user_prompt(
                reference_left=ref_left,
                reference_right=ref_right,
                user_left=user_left,
                correct_examples=correct_example,
                incorrect_examples=incorrect_example,
                task_specific_prompt=task_specific_prompt,
            )

            verdict = ask_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tokenizer=tokenizer,
                model=model,
                output_format=output_format,
            )

            verdict_normalized = str(verdict).strip().lower()

            if verdict_normalized in ["1", "correct"]:
                correct_count += 1
            elif verdict_normalized in ["0", "incorrect"]:
                incorrect_count += 1
            else:
                unknown_count += 1

            user_task_results.append(
                {
                    "test_image": test_image,
                    "task_type": task_type,
                    "task_complexity": task_complexity,
                    "task_specific_prompt": task_specific_prompt,
                    "reference_left": ref_left,
                    "reference_right": ref_right,
                    "user_left": user_left,
                    "user_right": user_right,
                    "correct_example": correct_example,
                    "incorrect_example": incorrect_example,
                    "user_prompt": user_prompt,
                    "verdict": verdict,
                }
            )

        detailed_results[user_id] = user_task_results

        summary_results[user_id] = {
            "total_tasks_checked": correct_count + incorrect_count + unknown_count,
            "correct": correct_count,
            "incorrect": incorrect_count,
            "unknown": unknown_count,
            "missing_reference_count": missing_reference_count,
        }

    return detailed_results, summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")

    parser.add_argument("--prompt_name", type=str, default="llm_1_correct_ru")

    parser.add_argument(
        "--reference_path", type=str, default="data/reference_answers.json"
    )

    parser.add_argument(
        "--model_answers", type=str, default="data/answers/model_answers"
    )

    parser.add_argument(
        "--examples_path", type=str, default="data/prompts/examples.json"
    )

    parser.add_argument("--results_subdir", type=str, default="test")
    parser.add_argument("--system_prompt", type=str, default="bmrs/llm_judge/prompts_data/judge_system_prompt_en.txt")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--use_task_specific_prompts",
        action="store_true",
    )

    parser.add_argument(
        "--tasks_path",
        type=str,
        default="data/tasks.json",
    )

    parser.add_argument(
        "--task_type_prompts_path",
        type=str,
        default="data/prompts/task_type_prompts.json",
    )

    args = parser.parse_args()

    model_name = args.model_name
    adapter_path = args.adapter_path

    prompt_name = args.prompt_name
    output_format = "correct_incorrect"

    reference_path = Path(args.reference_path)
    model_answers = Path(args.model_answers)
    examples_path = Path(args.examples_path)
    system_prompt_path = Path(args.system_prompt)

    tasks_metadata = {}
    task_type_prompts = {}

    if args.use_task_specific_prompts:
        tasks_data = load_json(args.tasks_path)
        task_type_prompts = load_json(args.task_type_prompts_path)

        tasks_metadata = {
            task["name"]: task for task in tasks_data.get("tasks", []) if "name" in task
        }

        print(f"Loaded task metadata: {len(tasks_metadata)} tasks")
        print(f"Loaded task type prompts: {len(task_type_prompts)} types")


    run_dir = make_run_dir(
        model_name=model_name,
        prompt_name=prompt_name,
    )

    results_dir = run_dir / args.results_subdir
    model_ans_dir = results_dir / "model_ans"
    model_ans_dir.mkdir(parents=True, exist_ok=True)

    all_detailed_results = []
    all_summary_results = []

    tokenizer, model = load_llm(
        model_name=model_name,
        adapter_path=adapter_path,
    )

    answers_path = model_answers
    print(f"Processing {answers_path.name}")

    users_name = answers_path.stem.replace("-", "_")

    output_dir = model_ans_dir / users_name
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_results, summary_results = check_all_users(
        reference_path=reference_path,
        users_path=answers_path,
        system_prompt_path=system_prompt_path,
        tokenizer=tokenizer,
        model=model,
        output_format=output_format,
        examples_path=examples_path,
        use_task_specific_prompts=args.use_task_specific_prompts,
        tasks_metadata=tasks_metadata,
        task_type_prompts=task_type_prompts,
    )

    save_json(detailed_results, output_dir / "predictions.json")
    save_json(summary_results, output_dir / "summary.json")

    run_config = {
        "model_name": model_name,
        "prompt_name": prompt_name,
        "output_format": output_format,
        "reference_path": str(reference_path),
        "users_path": str(answers_path),
        "system_prompt_path": str(system_prompt_path),
        "examples_path": str(examples_path),
        "output_dir": str(output_dir),
        "output_predictions": str(output_dir / "predictions.json"),
        "output_summary": str(output_dir / "summary.json"),
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "use_task_specific_prompts": args.use_task_specific_prompts,
        "tasks_path": (
            str(args.tasks_path) if args.use_task_specific_prompts else None
        ),
        "task_type_prompts_path": (
            str(args.task_type_prompts_path)
            if args.use_task_specific_prompts
            else None
        ),
    }

    save_json(run_config, output_dir / "run_config.json")

    all_detailed_results.append(
        {
            "answers_name": users_name,
            "answers_path": str(answers_path),
            "output_dir": str(output_dir),
            "predictions": detailed_results,
        }
    )

    all_summary_results.append(
        {
            "answers_name": users_name,
            "answers_path": str(answers_path),
            "output_dir": str(output_dir),
            "summary": summary_results,
        }
    )


    print("Done")
