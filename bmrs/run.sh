
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_JUDGE_ROOT="$PROJECT_ROOT/bmrs/llm_judge"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$LLM_JUDGE_ROOT:$PYTHONPATH"

## Run configuration
CONFIG_PATH="$PROJECT_ROOT/configs/test_config.json"
JUDGE_MODEL_NAME="Qwen/Qwen3.5-2B"
JUDGE_PROMPT_NAME="judge_system_prompt_en"
RESULTS_SUBDIR="test_specific"


## Paths configuration
VLM_MODEL_NAME=$(python -c "import json; print(json.load(open('$CONFIG_PATH'))['model'])")
VLM_RESULTS="$PROJECT_ROOT/results/$VLM_MODEL_NAME/vlm_results.json"
ANSWERS_REFORMAT="$PROJECT_ROOT/results/$VLM_MODEL_NAME/answers_reformat.json"
SYSTEM_PROMPT="$LLM_JUDGE_ROOT/prompts_data/judge_system_prompt_en.txt"
REFERENCE_PATH="$LLM_JUDGE_ROOT/prompts_data/reference_answers_en.json"
EXAMPLES_PATH="$LLM_JUDGE_ROOT/prompts_data/answer_examples_en.json"
TASKS_PATH="$LLM_JUDGE_ROOT/prompts_data/tasks_en.json"
TASK_TYPE_PROMPTS_PATH="$LLM_JUDGE_ROOT/prompts_data/task_type_judge_prompts_en.json"

python "$PROJECT_ROOT/bmrs/run_answers_collection.py"\
  --config_path "$CONFIG_PATH"

python "$PROJECT_ROOT/bmrs/run_data_reformat.py"\
  --vlm_answers_path "$VLM_RESULTS"\
  --reformat_path "$ANSWERS_REFORMAT"

python "$PROJECT_ROOT/bmrs/run_llm_judge.py" \
  --model_name "$JUDGE_MODEL_NAME" \
  --prompt_name "$JUDGE_PROMPT_NAME" \
  --reference_path "$REFERENCE_PATH" \
  --model_answers "$ANSWERS_REFORMAT" \
  --examples_path "$EXAMPLES_PATH" \
  --results_subdir "$RESULTS_SUBDIR" \
  --system_prompt "$SYSTEM_PROMPT" \
  --use_task_specific_prompts \
  --tasks_path "$TASKS_PATH" \
  --task_type_prompts_path "$TASK_TYPE_PROMPTS_PATH"