#!/usr/bin/env bash
set -e

LLM_JUDGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$LLM_JUDGE_ROOT/.." && pwd)"
LLM_JUDGE_ROOT="$PROJECT_ROOT/bmrs/llm_judge"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$LLM_JUDGE_ROOT:$PYTHONPATH"

MODEL_NAME="Qwen/Qwen3.5-2B"
PROMPT_NAME="judge_system_prompt_en"
RESULTS_SUBDIR="test_specific"

SYSTEM_PROMPT="$LLM_JUDGE_ROOT/prompts_data/judge_system_prompt_en.txt"
REFERENCE_PATH="$LLM_JUDGE_ROOT/prompts_data/reference_answers_en.json"
EXAMPLES_PATH="$LLM_JUDGE_ROOT/prompts_data/answer_examples_en.json"
TASKS_PATH="$LLM_JUDGE_ROOT/prompts_data/tasks_en.json"
TASK_TYPE_PROMPTS_PATH="$LLM_JUDGE_ROOT/prompts_data/task_type_judge_prompts_en.json"

ANSWERS_TO_EVALUATE="$PROJECT_ROOT/results/test/test_evaluation_s.json"

python "$PROJECT_ROOT/bmrs/run_llm_judge.py" \
  --model_name "$MODEL_NAME" \
  --prompt_name "$PROMPT_NAME" \
  --reference_path "$REFERENCE_PATH" \
  --model_answers "$ANSWERS_TO_EVALUATE" \
  --examples_path "$EXAMPLES_PATH" \
  --results_subdir "$RESULTS_SUBDIR" \
  --system_prompt "$SYSTEM_PROMPT" \
  --use_task_specific_prompts \
  --tasks_path "$TASKS_PATH" \
  --task_type_prompts_path "$TASK_TYPE_PROMPTS_PATH"