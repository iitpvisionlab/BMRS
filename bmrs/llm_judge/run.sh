#!/bin/bash

export PYTHONPATH=.

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_NAME="judge_system_prompt_en"
RESULTS_SUBDIR="test_specific"


SYSTEM_PROMPT="bmrs/llm_judge/prompts_data/judge_system_prompts_en.json"
MODEL_ANSWERS="data/finetune/test" #finetune/test answers/model_answers
REFERENCE_PATH="bmrs/llm_judge/prompts_data/reference_answers_en.json"
EXAMPLES_PATH="bmrs/llm_judge/prompts_data/answer_examples_en.json" 

TASKS_PATH="bmrs/llm_judge/prompts_data/tasks_en.json"
TASK_TYPE_PROMPTS_PATH="bmrs/llm_judge/prompts_data/task_type_judge_prompts_en.json"

python scripts/run_llm_judge.py \
  --model_name "$MODEL_NAME" \
  --prompt_name "$PROMPT_NAME" \
  --reference_path "$REFERENCE_PATH" \
  --model_answers "$MODEL_ANSWERS" \
  --examples_path "$EXAMPLES_PATH"\
  --results_subdir "$RESULTS_SUBDIR"\
  --system_prompt "$SYSTEM_PROMPT"\
  --use_task_specific_prompts \
  --tasks_path "$TASKS_PATH" \
  --task_type_prompts_path "$TASK_TYPE_PROMPTS_PATH"
