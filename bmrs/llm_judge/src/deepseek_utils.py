import os
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class DeepSeekRouterModel:
    client: OpenAI
    model_name: str


def load_llm(
    model_name: str = "deepseek-v4-flash",
    adapter_path: str | None = None,
    api_key_env: str = "DEEPSEEK_API_KEY",
    base_url: str = "https://api.deepseek.com",
):
    # adapter_path оставляем только для совместимости с run_llm_judge.py.
    # Через remote API локальные LoRA/PEFT adapter_path не применяются.
    if adapter_path is not None:
        print(
            f"Warning: adapter_path={adapter_path!r} ignored, "
            "because DeepSeek API uses a remote model."
        )

    api_key = os.environ.get(api_key_env)

    if not api_key:
        raise RuntimeError(
            f"Missing DeepSeek API key. Set it with:\n"
            f'export {api_key_env}="your_deepseek_api_key_here"'
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    tokenizer = None
    model = DeepSeekRouterModel(
        client=client,
        model_name=model_name,
    )

    return tokenizer, model


def parse_binary_answer(answer: str) -> str:
    answer = answer.strip()

    if answer.startswith("1"):
        return "1"

    if answer.startswith("0"):
        return "0"

    return "unknown"


def parse_correct_incorrect_answer(answer: str) -> str:
    answer = answer.strip().lower()

    if answer.startswith("incorrect"):
        return "incorrect"

    if answer.startswith("correct"):
        return "correct"

    # fallback на случай фразы вроде "The answer is incorrect."
    if "incorrect" in answer:
        return "incorrect"

    if "correct" in answer:
        return "correct"

    return "unknown"


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    tokenizer,
    model: DeepSeekRouterModel,
    output_format: str = "binary",
) -> str:
    if output_format == "binary":
        format_instruction = (
            "\n\nReturn only one character: 1 or 0. " "Do not include explanations."
        )
    elif output_format == "correct_incorrect":
        format_instruction = (
            "\n\nReturn only one word: correct or incorrect. "
            "Do not include explanations."
        )
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt + format_instruction,
        },
    ]

    completion = model.client.chat.completions.create(
        model=model.model_name,
        messages=messages,
        stream=False,
        temperature=0,
        max_tokens=8192,
        reasoning_effort="high",
        extra_body={
            "thinking": {
                "type": "enabled",
            }
        },
    )

    message = completion.choices[0].message
    raw_answer = message.content or ""
    raw_answer = raw_answer.strip()

    print(f"RAW LLM ANSWER: {raw_answer!r}")

    if output_format == "binary":
        return parse_binary_answer(raw_answer)

    if output_format == "correct_incorrect":
        return parse_correct_incorrect_answer(raw_answer)

    raise ValueError(f"Unknown output_format: {output_format}")
