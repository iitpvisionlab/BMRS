import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_llm(
    model_name: str,
    adapter_path: str | None = None,
):
    tokenizer_path = adapter_path if adapter_path is not None else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if torch.cuda.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    elif torch.backends.mps.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to("mps")

    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )
    else:
        model = base_model

    model.eval()

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

    if "incorrect" in answer:
        return "incorrect"

    if "correct" in answer:
        return "correct"

    return "unknown"


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    tokenizer,
    model,
    output_format: str = "binary",
) -> str:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if output_format == "binary":
        verdict = parse_binary_answer(raw_answer)
    elif output_format == "correct_incorrect":
        verdict = parse_correct_incorrect_answer(raw_answer)
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

    return verdict