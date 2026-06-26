def build_examples_block(examples: list[dict] | None, title: str) -> str:
    if not examples:
        return ""

    blocks = [f"{title}:"]

    for i, example in enumerate(examples, start=1):
        blocks.append(
            f"Example {i}:\n"
            f"Left: {example.get('left_answer', '')}\n"
            f"Right: {example.get('right_answer', '')}"
        )

    return "\n\n".join(blocks)


def build_user_prompt(
    reference_left: str,
    reference_right: str,
    user_left: str,
    correct_examples: list[dict] | None = None,
    incorrect_examples: list[dict] | None = None,
    task_specific_prompt: str | None = None,
) -> str:
    parts = []

    if task_specific_prompt:
        parts.append(
            f"Task-specific evaluation instruction:\n"
            f"{task_specific_prompt}"
        )
    

    parts.append(
        f"Reference answers:\n"
        f"Left: {reference_left}\n"
        f"Right: {reference_right}"
    )

    correct_examples_block = build_examples_block(
        correct_examples,
        "Examples of correct answers"
    )

    incorrect_examples_block = build_examples_block(
        incorrect_examples,
        "Examples of incorrect answers"
    )

    if correct_examples_block:
        parts.append(correct_examples_block)

    if incorrect_examples_block:
        parts.append(incorrect_examples_block)

    parts.append(
        f"Model answer:\n"
        f"{user_left}"
    )

    return "\n\n".join(parts).strip()