import json
import re
from pathlib import Path


def load_json(path: str | Path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_text(path: str | Path) -> str:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_text(text: str, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def slugify_name(name: str) -> str:
    """
    Converts strings like:
    Qwen/Qwen2.5-1.5B-Instruct

    into:
    qwen_qwen2_5_1_5b_instruct
    """

    name = name.strip().lower()
    name = name.replace("/", "_")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    return name


def make_run_dir(
    model_name: str,
    prompt_name: str,
    base_dir: str | Path = "results/runs",
) -> Path:
    model_slug = slugify_name(model_name)
    prompt_slug = slugify_name(prompt_name)

    run_dir = Path(base_dir) / f"{model_slug}_{prompt_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir