from typing import List
from pathlib import Path

from bmrs.answers_collection.benchmark import BenchmarkConfig, BenchmarkResult, BMRS
from bmrs.answers_collection.strategies import StrategyName
from bmrs.definitions import CONFIG_DIR, RESULTS_DIR


def ask_model(prompt: str, image: Path):
    if image.exists():
        return f"answer for image `{image}` and prompt `{prompt}`"
    return "no image there :("


def ask_model_multiimage(prompt: str, images: List[Path]) -> str:
    return "answer from multi-image model"


def reload_model():
    pass


if __name__ == "__main__":
    # inference model
    config_path = CONFIG_DIR / "sample_config.json"

    config = BenchmarkConfig.load(config_path)
    benchmark = BMRS(config)

    # run for single-image model
    results = benchmark.run(
        ask_model,
        reload_model,
        checkpoint_dir=RESULTS_DIR / "checkpoints",
        strategies=[
            StrategyName.CONTRASTIVE_ITERATIVE,
            StrategyName.CONTRASTIVE_DIRECT,
        ],
    )
    results.save_as_json("results.json")

    # run for multi-image model
    results = benchmark.run_multiimage(
        ask_model_multiimage,
        reload_model,
        checkpoint_dir="bench_checkpoints",
        strategies=[
            StrategyName.CONTRASTIVE_DIRECT_MULTIIMAGE,
            StrategyName.CONTRASTIVE_ITERATIVE_MULTIIMAGE,
            StrategyName.CONTRASTIVE_ITERATIVE,
        ],
    )
    results.save_as_json("results.json")

    # load inference results
    results_from_json = BenchmarkResult.load_from_json_file("results.json")
    results_from_json.print_stats()
