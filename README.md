# BMRS: Bongard-Maximov problems for Remote Sensing.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/iitpvisionlab/BMRS/blob/main/LICENSE)
[![Paper](https://img.shields.io/badge/paper-preprints.org-orange)](https://www.preprints.org/manuscript/202606.1484)
[![Dataset](https://img.shields.io/badge/dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/nikos74/BMRS)

Official code for the BMRS benchmark paper.

## Table of contents

- [Installation](#installation)
- [Benchmark structure](#benchmark-structure)
- [Generating answers](#generating-answers)
  - [Asking model](#asking-model)
  - [Configuration and strategies](#configuration-and-strategies)
  - [Saving the results](#saving-the-results)
- [Answers evaluation](#answers-evaluation)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/iitpvisionlab/BMRS.git
cd BMRS
pip install -r requirements.txt
```
## Benchmark structure
The benchmark has two stages: generating model answers and evaluating those answers. At the moment, this repository provides the code for the first stage only — inferencing model on our dataset and collecting model outputs.

## Generating answers

An example run is shown in `bmrs/answers_collection/demo.py`. You can run demo using:
```bash
python3 -m bmrs.answers_collection.demo
```


To use BMRS with you model, you need to build a benchmark config and provide two functions:

- `ask_model`: sends one prompt and one or more images to the model and returns the answer.

- `reload_context`: resets the model context between different problem classes so that each class is processed independently.

Communication with model and benchmark config are explained below. 
After you've defined them you can run inference on dataset and save results:

```python
from bmrs.answers_collection.benchmark import BenchmarkConfig, BenchmarkResult, BMRS
from bmrs.answers_collection.strategies import StrategyName
from bmrs.definitions import CONFIG_DIR, RESULTS_DIR

config = BenchmarkConfig.load(CONFIG_DIR / "your_config.json")
benchmark = BMRS(config)

results = benchmark.run(
    ask_model=ask_model,
    reload_context=reload_model,
    checkpoint_dir=RESULTS_DIR / "checkpoints",
)
results.save_as_json(RESULTS_DIR / "results.json")
```

The benchmark will:
- load the dataset directory,
- iterate over each problem folder,
- run the selected strategy for each problem,
- collect answers and skipped items,
- return a `BenchmarkResult`,
- save answers.

### Asking model

BMRS supports two model modes: single_image and multi_image. Some strategies work in both modes, while others are only available for multi-image models. For strategies that are shared between both modes, the results are the same regardless of which interface you use.

1. Single-image model

If your model only accepts one image at a time, you can use it with benchmark.run(...). In this case, your ask_model function should take a prompt and a single image, then return the model output:

```python
def ask_model(prompt: str, image: Path) -> str:
    ...
```

2. Multi-image model

If your model accepts a list of images, you can use it with benchmark.run_multiimage(...). In this case, your ask_model function should take a prompt and a list of images, then return the model output:

```python
def ask_model(prompt: str, images: List[Path]) -> str:
    ...
```

Multi-image models can use additional strategies that rely on comparing several images at once.

### Configuration and strategies

The benchmark is configured through a JSON file loaded with `BenchmarkConfig.load(...)`. Sample config can be found at `configs/sample_config.json`.

The config defines:
- which strategies to run,
- which prompts to use,
- where dataset is stored,
- how to name your model in results files.

At runtime, the benchmark reads the config, loads the dataset, and runs each selected strategy on each problem folder.

To reproduce our results or to compare your model your model with others you can use our config at `configs/bmrs_config.json`.

### Saving the results

You can save and reload benchmark outputs:

```python
results.save_as_json("results.json")

results_from_json = BenchmarkResult.load_from_json_file("results.json")
results_from_json.print_stats()
```

A saved result contains:
- strategy name,
- prompts used,
- answers,
- skipped problems.

## Answers evaluation

Coming soon...

## Troubleshooting

- Make sure your dataset path in config points to a valid dataset folder.

- Make sure each strategy is given the correct number of prompts.

- Make sure the benchmark is called with the correct single-image or multi-image function.

- If a problem is skipped, check the dataset folder structure and your strategy logic.

## Citation

If you use BMRS in your research, please cite the paper linked above.

```bibtex
@article{202606.1484,
    doi = {10.20944/preprints202606.1484.v1},
    url = {https://doi.org/10.20944/preprints202606.1484.v1},
    year = 2026,
    month = {June},
    publisher = {Preprints},
    author = {Nikita Firsov and Olga Terekhova and Nikita Odinets and Alexey Fedotov and Artem Muzyka and Anna Ukhanaeva and Anastasia Sarycheva and Sergei Gladilin and Dmitry Sidorchuk},
    title = {BMRS: Bongard–Maximov Problems for Remote Sensing},
    journal = {Preprints}
}
```

## License

This project is licensed under the MIT License.
