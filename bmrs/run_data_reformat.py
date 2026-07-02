import argparse
from pathlib import Path
from bmrs.reformat.results_set import ResultSet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vlm_answers_path", type=str, default="results/test_name/vlm_results.json")
    parser.add_argument("--reformat_path", type=str, default="results/test_name/answers_to_evaluate.json")

    args = parser.parse_args()
    vlm_answers_path = Path(args.vlm_answers_path)
    reformat_path = Path(args.reformat_path)

    results = ResultSet(results_path=vlm_answers_path, evaluation_path=reformat_path, prompts_path=Path("bmrs/reformat/prompts.json"))
    results.save_evaluation()

