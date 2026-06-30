from pathlib import Path
from bmrs.evaluation.results_set import ResultSet

vlm_results = Path("results/test_name/vlm_results.json")
answers_to_evaluate = Path("results/test_name/answers_to_evaluate.json")
prompts_path = Path("bmrs/evaluation/prompts.json") # VLM prompts 
if __name__ == "__main__":
    
    results = ResultSet(results_path=vlm_results, evaluation_path=answers_to_evaluate, prompts_path=prompts_path)
    results.save_evaluation()

