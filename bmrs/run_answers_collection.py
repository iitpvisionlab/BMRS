import argparse
from pathlib import Path
from bmrs.answers_collection.benchmark import BMRS, BenchmarkConfig
from bmrs.definitions import RESULTS_DIR
from bmrs.answers_collection.strategies import StrategyName
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from PIL import Image

config_path = ("configs/test_config.json")
config = BenchmarkConfig.load(config_path)
benchmark = BMRS(config)


vlm_path = "Qwen/Qwen3.5-2B"

processor = AutoProcessor.from_pretrained(vlm_path)
model = AutoModelForImageTextToText.from_pretrained(
    vlm_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
    attn_implementation="sdpa",
).eval()

# print(model.hf_device_map)

# SYS_PROMPT = ""

SYS_PROMPT = """You are a vision understanding module designed to provide short, clear and accurate answers. Your goal is to solve Bongard problem, consinting of a collage with six images on the left side and six on the right side. All left images share a common concept that none of the right images have, and all right images share a different common concept that none of the left images have. Your task is to identify both concepts. Answer is exactly two plain sentences: the first sentence describes the concept of the left side, the second sentence describes the concept of the right side. Do not use markdown, bullet points, or any formatting. Keep each sentence short and clear.""" 

def ask_model(prompt, image, model=model, processor=processor):
    if not image.exists():
        return "no image there :("

    image = Image.open(image).convert("RGB")

    full_prompt = SYS_PROMPT + "\n\n" + prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda:0")
    eos_id = processor.tokenizer.eos_token_id
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            pad_token_id=eos_id,
            eos_token_id=eos_id
        )

    decoded_output = processor.decode(
        generate_ids[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    print("ANSWER:", decoded_output)
    return decoded_output

def reload_model():
    pass

# print("results:", RESULTS_DIR)
results = benchmark.run(
    ask_model,
    reload_model,
    checkpoint_dir = RESULTS_DIR / config.model / "checkpoints",
    strategies=[
        StrategyName.DESCRIPTIVE_ITERATIVE,
        StrategyName.CONTRASTIVE_DIRECT,
    ],
)
results.save_as_json(RESULTS_DIR / config.model / "vlm_results.json")