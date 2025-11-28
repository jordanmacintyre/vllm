from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main", split="test")

prompts = [ex["question"] for ex in ds.select(range(25))]

open("prompts.txt", "w").write("\n\n---\n\n".join(prompts))
