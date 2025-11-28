from vllm import LLM, SamplingParams
import os, json, time

MOE_LOG_ENABLED = True
SEED = 1234

# Set environment variables
os.environ["VLLM_LOG_MOE_SEED"] = f"{SEED}"
os.environ["VLLM_LOG_MOE_LAYER"] = "0"
os.environ["VLLM_LOG_MOE"] = (
    "EXPERT_TRACKING_DELIVERABLES/moe_routes.jsonl" if MOE_LOG_ENABLED else ""
)


if __name__ == "__main__":
    # Read GSM8K prompts (25 total)
    prompts = (
        open("EXPERT_TRACKING_DELIVERABLES/prompts.txt").read().split("\n\n---\n\n")
    )

    # Initiate sampling params
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    # Load model from checkpoint
    llm = LLM(
        model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        max_model_len=512,
        cpu_offload_gb=30,
        max_num_seqs=1,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.7,
        seed=SEED,
    )

    # Start timer
    t0 = time.time()

    # Run inference
    outs = llm.generate(prompts, sp)

    # End timer
    t1 = time.time()

    # Build timing record
    record = {
        "mode": "log" if MOE_LOG_ENABLED else "no_log",
        "wall_time_sec": t1 - t0,
        "tokens_generated": sum(len(o.outputs[0].token_ids) for o in outs),
    }

    # Append JSON record on a new line (JSON format)
    with open("EXPERT_TRACKING_DELIVERABLES/timing_test.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
