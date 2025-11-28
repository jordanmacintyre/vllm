#!/usr/bin/env python3
"""
Generate a histogram of expert usage from vLLM MoE expert tracking logs.

Reads moe_routes.jsonl and creates a histogram showing how many times each
expert was selected across all tokens.
"""

import json
from collections import Counter
import matplotlib.pyplot as plt


def generate_expert_histogram(
    input_file="EXPERT_TRACKING_DELIVERABLES/moe_routes.jsonl",
    output_file="EXPERT_TRACKING_DELIVERABLES/expert_hist.png",
):
    """
    Read expert tracking log and generate a histogram.

    Args:
        input_file: Path to the JSONL log file
        output_file: Path to save the histogram PNG
    """
    expert_counts = Counter()
    meta_info = None
    total_tokens = 0

    # Read the JSONL file
    with open(input_file, "r") as f:
        for line in f:
            record = json.loads(line.strip())

            if record["type"] == "meta":
                meta_info = record
            elif record["type"] == "route":
                for expert_id in record["topk_ids"]:
                    expert_counts[expert_id] += 1
                total_tokens += 1

    if not expert_counts:
        print("No expert routing data found in log file!")
        return

    # Prepare data for histogram
    experts = sorted(expert_counts.keys())
    counts = [expert_counts[e] for e in experts]
    total_selections = sum(counts)
    percentages = [(count / total_selections) * 100 for count in counts]

    # Statistics
    num_experts = len(experts)
    avg_percentage = 100.0 / num_experts if num_experts else 0
    max_expert = max(expert_counts, key=expert_counts.get)
    min_expert = min(expert_counts, key=expert_counts.get)
    max_pct = (expert_counts[max_expert] / total_selections) * 100
    min_pct = (expert_counts[min_expert] / total_selections) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(experts, percentages, color="steelblue", edgecolor="black", alpha=0.7)

    # Labels and title
    ax.set_xlabel("Expert ID", fontsize=12)
    ax.set_ylabel("Selection Percentage (%)", fontsize=12)

    title = "MoE Expert Selection Frequency (Normalized)"
    if meta_info:
        top_k = meta_info.get("top_k", "?")
        layers = meta_info.get("layers_logged", [])
        title += f' (Top-{top_k}, Layer {layers[0] if layers else "?"})'
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Statistics annotation
    stats_text = (
        f"Total Tokens: {total_tokens}\n"
        f"Num Experts: {num_experts}\n"
        f"Expected (uniform): {avg_percentage:.2f}%"
    )
    ax.text(
        0.5,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    # Get top-3 experts
    top_3_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    # Print statistics
    print(f"Histogram saved to {output_file}")
    print(f"\n=== Summary Statistics ===")
    print(f"  Total tokens processed: {total_tokens}")
    print(f"  Number of unique experts: {num_experts}")
    print(f"  Expected percentage (uniform): {avg_percentage:.2f}%")
    print(f"\n=== Top-3 Most Used Experts ===")
    for i, (expert_id, count) in enumerate(top_3_experts, 1):
        pct = (count / total_selections) * 100
        print(f"  {i}. Expert {expert_id}: {pct:.2f}%")
    print(f"\n=== Distribution Range ===")
    print(f"  Most used: Expert {max_expert} ({max_pct:.2f}%)")
    print(f"  Least used: Expert {min_expert} ({min_pct:.2f}%)")


if __name__ == "__main__":
    import sys

    input_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "EXPERT_TRACKING_DELIVERABLES/moe_routes.jsonl"
    )
    output_file = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "EXPERT_TRACKING_DELIVERABLES/expert_hist.png"
    )

    generate_expert_histogram(input_file, output_file)
