#!/usr/bin/env python3
"""
Analyze MoE routing patterns from vLLM expert tracking logs.

Computes entropy, top-choice expert preferences, and overall selection statistics.
"""

import json
import math
from collections import Counter


def calculate_entropy(counts, total):
    """
    Calculate Shannon entropy of expert distribution.

    Args:
        counts: Dictionary of expert_id -> count
        total: Total number of selections

    Returns:
        Entropy in bits
    """
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def analyze_moe_routing(input_file="EXPERT_TRACKING_DELIVERABLES/moe_routes.jsonl"):
    """
    Analyze MoE routing patterns from expert tracking log.

    Args:
        input_file: Path to the JSONL log file
    """
    expert_counts = Counter()  # Total selections (all positions)
    top_choice_counts = Counter()  # Only first choice in topk_ids
    meta_info = None
    total_tokens = 0

    # Read the JSONL file
    with open(input_file, "r") as f:
        for line in f:
            record = json.loads(line.strip())

            if record["type"] == "meta":
                meta_info = record
            elif record["type"] == "route":
                topk_ids = record["topk_ids"]

                # Count all expert selections
                for expert_id in topk_ids:
                    expert_counts[expert_id] += 1

                # Count only top choice (first expert in topk_ids)
                if topk_ids:
                    top_choice_counts[topk_ids[0]] += 1

                total_tokens += 1

    if not expert_counts:
        print("No expert routing data found in log file!")
        return

    # Statistics
    num_experts = len(expert_counts)
    total_selections = sum(expert_counts.values())

    # Calculate entropy
    entropy = calculate_entropy(expert_counts, total_selections)
    top_choice_entropy = calculate_entropy(top_choice_counts, total_tokens)

    # Max entropy for reference
    max_entropy = math.log2(num_experts)

    # Top-3 and Bottom-3 overall selections
    sorted_overall = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)
    top_3_overall = sorted_overall[:3]
    bottom_3_overall = sorted_overall[-3:][::-1]  # Reverse to show least used first

    # Top-3 and Bottom-3 first choices
    sorted_first_choice = sorted(
        top_choice_counts.items(), key=lambda x: x[1], reverse=True
    )
    top_3_first_choice = sorted_first_choice[:3]
    bottom_3_first_choice = sorted_first_choice[-3:][::-1]

    # Print results
    print("=" * 70)
    print("MoE ROUTING ANALYSIS")
    print("=" * 70)

    if meta_info:
        print(f"\nModel: {meta_info.get('model_id', 'Unknown')}")
        print(f"Layer: {meta_info.get('layers_logged', ['?'])[0]}")
        print(f"Top-K: {meta_info.get('top_k', '?')}")
        print(f"Seed: {meta_info.get('seed', '?')}")

    print(f"\n{'OVERALL STATISTICS':-^70}")
    print(f"  Total tokens processed: {total_tokens}")
    print(f"  Total expert selections: {total_selections}")
    print(f"  Number of unique experts: {num_experts}")
    print(f"  Expected (uniform): {100.0 / num_experts:.2f}%")

    print(f"\n{'ENTROPY ANALYSIS':-^70}")
    print(f"  Overall selection entropy: {entropy:.3f} bits")
    print(f"  First-choice entropy: {top_choice_entropy:.3f} bits")
    print(f"  Maximum possible entropy: {max_entropy:.3f} bits")
    print(f"  Entropy ratio (overall): {entropy / max_entropy * 100:.1f}%")
    print(
        f"  Entropy ratio (first-choice): {top_choice_entropy / max_entropy * 100:.1f}%"
    )

    print(f"\n{'TOP-3 OVERALL SELECTIONS':-^70}")
    print(f"  {'Rank':<6} {'Expert':<8} {'Count':<12} {'Percentage':<12}")
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12}")
    for i, (expert_id, count) in enumerate(top_3_overall, 1):
        pct = (count / total_selections) * 100
        print(f"  {i:<6} {expert_id:<8} {count}/{total_selections:<6} {pct:>6.2f}%")

    print(f"\n{'BOTTOM-3 OVERALL SELECTIONS':-^70}")
    print(f"  {'Rank':<6} {'Expert':<8} {'Count':<12} {'Percentage':<12}")
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12}")
    for i, (expert_id, count) in enumerate(bottom_3_overall, 1):
        pct = (count / total_selections) * 100
        print(f"  {i:<6} {expert_id:<8} {count}/{total_selections:<6} {pct:>6.2f}%")

    print(f"\n{'TOP-3 FIRST CHOICE (Primary Expert)':-^70}")
    print(
        f"  {'Rank':<6} {'Expert':<8} {'Count':<12} {'Percentage':<12} {'Overall %':<12}"
    )
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for i, (expert_id, count) in enumerate(top_3_first_choice, 1):
        pct = (count / total_tokens) * 100
        overall_pct = (expert_counts[expert_id] / total_selections) * 100
        print(
            f"  {i:<6} {expert_id:<8} {count}/{total_tokens:<7} {pct:>6.2f}%      {overall_pct:>6.2f}%"
        )

    print(f"\n{'BOTTOM-3 FIRST CHOICE (Primary Expert)':-^70}")
    print(
        f"  {'Rank':<6} {'Expert':<8} {'Count':<12} {'Percentage':<12} {'Overall %':<12}"
    )
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for i, (expert_id, count) in enumerate(bottom_3_first_choice, 1):
        pct = (count / total_tokens) * 100
        overall_pct = (expert_counts[expert_id] / total_selections) * 100
        print(
            f"  {i:<6} {expert_id:<8} {count}/{total_tokens:<7} {pct:>6.2f}%      {overall_pct:>6.2f}%"
        )

    print(f"\n{'DISTRIBUTION RANGE':-^70}")
    max_expert = max(expert_counts, key=expert_counts.get)
    min_expert = min(expert_counts, key=expert_counts.get)
    max_pct = (expert_counts[max_expert] / total_selections) * 100
    min_pct = (expert_counts[min_expert] / total_selections) * 100
    print(f"  Overall selections: {min_pct:.2f}% to {max_pct:.2f}%")

    max_first = max(top_choice_counts, key=top_choice_counts.get)
    min_first = min(top_choice_counts, key=top_choice_counts.get)
    max_first_pct = (top_choice_counts[max_first] / total_tokens) * 100
    min_first_pct = (top_choice_counts[min_first] / total_tokens) * 100
    print(f"  First choice: {min_first_pct:.2f}% to {max_first_pct:.2f}%")

    print("=" * 70)

    # Generate markdown report
    output_md = "EXPERT_TRACKING_DELIVERABLES/MOE_ROUTING_ANALYSIS.md"
    with open(output_md, "w") as f:
        f.write("# MoE Routing Analysis\n\n")

        if meta_info:
            f.write(f"**Model:** {meta_info.get('model_id', 'Unknown')}\n\n")
            f.write(f"**Configuration:**\n")
            f.write(f"- Layer: {meta_info.get('layers_logged', ['?'])[0]}\n")
            f.write(f"- Top-K: {meta_info.get('top_k', '?')}\n")
            f.write(f"- Seed: {meta_info.get('seed', '?')}\n")
            f.write(f"- Tokens processed: {total_tokens}\n\n")

        f.write("## Entropy Analysis\n\n")
        f.write(
            f"- Overall selection entropy: **{entropy:.3f} bits** ({entropy / max_entropy * 100:.1f}% of max)\n"
        )
        f.write(
            f"- First-choice entropy: **{top_choice_entropy:.3f} bits** ({top_choice_entropy / max_entropy * 100:.1f}% of max)\n"
        )
        f.write(f"- Maximum possible entropy: {max_entropy:.3f} bits\n\n")

        f.write("## Overall Expert Selections\n\n")
        f.write("### Top-3 Most Used Experts\n\n")
        f.write("| Rank | Expert | Count | Percentage |\n")
        f.write("|------|--------|-------|------------|\n")
        for i, (expert_id, count) in enumerate(top_3_overall, 1):
            pct = (count / total_selections) * 100
            f.write(
                f"| {i} | {expert_id} | {count}/{total_selections} | {pct:.2f}% |\n"
            )

        f.write("\n### Bottom-3 Least Used Experts\n\n")
        f.write("| Rank | Expert | Count | Percentage |\n")
        f.write("|------|--------|-------|------------|\n")
        for i, (expert_id, count) in enumerate(bottom_3_overall, 1):
            pct = (count / total_selections) * 100
            f.write(
                f"| {i} | {expert_id} | {count}/{total_selections} | {pct:.2f}% |\n"
            )

        f.write("\n## First-Choice Expert Preferences\n\n")
        f.write("### Top-3 Primary Experts\n\n")
        f.write("| Rank | Expert | Count | First-Choice % | Overall % |\n")
        f.write("|------|--------|-------|----------------|----------|\n")
        for i, (expert_id, count) in enumerate(top_3_first_choice, 1):
            pct = (count / total_tokens) * 100
            overall_pct = (expert_counts[expert_id] / total_selections) * 100
            f.write(
                f"| {i} | {expert_id} | {count}/{total_tokens} | {pct:.2f}% | {overall_pct:.2f}% |\n"
            )

        f.write("\n### Bottom-3 Primary Experts\n\n")
        f.write("| Rank | Expert | Count | First-Choice % | Overall % |\n")
        f.write("|------|--------|-------|----------------|----------|\n")
        for i, (expert_id, count) in enumerate(bottom_3_first_choice, 1):
            pct = (count / total_tokens) * 100
            overall_pct = (expert_counts[expert_id] / total_selections) * 100
            f.write(
                f"| {i} | {expert_id} | {count}/{total_tokens} | {pct:.2f}% | {overall_pct:.2f}% |\n"
            )

        f.write("\n## Distribution Summary\n\n")
        f.write(f"- Overall selections range: {min_pct:.2f}% to {max_pct:.2f}%\n")
        f.write(f"- First-choice range: {min_first_pct:.2f}% to {max_first_pct:.2f}%\n")
        f.write(f"- Uniform expectation: {100.0 / num_experts:.2f}%\n")

    print(f"\nMarkdown report saved to {output_md}")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    input_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "EXPERT_TRACKING_DELIVERABLES/moe_routes.jsonl"
    )
    analyze_moe_routing(input_file)
