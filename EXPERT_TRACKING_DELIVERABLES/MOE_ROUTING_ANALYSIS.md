# MoE Routing Analysis

**Model:** Qwen/Qwen1.5-MoE-A2.7B-Chat

**Configuration:**
- Layer: 0
- Top-K: 4
- Seed: 1234
- Tokens processed: 1424

## Entropy Analysis

- Overall selection entropy: **5.840 bits** (98.9% of max)
- First-choice entropy: **5.542 bits** (93.8% of max)
- Maximum possible entropy: 5.907 bits

## Overall Expert Selections

### Top-3 Most Used Experts

| Rank | Expert | Count | Percentage |
|------|--------|-------|------------|
| 1 | 5 | 152/5696 | 2.67% |
| 2 | 58 | 151/5696 | 2.65% |
| 3 | 59 | 146/5696 | 2.56% |

### Bottom-3 Least Used Experts

| Rank | Expert | Count | Percentage |
|------|--------|-------|------------|
| 1 | 33 | 33/5696 | 0.58% |
| 2 | 29 | 36/5696 | 0.63% |
| 3 | 6 | 38/5696 | 0.67% |

## First-Choice Expert Preferences

### Top-3 Primary Experts

| Rank | Expert | Count | First-Choice % | Overall % |
|------|--------|-------|----------------|----------|
| 1 | 59 | 65/1424 | 4.56% | 2.56% |
| 2 | 18 | 63/1424 | 4.42% | 1.56% |
| 3 | 51 | 62/1424 | 4.35% | 2.26% |

### Bottom-3 Primary Experts

| Rank | Expert | Count | First-Choice % | Overall % |
|------|--------|-------|----------------|----------|
| 1 | 9 | 1/1424 | 0.07% | 0.91% |
| 2 | 29 | 2/1424 | 0.14% | 0.63% |
| 3 | 6 | 3/1424 | 0.21% | 0.67% |

## Distribution Summary

- Overall selections range: 0.58% to 2.67%
- First-choice range: 0.07% to 4.56%
- Uniform expectation: 1.67%
