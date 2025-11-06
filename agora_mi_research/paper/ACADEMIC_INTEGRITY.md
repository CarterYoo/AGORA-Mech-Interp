# Academic Integrity Guidelines

## Core Principle

**Report only actual experimental results. Never fabricate or embellish data.**

## What Happened

In initial paper draft (`main.tex`), I included example numerical values:
- r = -0.67 (correlation)
- AUC = 0.91
- 23% improvement
- 31% reduction
- 78% of cases
- layer 25 convergence

**These are NOT real experimental results.** They were placeholders/examples.

## Corrections Made

All specific numbers have been replaced with:
- "TODO" comments with templates
- General descriptions
- Hypotheses to be tested
- Placeholders: "X.XX"

## Proper Workflow

### 1. Before Experiments

**Paper should contain**:
- ✅ Hypotheses
- ✅ Methodology
- ✅ Experimental design
- ✅ Planned analyses
- ✅ Metric definitions

**Paper should NOT contain**:
- ❌ Specific results
- ❌ p-values
- ❌ Correlations
- ❌ Effect sizes
- ❌ Percentages

### 2. During Experiments

**Record everything**:
```python
# Save ALL outputs
results = {
    'raw_data': ...,
    'timestamp': ...,
    'config': ...,
    'random_seed': ...
}

with open('experiment_results.json', 'w') as f:
    json.dump(results, f)
```

**Document failures**:
- If experiment fails, document why
- If results are unexpected, investigate
- If hypothesis is wrong, report honestly

### 3. After Experiments

**Extract real values**:
```python
# Example: Computing correlation
from scipy.stats import pearsonr

actual_correlation, actual_p_value = pearsonr(ecs_scores, hallucination_labels)

print(f"ACTUAL correlation: r={actual_correlation:.3f}, p={actual_p_value:.4f}")

# Use ONLY these values in paper
```

**Update paper**:
```latex
% Find TODO comment
% TODO: Fill after Phase 5 statistical analysis

% Replace with ACTUAL result
ECS exhibits strong negative correlation with hallucination 
(Pearson r = -0.67, p < 0.001)
% ONLY if actual_correlation ≈ -0.67 and actual_p_value < 0.001
```

## What If Results Differ from Hypothesis?

### Example Scenario

**Hypothesis**: ColBERT will increase ECS by 20%+

**Actual Result**: ColBERT increases ECS by only 5% (not significant)

### WRONG Response ❌
- Adjust parameters until you get 20%
- Exclude "outliers" to reach significance
- Try different metrics until one works
- Don't report the negative result

### RIGHT Response ✅
```latex
\section{Results}

We hypothesized that ColBERT would substantially increase ECS. 
However, we observed a modest 5\% increase that did not reach 
statistical significance (p = 0.23). This suggests that...

\section{Discussion}

The limited impact of ColBERT on ECS may be due to:
1. Sentence-level embeddings already capturing sufficient context
2. Mistral-7B's strong parametric knowledge
3. AGORA policy documents having similar semantic density

These findings highlight the importance of...
```

## Statistical Integrity

### Multiple Comparisons

If testing N hypotheses, apply correction:

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.03, 0.01, 0.45, 0.02, 0.08]  # From 5 tests
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# Report CORRECTED p-values
```

### Effect Size Reporting

Always report effect sizes, not just p-values:

```latex
% BAD:
The difference was significant (p < 0.05).

% GOOD:
ColBERT showed higher ECS (M = 0.45, SD = 0.12) compared to 
baseline (M = 0.38, SD = 0.15), t(198) = 3.45, p = 0.001, 
Cohen's d = 0.52 (medium effect).
```

### Confidence Intervals

Report CI for all point estimates:

```python
from src.evaluation.statistical_tests import StatisticalTests

stat_tests = StatisticalTests()
ci_result = stat_tests.bootstrap_confidence_interval(
    ecs_scores,
    statistic='mean',
    confidence_level=0.95
)

# Report:
print(f"Mean ECS: {ci_result['point_estimate']:.3f} "
      f"[95% CI: {ci_result['ci_lower']:.3f}, {ci_result['ci_upper']:.3f}]")
```

```latex
Mean ECS: 0.42 [95\% CI: 0.38, 0.46]
```

## Pre-Registration (Optional but Recommended)

Before running experiments, document:

1. **Hypotheses**: What you expect to find
2. **Analyses**: Exactly which tests you'll run
3. **Criteria**: What constitutes "significant"
4. **Sample size**: Power analysis

This prevents p-hacking and selective reporting.

## Reproducibility Requirements

Every result must be reproducible:

```bash
# Someone else should be able to:
git clone your-repo
cd agora_mi_research
pip install -r requirements.txt

# Copy data
# ...

python experiments/phase1_data_preparation.py
python experiments/phase2_rag_generation_agora.py
# ... all phases

# Get SAME results (within random variation)
```

## Reporting Checklist

For each claim in paper:

- [ ] Based on actual experiment
- [ ] Data files exist in outputs/
- [ ] Statistical test performed
- [ ] p-value computed correctly
- [ ] Effect size reported
- [ ] CI computed
- [ ] Multiple comparison corrected (if applicable)
- [ ] Reproducible (documented in code)
- [ ] Negative results reported honestly

## Red Flags to Avoid

❌ **p = 0.049** (suspiciously close to 0.05)
❌ **Excluding data points** without justification
❌ **Trying multiple analyses** until one is significant
❌ **"Trending towards significance"** (p = 0.08)
❌ **Rounding p-values** favorably (0.054 → "< 0.05")
❌ **Cherry-picking results**
❌ **HARKing** (Hypothesizing After Results Known)

## Ethical Research Conduct

1. **Be honest** about what you find
2. **Report failures** and negative results
3. **Document decisions** (why you excluded data, changed methods)
4. **Share code and data** when possible
5. **Acknowledge limitations**
6. **Give credit** to prior work

## Summary

**Current paper status**: 
- ✅ No false claims (all placeholders)
- ✅ Clear TODO markers
- ✅ Honest framing (hypotheses, not results)
- ✅ Ready for real data

**When filling results**:
- Use ONLY actual experimental values
- Include ALL required statistical information
- Report honestly, even if unexpected
- Maintain academic integrity

Science depends on honesty. Report what you find, not what you hope to find.

