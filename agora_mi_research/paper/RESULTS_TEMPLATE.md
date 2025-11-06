# Results Template for Paper

## IMPORTANT: All values must be filled from ACTUAL experiments

This document provides templates for reporting results. DO NOT use placeholder values.

## Critical Values to Extract from Experiments

### From Phase 2 (RAG Generation)

**Baseline Configuration**:
```python
with open('outputs/phase2_rag/rag_responses.json', 'r') as f:
    baseline_results = json.load(f)

baseline_successful = [r for r in baseline_results if r.get('success', False)]
baseline_total = len(baseline_results)
baseline_success_rate = len(baseline_successful) / baseline_total

print(f"Baseline: {len(baseline_successful)}/{baseline_total} successful ({baseline_success_rate:.1%})")
```

**AGORA ColBERT Configuration**:
```python
with open('outputs/phase2_agora_rag/rag_responses_agora.json', 'r') as f:
    agora_results = json.load(f)

agora_successful = [r for r in agora_results if r.get('success', False)]
agora_total = len(agora_results)
agora_success_rate = len(agora_successful) / agora_total

print(f"AGORA: {len(agora_successful)}/{agora_total} successful ({agora_success_rate:.1%})")
```

### From Phase 3 (MI Analysis)

**Extract ECS values**:
```python
with open('outputs/phase3_mi/mi_analysis_results.json', 'r') as f:
    mi_results = json.load(f)

ecs_values = [mi['ecs_analysis']['overall_ecs'] for mi in mi_results if mi.get('success', False)]

mean_ecs = np.mean(ecs_values)
std_ecs = np.std(ecs_values)
median_ecs = np.median(ecs_values)

print(f"ECS: mean={mean_ecs:.3f}, std={std_ecs:.3f}, median={median_ecs:.3f}")
```

**Extract PKS values**:
```python
pks_values = [mi['pks_analysis']['overall_pks'] for mi in mi_results if mi.get('success', False)]

mean_pks = np.mean(pks_values)
std_pks = np.std(pks_values)

print(f"PKS: mean={mean_pks:.3f}, std={std_pks:.3f}")
```

**Count copying heads**:
```python
copying_heads_counts = [
    mi['ecs_analysis']['num_copying_heads'] 
    for mi in mi_results if mi.get('success', False)
]

mean_copying_heads = np.mean(copying_heads_counts)
std_copying_heads = np.std(copying_heads_counts)

print(f"Copying heads: mean={mean_copying_heads:.1f}, std={std_copying_heads:.1f}")
```

### From Phase 4 (Annotations)

**Hallucination rate**:
```python
with open('data/annotations/parsed_annotations.json', 'r') as f:
    annotations = json.load(f)

total_responses = len(annotations)
hallucinated_responses = sum(
    1 for ann in annotations 
    if len(ann.get('hallucination_spans', [])) > 0
)

hallucination_rate = hallucinated_responses / total_responses

print(f"Hallucination rate: {hallucination_rate:.1%} ({hallucinated_responses}/{total_responses})")
```

**By category**:
```python
category_counts = {
    'Evident Baseless Info': 0,
    'Subtle Conflict': 0,
    'Nuanced Misrepresentation': 0,
    'Reasoning Error': 0
}

for ann in annotations:
    for span in ann.get('hallucination_spans', []):
        for label in span.get('labels', []):
            if label in category_counts:
                category_counts[label] += 1

print("Category distribution:")
for cat, count in category_counts.items():
    print(f"  {cat}: {count}")
```

### From Phase 5 (Statistical Analysis)

**ECS-Hallucination Correlation**:
```python
from src.evaluation.metrics import EvaluationMetrics

metrics = EvaluationMetrics()

ecs_scores = [...]  # From MI analysis
hallucination_labels = [...]  # From annotations

correlation = metrics.compute_point_biserial_correlation(
    ecs_scores, hallucination_labels
)

# USE THESE VALUES IN PAPER:
r_value = correlation['correlation']
p_value = correlation['p_value']

print(f"Correlation: r={r_value:.3f}, p={p_value:.4f}")
```

**Group Comparison**:
```python
from src.evaluation.statistical_tests import StatisticalTests

stat_tests = StatisticalTests()

ecs_no_hall = [ecs for ecs, hall in zip(ecs_scores, hallucination_labels) if not hall]
ecs_hall = [ecs for ecs, hall in zip(ecs_scores, hallucination_labels) if hall]

comparison = stat_tests.compare_groups_comprehensive(
    ecs_no_hall, ecs_hall,
    group1_name="Factual",
    group2_name="Hallucinated"
)

# USE THESE VALUES:
mean_factual = comparison['groups']['Factual']['mean']
mean_hallucinated = comparison['groups']['Hallucinated']['mean']
mean_diff = comparison['groups']['Factual']['mean'] - comparison['groups']['Hallucinated']['mean']
t_stat = comparison['statistical_test']['t_statistic']
p_value = comparison['statistical_test']['p_value']
cohens_d = comparison['effect_size']['cohens_d']

print(f"Factual: {mean_factual:.3f}")
print(f"Hallucinated: {mean_hallucinated:.3f}")
print(f"Difference: {mean_diff:.3f}")
print(f"t={t_stat:.3f}, p={p_value:.4f}, d={cohens_d:.3f}")
```

**ROC Analysis**:
```python
evaluation = metrics.evaluate_hallucination_predictor(
    y_true=hallucination_labels,
    y_scores=[1-ecs for ecs in ecs_scores]  # Invert: lower ECS = higher risk
)

# USE THESE VALUES:
auc = evaluation['roc_analysis']['auc']
optimal_threshold = evaluation['roc_analysis']['optimal_threshold']
optimal_f1 = evaluation['classification_metrics']['f1_score']

print(f"AUC: {auc:.3f}")
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 at optimal: {optimal_f1:.3f}")
```

**ColBERT Comparison**:
```python
# Compare baseline vs ColBERT ECS
with open('outputs/phase3_mi_baseline/mi_analysis_results.json', 'r') as f:
    baseline_mi = json.load(f)

with open('outputs/phase3_mi_agora/mi_analysis_results.json', 'r') as f:
    colbert_mi = json.load(f)

baseline_ecs = [mi['ecs_analysis']['overall_ecs'] for mi in baseline_mi if mi.get('success')]
colbert_ecs = [mi['ecs_analysis']['overall_ecs'] for mi in colbert_mi if mi.get('success')]

mean_baseline_ecs = np.mean(baseline_ecs)
mean_colbert_ecs = np.mean(colbert_ecs)

ecs_improvement = (mean_colbert_ecs - mean_baseline_ecs) / mean_baseline_ecs * 100

comparison = stat_tests.independent_ttest(colbert_ecs, baseline_ecs)

print(f"Baseline ECS: {mean_baseline_ecs:.3f}")
print(f"ColBERT ECS: {mean_colbert_ecs:.3f}")
print(f"Improvement: {ecs_improvement:.1f}%")
print(f"p-value: {comparison['p_value']:.4f}")
```

## How to Fill Paper Results

### Step 1: Run all experiments and save outputs

### Step 2: Extract values using above code snippets

### Step 3: Update main.tex ONLY with actual values

**Example - Abstract**:
```latex
% WRONG (never do this):
ECS exhibits strong negative correlation (r = -0.67, p < 0.001)
% ^ This is FABRICATED DATA if you haven't run experiments

% RIGHT (before experiments):
We analyze the correlation between ECS and hallucination occurrence
% OR:
% [Results to be filled after experimental validation]

% RIGHT (after ACTUAL experiments):
% ONLY if your actual result is r=-0.67:
ECS exhibits strong negative correlation (r = -0.67, p < 0.001)
% If your actual result is r=-0.42:
ECS exhibits moderate negative correlation (r = -0.42, p = 0.003)
% Report what you ACTUALLY found, not what you hoped for
```

### Step 4: Mark clearly what needs data

Use comments:
```latex
% TODO: Fill after Phase 5 statistical analysis
% Expected format: r = X.XX, p < 0.0XX
```

## Verification Checklist

Before claiming any result in the paper:

- [ ] Actual experiment has been run
- [ ] Raw data files exist in outputs/
- [ ] Statistical test has been performed
- [ ] p-value is below significance threshold
- [ ] Effect size is reported
- [ ] Confidence interval is computed
- [ ] Result is reproducible (check with re-run)

## What Should Be In Paper NOW

Only include:
- ✅ Methodology descriptions
- ✅ Experimental design
- ✅ Metric definitions
- ✅ Statistical test procedures
- ✅ Dataset descriptions
- ✅ Architecture descriptions

Do NOT include:
- ❌ Specific numerical results (until experiments done)
- ❌ Correlation values (until computed)
- ❌ p-values (until tests performed)
- ❌ Percentages (until measured)
- ❌ AUC scores (until evaluated)

## Honest Reporting

If experiments show unexpected results:
- Report them honestly
- Discuss why results differ from hypotheses
- Do not cherry-pick or p-hack
- Include negative results

Academic integrity requires reporting actual findings, not desired findings.

