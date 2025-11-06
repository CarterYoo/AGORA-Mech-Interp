# Data Integrity Checklist

## Before Claiming ANY Result in Paper

### Checklist for Each Numerical Claim

For EVERY number in your paper (correlation, p-value, percentage, etc.):

- [ ] **Actual experiment performed**: Corresponding Python script has been executed
- [ ] **Output file exists**: Results saved in `outputs/` directory with timestamp
- [ ] **Raw data available**: Original measurements stored, not just summary statistics
- [ ] **Code committed**: Analysis code is in git repository
- [ ] **Reproducible**: Can re-run and get same result (within random variation)
- [ ] **Verified**: Double-checked the number is correct
- [ ] **Documented**: Noted which script/function generated this number
- [ ] **Peer-reviewed**: If multi-author, co-author verified the analysis

### Example: Claiming Correlation

**Claim**: "ECS shows negative correlation with hallucination (r = -0.67, p < 0.001)"

**Evidence Required**:
1. File exists: `outputs/phase5_evaluation/correlation_analysis.json`
2. Code exists: `experiments/phase5_statistical_analysis.py`
3. Log exists: `logs/phase5_*.log` showing execution
4. Can verify:
   ```python
   results = json.load(open('outputs/phase5_evaluation/correlation_analysis.json'))
   assert abs(results['ecs_hallucination_correlation']['r'] - (-0.67)) < 0.01
   assert results['ecs_hallucination_correlation']['p'] < 0.001
   ```

### Example: Claiming Improvement

**Claim**: "ColBERT increases ECS by 23%"

**Evidence Required**:
1. Baseline results: `outputs/phase3_mi_baseline/mi_analysis_results.json`
2. ColBERT results: `outputs/phase3_mi_agora/mi_analysis_results.json`
3. Comparison code: Function that computed the 23%
4. Statistical test: t-test or equivalent showing significance
5. Can verify:
   ```python
   baseline_ecs = np.mean([...])  # From baseline file
   colbert_ecs = np.mean([...])   # From ColBERT file
   improvement = (colbert_ecs - baseline_ecs) / baseline_ecs * 100
   assert abs(improvement - 23.0) < 1.0  # Within 1%
   ```

## Red Flags

### Suspiciously Perfect Numbers

❌ **AUC = 0.95** (too good, raises suspicion)
❌ **p = 0.0001** (exact powers of 10 are rare)
❌ **Correlation = 0.70** (round numbers suspicious)
❌ **Exactly 50% improvement** (too clean)

Real data is messy:
✅ **AUC = 0.8734**
✅ **p = 0.00347**
✅ **Correlation = 0.6812**
✅ **43.7% improvement**

### Selective Reporting

❌ **Only reporting significant results**
❌ **Not mentioning failed experiments**
❌ **Hiding contradictory evidence**

Honest reporting:
✅ **Report all hypotheses tested**
✅ **Include non-significant results**
✅ **Discuss unexpected findings**

## What To Do If Results Are Bad

### Scenario: Low correlation

You hoped for r > 0.7, but got r = 0.32 (p = 0.15, not significant)

**WRONG Response**:
- Try different metrics until one correlates
- Exclude "outlier" data points
- Don't report it

**RIGHT Response**:
```latex
\section{Results}

Contrary to our hypothesis, we did not find strong correlation between 
ECS and hallucination (r = 0.32, p = 0.15, n = 50). This may be due to:

1. Insufficient sample size (post-hoc power analysis shows power = 0.42)
2. High variance in ECS across different task types
3. Annotation noise (inter-rater kappa = 0.68)

We recommend future work with larger samples to clarify this relationship.
```

This is GOOD science. Negative results are valuable.

## Data Fabrication is Research Misconduct

**Definition**: Making up data or results that don't exist

**Examples**:
- Writing "r = -0.67" without computing it
- Claiming "p < 0.001" without running the test
- Inventing "78% of cases" without counting

**Consequences**:
- Paper retraction
- Loss of credibility
- Career damage
- Violates research ethics

## Our Current Status

### What's in the paper NOW (after corrections):

✅ **Abstract**: General descriptions, no specific numbers
✅ **Introduction**: Hypotheses, not results  
✅ **Results section**: All marked with TODO or TBD
✅ **Discussion**: Conditional framing ("If ColBERT improves...")

### What's NOT in the paper:

✅ No fabricated correlations
✅ No invented p-values
✅ No made-up percentages
✅ No fictional AUC scores

### What you MUST do before filling numbers:

1. **Run Phase 2**: Generate RAG responses
2. **Run Phase 3**: Perform MI analysis
3. **Run Phase 4**: Complete manual annotations
4. **Run Phase 5**: Compute statistics
5. **Extract actual values** from output files
6. **Verify values** are correct
7. **ONLY THEN** update paper with real numbers

## Summary

**Problem Identified**: ✅ Corrected
- Initial draft had example numbers that looked like real results
- All specific values removed
- Replaced with TODO templates and hypotheses

**Current Paper Status**: ✅ Clean
- No fabricated data
- Honest framing
- Ready for real experimental results

**Next Steps**: 
1. Run actual experiments
2. Extract REAL values
3. Fill paper with ACTUAL results
4. Maintain academic integrity

감사합니다 for catching this! 학술적 정직성이 가장 중요합니다. 🎓
