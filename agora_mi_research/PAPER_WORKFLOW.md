# Complete Paper Workflow: From Data to Publication

## Overview

This document provides the complete end-to-end workflow from data collection to paper submission.

## Phase-by-Phase Workflow

### Phase 0: Setup (30 minutes)

```bash
cd agora_mi_research

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ragatouille import RAGPretrainedModel; print('RAGatouille OK')"
```

### Phase 1: Data Preparation (2 minutes)

```bash
# Copy AGORA data
mkdir -p data/raw/agora
# Copy from: C:\Users\23012\Downloads\agora-master\agora-master\data\agora\
# Or use existing data in parent directory

# Run stratified sampling
python experiments/phase1_data_preparation.py

# Verify output
ls data/processed/
# Should see: filtered_documents.csv, filtered_segments.csv
```

**Paper Section**: Section 3.2 (Dataset)

### Phase 2: RAG Response Generation (1-3 hours)

```bash
# Option A: AGORA architecture (ColBERT)
python experiments/phase2_rag_generation_agora.py

# Option B: Baseline (Sentence-Transformers)
python experiments/phase2_rag_generation.py

# Option C: Compare all configurations
python experiments/compare_configurations.py
```

**Output**: 
- `outputs/phase2_agora_rag/rag_responses_agora.json`
- `outputs/comparison/*.json`

**Paper Sections**: 
- Section 3.4 (Experimental Design)
- Section 4.1 (Experimental Setup)

### Phase 3: MI Analysis (1-3 hours)

**Note**: This script needs to be created based on actual model outputs.

```python
# Create phase3_mi_analysis.py
from src.mi.ecs_calculator import ECSCalculator
from src.mi.pks_calculator import PKSCalculator
from src.mi.logit_lens import LogitLens

# Load RAG responses with attention/hidden states
# Compute ECS, PKS, Logit Lens for each response
# Save results to outputs/phase3_mi/
```

**Output**:
- `outputs/phase3_mi/mi_analysis_results.json`
- `outputs/phase3_mi/mi_summary_statistics.json`

**Paper Sections**:
- Section 3.3 (Mechanistic Interpretability Metrics)
- Section 4.3 (Implementation)

### Phase 4: Manual Annotation (5-10 hours human time)

```bash
# Start Label Studio
label-studio start
```

In Python:
```python
from src.annotation.interface import AnnotationInterface

interface = AnnotationInterface(label_studio_url="http://localhost:8080")

# Load results
import json
with open('outputs/phase2_agora_rag/rag_responses_agora.json', 'r') as f:
    rag_results = json.load(f)

with open('outputs/phase3_mi/mi_analysis_results.json', 'r') as f:
    mi_results = json.load(f)

# Export for annotation
interface.export_for_annotation(
    rag_results,
    mi_results,
    output_path='data/annotations/tasks.json',
    sample_size=50
)
```

Then:
1. Import tasks.json to Label Studio
2. Annotate 50-100 responses
3. Export annotations
4. Save as `data/annotations/annotations_export.json`

**Paper Sections**:
- Section 3.2 (Dataset - Annotations)
- Appendix A.2 (Annotation Protocol)
- Appendix F (Annotation Examples)

### Phase 5: Statistical Analysis (1 hour)

```python
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.statistical_tests import StatisticalTests

metrics = EvaluationMetrics()
stat_tests = StatisticalTests()

# Load data
mi_analyses = ...
annotations = ...

# Extract variables
ecs_scores = [mi['overall_ecs'] for mi in mi_analyses]
hallucination_labels = [len(ann['hallucination_spans']) > 0 for ann in annotations]

# Correlation analysis
correlation = metrics.compute_point_biserial_correlation(ecs_scores, hallucination_labels)
print(f"ECS-Hallucination correlation: r={correlation['correlation']:.3f}, p={correlation['p_value']:.4f}")

# Group comparison
ecs_no_hall = [ecs for ecs, hall in zip(ecs_scores, hallucination_labels) if not hall]
ecs_hall = [ecs for ecs, hall in zip(ecs_scores, hallucination_labels) if hall]

comparison = stat_tests.compare_groups_comprehensive(
    ecs_no_hall, ecs_hall,
    group1_name="Factual",
    group2_name="Hallucinated"
)

print(f"Mean difference: {comparison['groups']['Factual']['mean'] - comparison['groups']['Hallucinated']['mean']:.3f}")
print(f"P-value: {comparison['statistical_test']['p_value']:.4f}")
print(f"Cohen's d: {comparison['effect_size']['cohens_d']:.3f}")

# ROC analysis
evaluation = metrics.evaluate_hallucination_predictor(hallucination_labels, ecs_scores)
print(f"AUC: {evaluation['roc_analysis']['auc']:.3f}")
```

**Paper Sections**:
- Section 3.5 (Statistical Validation)
- Section 5 (Results)
- Appendix G (Statistical Test Details)

### Phase 6: Generate Figures (30 minutes)

```bash
python experiments/generate_publication_figures.py
```

**Output**: `paper/figures/*.pdf`

Figures generated:
1. `figure1_ecs_pks_scatter.pdf`: ECS vs PKS scatter plot
2. `figure2_roc_curve.pdf`: ROC curve for ECS predictor
3. `figure3_layer_wise_ecs.pdf`: Layer-wise ECS comparison
4. `table1_configuration_comparison.tex`: Performance table

**Paper Sections**: All figures in Section 5 (Results)

### Phase 7: Fill Paper Results (2-3 hours)

Edit `paper/main.tex`:

1. **Replace TBD values** with actual results:
   - Table 1: Retrieval metrics (P@5, R@5, MRR)
   - Table 2: Hallucination rates with CI
   - Section 5.3: ECS correlation value
   - Section 5.3: ColBERT ECS improvement
   - Section 5.3: Copying heads statistics
   - Section 5.3: Logit Lens convergence layers
   - Section 5.4: ROC AUC values

2. **Add statistical test results**:
   ```latex
   ECS exhibits strong negative correlation with 
   hallucination (r = -0.67, p < 0.001)
   ```

3. **Update figures**:
   ```latex
   \begin{figure}[t]
   \centering
   \includegraphics[width=0.45\textwidth]{figures/figure1_ecs_pks_scatter.pdf}
   \caption{Scatter plot showing ...}
   \label{fig:scatter}
   \end{figure}
   ```

### Phase 8: Compile Paper (5 minutes)

```bash
cd paper

# Method 1: Using Makefile
make

# Method 2: Manual compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Method 3: Overleaf (recommended)
# Upload all files to Overleaf and compile online
```

**Output**: `paper/main.pdf`

### Phase 9: Review and Polish (1-2 days)

Checklist:
- [ ] All TBD values replaced with actual results
- [ ] All figures included and referenced
- [ ] All tables completed
- [ ] Statistical significance reported correctly
- [ ] References formatted properly
- [ ] Abstract updated with final numbers
- [ ] Limitations section complete
- [ ] Ethical considerations addressed
- [ ] Code/data availability statement
- [ ] Author contributions (if multi-author)
- [ ] Proofread for typos and grammar
- [ ] Check notation consistency
- [ ] Verify mathematical equations
- [ ] Ensure reproducibility information

### Phase 10: Submission

1. **Prepare submission package**:
   - `main.pdf`: Camera-ready paper
   - `supplementary.pdf`: Appendices
   - `code.zip`: Source code
   - `data.zip`: Dataset (if allowed)

2. **Check conference requirements**:
   - ACL 2024: 8 pages + unlimited appendices
   - Anonymization for review
   - Supplementary material limits

3. **Submit via conference portal**

## Results Checklist

### Critical Values Needed

From experiments, extract:

1. **Retrieval Quality** (Table 1):
   - Precision@5: Baseline vs ColBERT
   - Recall@5: Baseline vs ColBERT
   - MRR: Baseline vs ColBERT

2. **Hallucination Rates** (Table 2):
   - Baseline: X% [CI_low, CI_high]
   - ColBERT: Y% [CI_low, CI_high]

3. **ECS Analysis** (Section 5.3):
   - Correlation: r = ?, p = ?
   - Mean ECS (No Hall): ?
   - Mean ECS (Hall): ?
   - t-statistic: ?, p-value: ?
   - Cohen's d: ?

4. **ColBERT Impact** (Section 5.3):
   - Baseline mean ECS: ?
   - ColBERT mean ECS: ?
   - Delta: ?
   - p-value: ?

5. **Copying Heads** (Section 5.3):
   - Number identified: ?
   - Mean ECS of copying heads: ?
   - Hallucination rate with copying heads: ?

6. **Logit Lens** (Section 5.3):
   - Mean convergence (No Hall): ?
   - Mean convergence (Hall): ?
   - t-statistic: ?, p-value: ?

7. **ROC Analysis** (Section 5.4):
   - AUC (ECS alone): ?
   - AUC (PKS alone): ?
   - AUC (Combined): ?
   - Optimal threshold: ?

## Directory Structure After Completion

```
agora_mi_research/
├── data/
│   ├── raw/agora/              # AGORA official data
│   ├── processed/              # Filtered 50 documents
│   └── annotations/            # Manual annotations
│
├── outputs/
│   ├── phase2_agora_rag/       # RAG responses
│   ├── phase3_mi/              # MI analyses
│   ├── phase4_evaluation/      # Statistical results
│   └── comparison/             # Configuration comparison
│
├── paper/
│   ├── main.tex                # Main paper
│   ├── references.bib          # Bibliography
│   ├── appendix.tex            # Appendices
│   ├── figures/                # All figures (PDF + PNG)
│   └── main.pdf                # Final compiled paper
│
├── src/                        # Source code (as documented)
├── experiments/                # Experiment scripts
└── docs/                       # Technical documentation
```

## Quality Assurance

### Before Submission

Run validation scripts:

```python
# Verify all results are filled
grep -r "TBD" paper/main.tex
# Should return: no matches

# Check figure references
grep "includegraphics" paper/main.tex
# Verify all figure files exist

# Verify citations
grep "cite{" paper/main.tex
# Check all keys exist in references.bib
```

### Reproducibility Check

```bash
# Run complete pipeline from scratch
python experiments/phase1_data_preparation.py
python experiments/phase2_rag_generation_agora.py
# ... complete all phases

# Verify results match paper
python experiments/verify_paper_results.py
```

## Timeline Summary

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 0 | Setup | 30 min | - |
| 1 | Data prep | 2 min | AGORA data |
| 2 | RAG generation | 1-3 hours | GPU |
| 3 | MI analysis | 1-3 hours | Phase 2 |
| 4 | Annotation | 5-10 hours | Phase 2 |
| 5 | Statistical analysis | 1 hour | Phases 3,4 |
| 6 | Generate figures | 30 min | Phase 5 |
| 7 | Fill results | 2-3 hours | Phases 1-6 |
| 8 | Compile paper | 5 min | LaTeX |
| 9 | Review & polish | 1-2 days | - |
| **Total** | | **3-5 days** | |

## Support

If you encounter issues:

1. **Data issues**: Check `docs/AGORA_INTEGRATION_GUIDE.md`
2. **Code issues**: Check `docs/API_REFERENCE.md`
3. **Method questions**: Check `docs/METHODOLOGY.md`
4. **LaTeX issues**: Check `paper/README.md`
5. **Results interpretation**: Check `docs/TECHNICAL_SPECIFICATION.md`

## Final Checklist

Before submission:

- [ ] All experiments completed
- [ ] All figures generated
- [ ] All tables filled
- [ ] Statistical tests passed
- [ ] Code released on GitHub
- [ ] Data availability statement
- [ ] Supplementary material prepared
- [ ] Paper proofread
- [ ] References checked
- [ ] Formatting verified
- [ ] Page limit respected
- [ ] Anonymized for review

## Citation

After publication, cite as:

```bibtex
@inproceedings{yourname2024mechanistic,
  title={Mechanistic Interpretability Analysis of Retrieval-Augmented Generation Systems for AI Governance},
  author={Your Name and Collaborators},
  booktitle={Proceedings of ACL},
  year={2024}
}
```

