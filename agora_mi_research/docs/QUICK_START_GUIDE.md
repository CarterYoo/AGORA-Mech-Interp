# Quick Start Guide

## Complete Setup and Execution Guide for AGORA MI Research

This guide provides step-by-step instructions to set up and run the complete research pipeline with ColBERT and DPO integration.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU with 24GB+ VRAM (NVIDIA RTX 3090/4090 recommended)
- 32GB system RAM minimum
- 50GB free disk space

### Check GPU
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## Installation Steps

### Step 1: Navigate to Project

```bash
cd "C:\Users\23012\OneDrive\바탕 화면\Exlpore AGORA\agora_mi_research"
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# This will install:
# - PyTorch and Transformers
# - ColBERT-AI for advanced retrieval
# - PEFT for DPO adapter support
# - ChromaDB for vector storage
# - All analysis and visualization tools
```

Expected installation time: 10-20 minutes

### Step 4: Prepare Data

```bash
# Copy AGORA dataset to data/raw/
mkdir -p data\raw
copy "..\data\documents.csv" "data\raw\"
copy "..\data\segments.csv" "data\raw\"

# Copy existing questions (optional)
mkdir -p outputs\phase1_2
copy "..\outputs\phase1_2\generated_questions.json" "outputs\phase1_2\"
```

## Configuration

### Choose Your Configuration

Edit `configs/config.yaml`:

#### Option 1: Baseline (Default)
```yaml
rag:
  retriever:
    type: "sentence_transformer"

model:
  type: "vanilla"
```

#### Option 2: ColBERT Retrieval
```yaml
rag:
  retriever:
    type: "colbert"  # Enable ColBERT

model:
  type: "vanilla"
```

#### Option 3: DPO Generation
```yaml
rag:
  retriever:
    type: "sentence_transformer"

model:
  type: "dpo"  # Enable DPO
  dpo:
    adapter_path: "path/to/dpo-adapter"  # Set if you have DPO adapter
    use_adapter: true
```

#### Option 4: Full Integration (ColBERT + DPO)
```yaml
rag:
  retriever:
    type: "colbert"

model:
  type: "dpo"
  dpo:
    adapter_path: "path/to/dpo-adapter"
    use_adapter: true
```

## Execution Workflow

### Phase 1: Data Preparation (2 minutes)

```bash
python experiments/phase1_data_preparation.py
```

**Output**:
- `data/processed/filtered_documents.csv` (50 documents)
- `data/processed/filtered_segments.csv` (filtered segments)
- `data/processed/experiment_metadata.json`

**Verify**:
```bash
python -c "import pandas as pd; print(f'Documents: {len(pd.read_csv(\"data/processed/filtered_documents.csv\"))}')"
```

### Phase 2: RAG Response Generation (1-3 hours)

```bash
python experiments/phase2_rag_generation.py
```

**What happens**:
1. Builds vector index (ColBERT or ChromaDB)
   - ColBERT: 10-30 minutes (first time)
   - Sentence-Transformers: 2-5 minutes
2. Loads language model (15GB download first time)
3. Generates 200 responses with full MI data

**Output**:
- `outputs/phase2_rag/rag_responses.json`
- `outputs/phase2_rag/rag_summary_statistics.json`
- `data/chromadb/` or `data/colbert_indexes/`

**Monitor progress**:
```bash
# In another terminal
tail -f logs/phase2_rag_generation_*.log
```

### Phase 3: Compare All Configurations (4-8 hours)

```bash
python experiments/compare_configurations.py
```

**What happens**:
Tests all 4 configurations with same 20 questions:
1. Baseline (Sentence-Transformers + Vanilla)
2. ColBERT Only
3. DPO Only  
4. Full Integration (ColBERT + DPO)

**Output**:
- `outputs/comparison/baseline_results.json`
- `outputs/comparison/colbert_only_results.json`
- `outputs/comparison/dpo_only_results.json`
- `outputs/comparison/full_integration_results.json`
- `outputs/comparison/comparison_summary.json`

**Note**: This requires loading models multiple times. Consider running overnight.

### Phase 4: Manual Annotation (5-10 hours human time)

#### 4.1 Start Label Studio

```bash
pip install label-studio
label-studio start
```

Opens at: http://localhost:8080

#### 4.2 Export Tasks for Annotation

```python
import sys
sys.path.insert(0, '.')

from src.annotation.interface import AnnotationInterface
import json

interface = AnnotationInterface(label_studio_url="http://localhost:8080")

# Load RAG results
with open('outputs/phase2_rag/rag_responses.json', 'r') as f:
    rag_results = json.load(f)

# Export 50 for annotation
interface.export_for_annotation(
    rag_results,
    output_path='data/annotations/tasks.json',
    sample_size=50
)
```

#### 4.3 Import to Label Studio

1. Create new project in Label Studio
2. Import `data/annotations/tasks.json`
3. Annotate hallucination spans
4. Export annotations when complete

#### 4.4 Load and Validate Annotations

```python
from src.annotation.validator import AnnotationValidator

validator = AnnotationValidator()
annotations = interface.load_annotations('annotations_export.json')
parsed = interface.parse_all_annotations(annotations)

# Check quality
stats = interface.compute_annotation_statistics(parsed)
print(f"Hallucination rate: {stats['hallucination_rate']:.2%}")
```

### Phase 5: Statistical Analysis

```python
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.statistical_tests import StatisticalTests

metrics = EvaluationMetrics()
stat_tests = StatisticalTests()

# Correlate ECS with hallucinations
ecs_scores = [mi['overall_ecs'] for mi in mi_results]
hallucination_labels = [len(ann['hallucination_spans']) > 0 for ann in parsed]

evaluation = metrics.evaluate_hallucination_predictor(
    y_true=hallucination_labels,
    y_scores=ecs_scores
)

print(f"AUC: {evaluation['roc_analysis']['auc']:.3f}")
print(f"Correlation: {evaluation['correlation']['correlation']:.3f}")

# Compare groups
ecs_no_hall = [ecs for ecs, hall in zip(ecs_scores, hallucination_labels) if not hall]
ecs_hall = [ecs for ecs, hall in zip(ecs_scores, hallucination_labels) if hall]

comparison = stat_tests.compare_groups_comprehensive(
    ecs_no_hall, ecs_hall,
    group1_name="No Hallucination",
    group2_name="Hallucination"
)

print(f"P-value: {comparison['statistical_test']['p_value']:.4f}")
print(f"Cohen's d: {comparison['effect_size']['cohens_d']:.3f}")
```

## Troubleshooting

### Out of Memory (OOM)

**Problem**: CUDA OOM during generation

**Solutions**:
1. Reduce batch size in code
2. Use smaller model (quantization already enabled)
3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### ColBERT Installation Issues

**Problem**: `pip install colbert-ai` fails

**Solution**:
```bash
# Try with specific version
pip install colbert-ai==0.2.19

# Or install from source
git clone https://github.com/stanford-futuredata/ColBERT.git
cd ColBERT
pip install -e .
```

### DPO Adapter Not Available

**Problem**: No DPO adapter available

**Options**:
1. Train your own using preference data from annotations
2. Use vanilla model only and skip DPO comparison
3. Contact AGORA Q&A authors for model access

### Slow Indexing

**Problem**: ColBERT indexing takes too long

**Solutions**:
1. Reduce `doc_maxlen` in config (e.g., 200 instead of 300)
2. Increase `nbits` compression (2 or 4)
3. Use smaller document set for testing

## Expected Timeline

| Phase | Task | Time (GPU) | Time (No GPU) |
|-------|------|------------|---------------|
| 1 | Data prep | 2 min | 2 min |
| 2 | RAG generation (200q) | 1-2 hours | N/A |
| 3 | Comparison (4 configs) | 4-8 hours | N/A |
| 4 | Manual annotation (50) | 5-10 hours | 5-10 hours |
| 5 | Analysis | 1 hour | 1 hour |
| **Total** | | **~2-3 days** | **N/A** |

## Verification Steps

### After Phase 1
```bash
ls data/processed/
# Should see: filtered_documents.csv, filtered_segments.csv
```

### After Phase 2
```bash
python -c "import json; data=json.load(open('outputs/phase2_rag/rag_responses.json')); print(f'Generated {len(data)} responses')"
```

### After Comparison
```bash
ls outputs/comparison/
# Should see 4 result files + summary
```

## Getting Help

### Check Logs
```bash
# Latest log file
ls -lt logs/ | head -n 1

# View log
cat logs/phase2_rag_generation_*.log
```

### Common Issues

1. **Missing data files**: Ensure AGORA dataset is in `data/raw/`
2. **Import errors**: Check virtual environment is activated
3. **GPU errors**: Verify CUDA installation with `nvidia-smi`
4. **Model download**: First run downloads 15GB, needs good internet

## Next Steps After Completion

1. **Analyze Results**: Compare metrics across configurations
2. **Write Paper**: Document findings in research paper
3. **Publish Code**: Share on GitHub with documentation
4. **Present Findings**: Create slides for presentation

## Support

For issues or questions:
1. Check `docs/TECHNICAL_SPECIFICATION.md` for details
2. Review `docs/METHODOLOGY.md` for research context
3. See `docs/COLBERT_DPO_INTEGRATION.md` for integration specifics
4. Check logs in `logs/` directory

