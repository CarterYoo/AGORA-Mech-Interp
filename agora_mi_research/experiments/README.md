# Experiments Directory

This directory contains experimental scripts for each phase of the AGORA MI research pipeline.

## Phase Overview

### Phase 1: Data Preparation and Filtering
**Script**: `phase1_data_preparation.py`

Implements stratified sampling of AGORA dataset following RAGTruth methodology.

**Steps**:
1. Load AGORA documents and segments
2. Preprocess and clean text data
3. Perform stratified sampling (50 documents)
4. Export filtered dataset

**Output**: `data/processed/`

### Phase 2: RAG Response Generation
**Script**: `phase2_rag_generation.py`

Generates RAG responses with full MI data collection.

**Steps**:
1. Build ChromaDB vector store
2. Load questions
3. Generate RAG responses with Mistral-7B
4. Capture attention and hidden states

**Output**: `outputs/phase2_rag/`

### Phase 3: Mechanistic Interpretability Analysis
**Script**: `phase3_mi_analysis.py` (to be implemented)

Performs comprehensive MI analysis on RAG responses.

**Steps**:
1. Extract attention patterns
2. Compute ECS (External Context Score)
3. Compute PKS (Parametric Knowledge Score)  
4. Apply Logit Lens analysis
5. Identify copying heads

**Output**: `outputs/phase3_mi/`

### Phase 4: Annotation and Evaluation
**Script**: `phase4_annotation_evaluation.py` (to be implemented)

Creates Gold Dataset through manual annotation and evaluates MI metrics.

**Steps**:
1. Export responses for Label Studio
2. Load manual annotations
3. Correlate MI metrics with hallucinations
4. Compute statistical significance
5. Generate evaluation reports

**Output**: `outputs/phase4_evaluation/`

### Phase 5: Publication Figures
**Script**: `generate_publication_figures.py` (to be implemented)

Generates publication-quality visualizations.

**Figures**:
1. Hallucination example with annotations
2. ECS vs PKS scatter plot
3. ROC curves
4. Attention heatmaps
5. Layer-wise analysis

**Output**: `outputs/figures/`

## Execution Order

Run experiments in sequential order:

```bash
cd experiments

python phase1_data_preparation.py

python phase2_rag_generation.py

python phase3_mi_analysis.py

python phase4_annotation_evaluation.py

python generate_publication_figures.py
```

## Configuration

All experiments use the central configuration file:
```
configs/config.yaml
```

Modify this file to adjust:
- Model parameters
- Data paths
- Hyperparameters
- Output directories

## Logging

All experiments generate logs in `logs/` directory with timestamps.

## Requirements

- Python 3.8+
- CUDA-capable GPU (24GB+ VRAM recommended)
- 32GB system RAM minimum
- All dependencies from `requirements.txt`

## Notes

- Each phase saves intermediate results
- Failed runs can be resumed from last checkpoint
- All experiments use fixed random seed (42) for reproducibility
- GPU memory is automatically managed with 4-bit quantization
