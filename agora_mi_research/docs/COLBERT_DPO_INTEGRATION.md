# ColBERT and DPO Integration Guide

## Overview

This document describes the integration of ColBERT retrieval and DPO (Direct Preference Optimization) fine-tuned generation into the AGORA MI Research framework.

## Motivation

Based on the AGORA Q&A system architecture (https://github.com/rrittner1/agora), we are integrating:

1. **ColBERT Retriever**: Token-level semantic matching for more precise evidence retrieval
2. **DPO Fine-tuned Generator**: Human-aligned generation for factually grounded responses

This integration allows us to:
- Improve retrieval precision for policy documents
- Generate more factually accurate responses
- Analyze how human alignment (DPO) affects mechanistic interpretability patterns

## Architecture Comparison

### Original Implementation
```
Question → Sentence-Transformers → ChromaDB → Top-K Segments
                                              ↓
                                      Vanilla Mistral-7B → Response
                                              ↓
                                      MI Analysis (ECS, PKS, Logit Lens)
```

### ColBERT + DPO Integration
```
Question → ColBERT Encoder → ColBERT Index → Top-K Segments
                                            ↓
                                    DPO Fine-tuned Model → Response
                                            ↓
                                    MI Analysis (ECS, PKS, Logit Lens)
                                            ↓
                                    Compare: Vanilla vs DPO Attention Patterns
```

## Implementation Components

### 1. ColBERT Retriever

**Key Features**:
- Token-level contextualized embeddings
- Late interaction mechanism for precise matching
- Higher quality retrieval for technical/legal texts

**Implementation**: `src/rag/colbert_retriever.py`

**Dependencies**:
```bash
pip install colbert-ai
```

**Usage**:
```python
from src.rag.colbert_retriever import ColBERTRetriever

retriever = ColBERTRetriever(
    checkpoint="colbert-ir/colbertv2.0",
    index_name="agora_colbert_index"
)

retriever.index_segments(segments_df)
results = retriever.retrieve(query, top_k=5)
```

### 2. DPO Fine-tuned Generator

**Key Features**:
- Human preference alignment
- Reduced hallucinations
- Improved factual consistency

**Model**: 
- Check `rrittner1/agora` repository for DPO model
- Alternative: Fine-tune using preference data

**Implementation**: Extends existing `ResponseGenerator`

**Usage**:
```python
from src.rag.generator import ResponseGenerator

generator = ResponseGenerator(
    model_name="path/to/dpo-model",
    model_type="dpo_finetuned"
)
```

## Configuration

### config.yaml Extensions

```yaml
retriever:
  type: "colbert"  # or "sentence_transformer"
  colbert:
    checkpoint: "colbert-ir/colbertv2.0"
    index_path: "data/colbert_index"
    nbits: 2  # compression bits
    doc_maxlen: 300
    query_maxlen: 32
  sentence_transformer:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"

model:
  type: "dpo"  # or "vanilla"
  dpo:
    base_model: "mistralai/Mistral-7B-Instruct-v0.2"
    adapter_path: "path/to/dpo-adapter"
    beta: 0.1  # DPO parameter
  vanilla:
    name: "mistralai/Mistral-7B-Instruct-v0.2"
```

## Comparative Analysis

### Research Questions

1. **Retrieval Quality**: Does ColBERT improve retrieval precision for policy documents?
2. **Generation Quality**: Does DPO reduce hallucinations?
3. **MI Patterns**: How does DPO fine-tuning change attention patterns?
4. **ECS/PKS**: Do DPO models rely more on external context (higher ECS)?

### Experimental Design

**Comparison Matrix**:
| Configuration | Retriever | Generator | Purpose |
|--------------|-----------|-----------|---------|
| Baseline | Sentence-Transformers | Vanilla Mistral | Current system |
| ColBERT Only | ColBERT | Vanilla Mistral | Isolate retrieval impact |
| DPO Only | Sentence-Transformers | DPO Model | Isolate generation impact |
| Full Integration | ColBERT | DPO Model | Combined effect |

**Metrics to Compare**:
- Retrieval: Precision@K, Recall@K, MRR
- Generation: Hallucination rate, ROUGE, BERTScore
- MI: ECS, PKS, Copying heads count
- Attention: Pattern changes, context reliance

## Implementation Phases

### Phase 0: Research and Validation (Current)
- ✅ Document integration plan
- ⏳ Research ColBERT library
- ⏳ Locate/validate DPO model
- ⏳ Test basic ColBERT indexing

### Phase 1: ColBERT Integration (2-3 days)
- Implement `ColBERTRetriever` class
- Build ColBERT index for AGORA segments
- Test retrieval quality
- Compare with Sentence-Transformers

### Phase 2: DPO Integration (1-2 days)
- Load DPO fine-tuned model
- Extend `ResponseGenerator` for DPO
- Validate generation quality
- Compare with vanilla model

### Phase 3: Combined System (1 day)
- Integrate ColBERT + DPO pipeline
- Run end-to-end tests
- Benchmark performance

### Phase 4: MI Analysis (2-3 days)
- Run MI analysis on all configurations
- Compare attention patterns
- Analyze ECS/PKS differences
- Generate comparative visualizations

### Phase 5: Documentation (1 day)
- Update user guides
- Write research findings
- Create comparison tables

## Expected Challenges

### ColBERT Challenges
1. **Memory**: Token-level embeddings require more memory
   - Solution: Use nbits=2 compression
   
2. **Indexing Time**: Slower than sentence embeddings
   - Solution: Build index once, reuse
   
3. **GPU Requirements**: ColBERT encoding needs GPU
   - Solution: Batch processing

### DPO Challenges
1. **Model Availability**: DPO model may not be public
   - Solution A: Request from authors
   - Solution B: Fine-tune our own DPO model
   
2. **Preference Data**: Need human preference data for fine-tuning
   - Solution: Use our manual annotations as preference data
   
3. **MI Compatibility**: Ensure DPO model outputs attention/hidden states
   - Solution: Verify model architecture compatibility

## Benefits for MI Research

### Novel Contributions

1. **Human Alignment Analysis**:
   - How does DPO affect attention patterns?
   - Do human-aligned models have higher ECS?
   
2. **Retrieval Quality Impact**:
   - Does better retrieval reduce hallucinations?
   - Correlation between retrieval precision and ECS
   
3. **Factual Grounding**:
   - Mechanism behind factual consistency
   - Role of copying heads in factual generation

### Paper Sections

This integration enables new research questions:

**Section: Impact of Retrieval Quality on MI Patterns**
- Compare ECS with ColBERT vs Sentence-Transformers
- Hypothesis: Better retrieval → Higher ECS → Fewer hallucinations

**Section: Human Alignment and Attention Mechanisms**
- Analyze DPO vs Vanilla attention patterns
- Hypothesis: DPO models have more focused copying heads

**Section: Combined Effect**
- ColBERT + DPO synergy analysis
- Optimal configuration for policy Q&A

## Migration Path

### For Existing Users

The integration is **backward compatible**:

```python
# Option 1: Use new ColBERT retriever
pipeline = RAGPipeline(
    retriever=ColBERTRetriever(...),
    generator=ResponseGenerator(...)
)

# Option 2: Keep existing setup
pipeline = RAGPipeline(
    retriever=SemanticRetriever(...),
    generator=ResponseGenerator(...)
)

# Option 3: Mix and match
pipeline = RAGPipeline(
    retriever=ColBERTRetriever(...),
    generator=ResponseGenerator(model_name="vanilla")
)
```

## References

1. Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR 2020.

2. Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS 2023.

3. AGORA Q&A System: https://github.com/rrittner1/agora

4. Wu, Y., et al. (2024). RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. arXiv:2401.00396.

## Status

Last Updated: 2024-11-05

Current Phase: Phase 0 (Research and Validation)

Next Steps:
1. Install and test python-colbert library
2. Locate DPO model checkpoint
3. Build prototype ColBERT retriever

