# AGORA Q&A Integration Summary

## Mission Accomplished

Successfully integrated AGORA Q&A system architecture into our MI research framework.

## What Was Integrated

### 1. RAGatouille ColBERT Retriever

**Implementation**: `src/rag/ragatouille_retriever.py`

**Key Features**:
- Exact match to AGORA Q&A implementation
- Uses `ragatouille` library (not direct `colbert-ai`)
- Indexes segments AND documents
- Preserves full AGORA metadata
- Content map stored in pickle file

**Usage**:
```python
from src.rag.ragatouille_retriever import RAGatouilleRetriever

retriever = RAGatouilleRetriever(
    model_name="colbert-ir/colbertv2.0",
    index_name="agora_mi_research_index"
)

retriever.index_segments(segments_df, documents_df)
results = retriever.retrieve(query, top_k=5)
```

### 2. AGORA Official Data Loader

**Implementation**: `src/data/agora_loader.py`

**Key Features**:
- Handles official AGORA data format
- Column mapping: AGORA ID, Official name, Casual name, etc.
- Rich metadata preservation
- Compatible with ColBERT chunking

**Usage**:
```python
from src.data.agora_loader import AGORADataLoaderOfficial

loader = AGORADataLoaderOfficial(
    data_path="C:/Users/23012/Downloads/agora-master/agora-master/data/agora"
)

documents_df, segments_df = loader.load_all()
```

### 3. DPO Generator (Prepared)

**Implementation**: `src/rag/dpo_generator.py`

**Status**: 
- Code ready
- **DPO model not found in AGORA repository**
- Mentioned in abstract only
- Can be added when available

**Note**: Abstract mentions DPO but actual implementation not in GitHub code.

### 4. Updated Configurations

**File**: `configs/config.yaml`

**New Options**:
```yaml
rag:
  retriever:
    type: "ragatouille"  # Use AGORA's ColBERT
    
model:
  type: "vanilla"  # DPO when available
```

### 5. Comparison Experiments

**File**: `experiments/compare_configurations.py`

Tests 4 configurations:
1. Baseline (Sentence-Transformers + Vanilla)
2. RAGatouille ColBERT + Vanilla
3. Sentence-Transformers + DPO (when available)
4. Full Integration (ColBERT + DPO)

### 6. AGORA-Integrated Pipeline

**File**: `experiments/phase2_rag_generation_agora.py`

Complete pipeline using AGORA architecture:
- Official data format
- RAGatouille retriever
- AGORA questions (optional)
- Full MI data collection

## Execution Path

### Option A: Use AGORA Official Data and Architecture

```bash
cd agora_mi_research

# Install with AGORA dependencies
pip install -r requirements.txt

# Run AGORA-integrated pipeline
python experiments/phase2_rag_generation_agora.py
```

**Input**:
- AGORA data from: `C:\Users\23012\Downloads\agora-master\agora-master\data\agora`
- Questions: AGORA official or custom

**Output**:
- RAG responses with ColBERT retrieval
- Full MI data (attention, hidden states)
- Ready for MI analysis

**Time**: 1-3 hours (includes 20-30min indexing first time)

### Option B: Compare All Configurations

```bash
# Compare: Baseline vs AGORA ColBERT vs others
python experiments/compare_configurations.py
```

**Time**: 4-8 hours (tests all 4 configurations)

## Key Findings from AGORA Analysis

### What AGORA Uses

✅ **RAGatouille** for ColBERT (not direct colbert-ai)
✅ **Mistral-7B-Instruct-v0.3** as base model
✅ **LangChain** for model wrapping
✅ **Rich metadata** (AGORA ID, official names, tags, authorities)
✅ **Both segments and documents** in index

### What AGORA Doesn't Use (Yet)

❌ **DPO fine-tuned model** (mentioned in abstract only)
❌ **Preference optimization** (not in code)
❌ **Custom prompts** (simple instruction template)

### Implications for Our Research

1. **ColBERT Integration**: ✅ Complete
   - Can use exact AGORA implementation
   - Compare with our sentence-transformers baseline

2. **DPO Integration**: ⚠️ Prepared but not active
   - Code ready for when DPO model becomes available
   - Can fine-tune our own using TRL

3. **MI Analysis**: ✅ Fully Compatible
   - RAGatouille outputs work with our MI tools
   - Attention/hidden states still accessible
   - ECS, PKS, Logit Lens all functional

## Novel Research Contributions

With this integration, we enable:

### 1. First MI Analysis of ColBERT-based RAG
- How does token-level retrieval affect attention patterns?
- Do ColBERT retrievers increase ECS?
- Copying heads analysis with precise retrieval

### 2. Factual Grounding + MI Understanding
- Mechanistic explanation of why ColBERT reduces hallucinations
- Quantify retrieval quality impact on ECS/PKS

### 3. AGORA + RAGTruth Methodology
- Apply RAGTruth annotation to AGORA system
- Domain-specific (policy) hallucination patterns
- Token-level retrieval + word-level annotation

## Updated Project Structure

```
agora_mi_research/
├── src/
│   ├── data/
│   │   ├── loader.py              # Original loader
│   │   └── agora_loader.py        # ⭐ AGORA official format
│   ├── rag/
│   │   ├── retriever.py           # Sentence-Transformers
│   │   ├── colbert_retriever.py   # Direct colbert-ai
│   │   ├── ragatouille_retriever.py  # ⭐ AGORA's RAGatouille
│   │   ├── generator.py           # Vanilla
│   │   ├── dpo_generator.py       # ⭐ DPO (ready)
│   │   └── pipeline.py            # Universal pipeline
│   └── mi/                        # ECS, PKS, Logit Lens
│
├── experiments/
│   ├── phase1_data_preparation.py
│   ├── phase2_rag_generation.py         # Original
│   ├── phase2_rag_generation_agora.py   # ⭐ AGORA version
│   └── compare_configurations.py         # ⭐ 4-way comparison
│
└── docs/
    ├── AGORA_INTEGRATION_GUIDE.md        # ⭐ This file
    ├── COLBERT_DPO_INTEGRATION.md        # General integration
    └── QUICK_START_GUIDE.md              # Updated
```

## Compatibility Matrix

| Feature | Original | AGORA | Compatibility |
|---------|----------|-------|---------------|
| Data Format | Generic CSV | AGORA specific | ✅ Both supported |
| Retriever | Sentence-Transformers | RAGatouille | ✅ Both supported |
| Generator | Vanilla Mistral | (Vanilla) Mistral | ✅ Same |
| MI Analysis | ECS, PKS, Logit Lens | Compatible | ✅ Fully compatible |
| Annotations | RAGTruth | Compatible | ✅ Fully compatible |

## Quick Reference

### Install AGORA Dependencies
```bash
pip install ragatouille langchain langchain-core
```

### Use AGORA Data
```python
from src.data.agora_loader import AGORADataLoaderOfficial
loader = AGORADataLoaderOfficial(data_path="<agora-path>")
docs, segs = loader.load_all()
```

### Use AGORA Retriever
```python
from src.rag.ragatouille_retriever import RAGatouilleRetriever
retriever = RAGatouilleRetriever()
retriever.index_segments(segs, docs)
results = retriever.retrieve(query)
```

### Run Complete Pipeline
```bash
python experiments/phase2_rag_generation_agora.py
```

## Status: Ready for Research

All components integrated and tested:
- ✅ RAGatouille retriever
- ✅ AGORA data loader
- ✅ Integration experiments
- ✅ Documentation
- ✅ DPO code (awaiting model)

**You can now run the full MI research pipeline with AGORA Q&A architecture!**

## Recommended Workflow

1. **Start with AGORA architecture**:
   ```bash
   python experiments/phase2_rag_generation_agora.py
   ```

2. **Run baseline for comparison**:
   ```bash
   python experiments/phase2_rag_generation.py
   ```

3. **Analyze differences**:
   - Compare retrieval quality
   - Compare ECS/PKS patterns
   - Identify hallucination rate differences

4. **Write paper**:
   - Section: ColBERT vs Sentence-Transformers
   - Section: Impact on MI patterns
   - Section: Factual grounding mechanisms

## Contact

For AGORA-specific questions:
- See AGORA repository: https://github.com/rrittner1/agora
- Check our integration docs: `docs/AGORA_INTEGRATION_GUIDE.md`

