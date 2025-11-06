# AGORA Q&A System Integration Guide

## Overview

This guide explains how to use the official AGORA Q&A system architecture within our MI research framework.

Official AGORA Repository: https://github.com/rrittner1/agora

## Key Differences: AGORA vs Original Implementation

| Component | Original | AGORA Q&A | Status |
|-----------|----------|-----------|--------|
| Retriever | Sentence-Transformers + ChromaDB | RAGatouille + ColBERT | ✅ Integrated |
| Data Format | Simple segments | Segments + Documents with metadata | ✅ Integrated |
| Generator | Vanilla Mistral | Mistral with LangChain | ✅ Compatible |
| DPO Model | N/A | Mentioned in abstract | ⚠️ Not found in code |

## What We Discovered

### AGORA Implementation Details

1. **RAGatouille Library**: AGORA uses `ragatouille` (not direct `colbert-ai`)
   - Simpler API
   - Better maintained
   - Easier integration

2. **Data Structure**:
   ```python
   # AGORA combines segments and documents
   chunks = [
       {
           'id': 'segment_2425_1',
           'type': 'segment',
           'official_name': 'One Big Beautiful Bill Act 2025',
           'text': 'Segment text...',
           'tags': 'Government support',
           'metadata': {...}
       },
       {
           'id': 'document_2425',
           'type': 'document',
           'text': 'Official name - Short summary - Long summary',
           'metadata': {...}
       }
   ]
   ```

3. **ColBERT Indexing**:
   - Indexes formatted strings with metadata
   - Stores content map in pickle file
   - Uses ColBERT v2.0

4. **DPO Model**:
   - **Not implemented yet** in AGORA code
   - Only mentioned in abstract
   - Future work or proprietary

## Setup Instructions

### Step 1: Install RAGatouille

```bash
cd agora_mi_research
pip install ragatouille
```

This will install:
- ColBERT dependencies
- PyTorch (if not already installed)
- Transformers

### Step 2: Verify AGORA Data Access

```bash
# Check data exists
ls "C:\Users\23012\Downloads\agora-master\agora-master\data\agora"

# Should see:
#   documents.csv
#   segments.csv
#   generated_questions_*.json
```

### Step 3: Test AGORA Data Loader

```python
from src.data.agora_loader import AGORADataLoaderOfficial

loader = AGORADataLoaderOfficial(
    data_path="C:/Users/23012/Downloads/agora-master/agora-master/data/agora"
)

documents_df, segments_df = loader.load_all()
print(f"Loaded {len(documents_df)} documents, {len(segments_df)} segments")
```

### Step 4: Run AGORA-Integrated Pipeline

```bash
# Use AGORA architecture
python experiments/phase2_rag_generation_agora.py
```

This will:
1. Load AGORA data in official format
2. Build ColBERT index with RAGatouille
3. Use AGORA official questions
4. Generate responses with full MI data

## Configuration

### Using AGORA Architecture

Edit `configs/config.yaml`:

```yaml
rag:
  retriever:
    type: "ragatouille"  # Use RAGatouille implementation
    
model:
  type: "vanilla"  # DPO not available yet
```

## Data Format Mapping

### AGORA Official → Our Format

| AGORA Column | Our Column | Notes |
|--------------|------------|-------|
| AGORA ID | Document ID | Primary key |
| Official name | Document name | Full name |
| Casual name | - | AGORA-specific |
| Short summary | - | AGORA-specific |
| Long summary | - | AGORA-specific |
| Authority | Authority | Same |
| Collections | - | AGORA grouping |
| Tags | Tags | Semicolon-separated |

### Segments

| AGORA Column | Our Column | Notes |
|--------------|------------|-------|
| Document ID | Document ID | References AGORA ID |
| Segment position | Segment position | Same |
| Text | Text | Same |
| Summary | - | AGORA-specific |
| Tags | Tags | Segment-level tags |
| Non-operative | - | AGORA metadata |
| Segment annotated | - | AGORA metadata |

## Research Questions Enabled

With AGORA integration, we can now study:

### 1. ColBERT vs Sentence-Transformers
- Retrieval precision comparison
- Impact on downstream hallucinations
- ECS differences

### 2. Rich Metadata Analysis
- Do official documents have different hallucination patterns?
- Impact of document collections on retrieval
- Authority-specific analysis

### 3. AGORA Questions
- Use official AGORA question sets
- Multiple question types (compl, def, eval, impl, stakeholder, sum_exp)
- Domain-specific evaluation

## Files Added/Modified

### New Files
- `src/rag/ragatouille_retriever.py`: RAGatouille-based retriever
- `src/data/agora_loader.py`: Official AGORA data loader
- `experiments/phase2_rag_generation_agora.py`: AGORA-integrated pipeline

### Modified Files
- `requirements.txt`: Added ragatouille, langchain
- `configs/config.yaml`: Added ragatouille config
- `README.md`: Updated references

## Execution Examples

### Example 1: Use AGORA Data with RAGatouille

```bash
# Step 1: Navigate to project
cd agora_mi_research

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run AGORA-integrated pipeline
python experiments/phase2_rag_generation_agora.py
```

### Example 2: Use AGORA Questions

```python
import json

# Load AGORA official questions
agora_path = "C:/Users/23012/Downloads/agora-master/agora-master/data/agora"

question_types = [
    'generated_questions_compl.json',  # Compliance questions
    'generated_questions_def.json',    # Definition questions
    'generated_questions_eval.json',   # Evaluation questions
    'generated_questions_impl.json',   # Implementation questions
]

for q_type in question_types:
    with open(f"{agora_path}/{q_type}", 'r') as f:
        questions = json.load(f)
    print(f"{q_type}: {len(questions)} questions")
```

### Example 3: Compare Retrievers

```python
from src.rag.retriever import SemanticRetriever
from src.rag.ragatouille_retriever import RAGatouilleRetriever

query = "What are the AI governance requirements?"

# Original retriever
semantic = SemanticRetriever()
results_semantic = semantic.retrieve(query, top_k=5)

# AGORA retriever
ragatouille = RAGatouilleRetriever()
results_ragatouille = ragatouille.retrieve(query, top_k=5)

# Compare
print(f"Semantic: {[r['similarity'] for r in results_semantic]}")
print(f"ColBERT: {[r['similarity'] for r in results_ragatouille]}")
```

## DPO Status

### Current Status
- **Not implemented in AGORA code** (as of our analysis)
- Mentioned in abstract only
- Likely future work or proprietary

### Options

**Option A: Wait for official DPO model**
- Check AGORA repo for updates
- Contact authors

**Option B: Train your own DPO model**
- Use manual annotations as preference data
- Fine-tune with TRL library

**Option C: Proceed without DPO**
- Focus on ColBERT integration
- Still achieves factually grounded retrieval
- DPO can be added later

## Performance Expectations

### RAGatouille Indexing
- **Time**: 10-30 minutes for full AGORA dataset
- **Memory**: 8-12GB GPU
- **Disk**: 2-5GB index size

### Query Performance
- **Single query**: 50-150ms
- **Batch (200 queries)**: 10-30 seconds
- **Quality**: Higher precision than sentence-transformers

## Troubleshooting

### Issue: RAGatouille installation fails

```bash
# Try installing dependencies first
pip install torch transformers

# Then install RAGatouille
pip install ragatouille
```

### Issue: CUDA out of memory during indexing

```bash
# Reduce batch size in RAGatouille
# Or index on CPU (slower but works)
```

### Issue: Index build is very slow

```bash
# Normal for large datasets
# First time: 20-30 min for 6000+ chunks
# Subsequent: Instant (loads from disk)
```

## Next Steps

1. **Run AGORA-integrated pipeline**:
   ```bash
   python experiments/phase2_rag_generation_agora.py
   ```

2. **Analyze results**:
   - Compare with baseline
   - Check retrieval quality
   - Evaluate MI patterns

3. **Write paper section**:
   - "Integration with AGORA Q&A System"
   - Report retrieval improvements
   - Discuss factual grounding

## References

1. AGORA Q&A System: https://github.com/rrittner1/agora
2. RAGatouille Documentation: https://github.com/bclavie/RAGatouille
3. ColBERT Paper: Khattab & Zaharia (2020), SIGIR
4. Our MI Framework: See `docs/METHODOLOGY.md`

## Contact

For questions about AGORA integration:
- Check AGORA repository issues
- Review our integration documentation
- See logs in `logs/` directory

