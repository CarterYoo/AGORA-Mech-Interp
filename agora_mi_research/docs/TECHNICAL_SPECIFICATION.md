# Technical Specification

## System Overview

The AGORA Mechanistic Interpretability Research Framework is designed to analyze and understand hallucinations in Retrieval-Augmented Generation (RAG) systems through mechanistic interpretability techniques. This document provides detailed technical specifications for all system components.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     AGORA MI Research                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ Data Layer   │   │  RAG Layer   │   │   MI Layer   │    │
│  │              │   │              │   │              │    │
│  │ - Loader     │──▶│ - Retriever  │──▶│ - Attention  │    │
│  │ - Processor  │   │ - Generator  │   │ - ECS Calc   │    │
│  │ - Sampler    │   │ - Pipeline   │   │ - PKS Calc   │    │
│  └──────────────┘   └──────────────┘   │ - Logit Lens │    │
│                                         └──────────────┘    │
│                                                               │
│  ┌──────────────┐   ┌──────────────┐                        │
│  │ Annotation   │   │ Evaluation   │                        │
│  │              │   │              │                        │
│  │ - Interface  │◀──│ - Metrics    │                        │
│  │ - Validator  │   │ - Stats      │                        │
│  └──────────────┘   └──────────────┘                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Data Layer Specifications

### Data Loader

**Module**: `src/data/loader.py`

**Purpose**: Load AGORA dataset from CSV files.

**Input Specifications**:
- `documents.csv`: Contains 650 AI governance documents
  - Fields: `Document ID`, `Document name`, `Authority`, `Tags`, `Text`, etc.
  - Encoding: UTF-8
  - Size: Approximately 50MB

- `segments.csv`: Contains 5,441 document segments
  - Fields: `Document ID`, `Segment position`, `Text`, etc.
  - Encoding: UTF-8
  - Size: Approximately 100MB

**Output Specifications**:
- pandas DataFrame objects with validated schemas
- Memory footprint: Approximately 500MB

**Error Handling**:
- Missing file errors with clear error messages
- Schema validation with detailed mismatch reporting
- Encoding error fallback to alternative encodings

### Data Preprocessor

**Module**: `src/data/preprocessor.py`

**Purpose**: Clean and normalize text data for downstream processing.

**Operations**:
1. **Text Normalization**
   - Unicode normalization (NFC form)
   - Whitespace standardization
   - Control character removal

2. **Segmentation**
   - Sentence boundary detection
   - Token counting
   - Length statistics computation

3. **Quality Filtering**
   - Minimum length threshold: 50 tokens
   - Maximum length threshold: 2000 tokens
   - Language detection (English only)

**Performance Requirements**:
- Processing speed: >100 documents per second
- Memory efficiency: <2GB for full dataset

### Stratified Sampler

**Module**: `src/data/sampler.py`

**Purpose**: Select balanced subset of documents following RAGTruth methodology.

**Sampling Strategy**:
```
Target: 50 documents from 650 total

Stratification dimensions:
1. Authority distribution (proportional sampling)
   - EU: 35% (18 documents)
   - US: 25% (12 documents)
   - UK: 15% (8 documents)
   - Canada: 15% (7 documents)
   - Others: 10% (5 documents)

2. Policy tag coverage (balanced)
   - Applications: 25%
   - Harms: 25%
   - Risk factors: 25%
   - Strategies: 25%

3. Document length distribution (representative)
   - Short (<1000 tokens): 30%
   - Medium (1000-5000 tokens): 50%
   - Long (>5000 tokens): 20%
```

**Random Seed**: 42 (for reproducibility)

**Output**: Filtered dataset with statistics report

## RAG Layer Specifications

### Semantic Retriever

**Module**: `src/rag/retriever.py`

**Purpose**: Retrieve relevant document segments for question answering.

**Technical Details**:

**Embedding Model**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Pooling: Mean pooling
- Normalization: L2 normalization

**Vector Store**:
- Database: ChromaDB
- Collection: `agora_documents`
- Distance metric: Cosine similarity
- Index type: HNSW

**Retrieval Parameters**:
```python
top_k: 5  # Number of segments to retrieve
similarity_threshold: 0.7  # Minimum cosine similarity
max_segment_length: 1000  # Maximum tokens per segment
```

**Performance Requirements**:
- Query latency: <100ms for top-5 retrieval
- Index building time: <2 minutes for 1247 segments

### Response Generator

**Module**: `src/rag/generator.py`

**Purpose**: Generate responses using retrieved context and language model.

**Model Specifications**:
- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- Parameters: 7 billion
- Quantization: 4-bit (BitsAndBytes NF4)
- Context length: 2048 tokens

**Generation Parameters**:
```python
temperature: 0.7
top_p: 0.9
top_k: 50
max_new_tokens: 512
repetition_penalty: 1.1
```

**Prompt Template**:
```
You are an AI assistant answering questions about AI governance policies.

Context:
BEGIN_CONTEXT
{retrieved_segments}
END_CONTEXT

Question: {question}

Answer based strictly on the provided context. If the context does not contain enough information to answer the question, state this clearly.

Answer:
```

**Context Markers**: `BEGIN_CONTEXT` and `END_CONTEXT` are special tokens for MI analysis to identify context boundaries.

**Hardware Requirements**:
- GPU: NVIDIA GPU with >=24GB VRAM
- CPU RAM: >=16GB
- Storage: 15GB for model weights

### RAG Pipeline

**Module**: `src/rag/pipeline.py`

**Purpose**: Orchestrate complete RAG workflow.

**Pipeline Stages**:
1. Question preprocessing
2. Segment retrieval
3. Context formatting
4. Response generation
5. Post-processing
6. Metadata collection

**Metadata Captured**:
- Question ID
- Retrieved segment IDs
- Retrieval scores
- Generation time
- Token counts
- Context boundaries

## Mechanistic Interpretability Layer

### Attention Analyzer

**Module**: `src/mi/attention_analyzer.py`

**Purpose**: Extract and analyze attention patterns from transformer layers.

**Extraction Specifications**:
- Extract attention tensors from all 32 layers
- Shape: `[batch_size, num_heads, seq_len, seq_len]`
- Number of heads per layer: 32
- Total attention heads: 1024

**Analysis Operations**:
1. **Attention Pattern Visualization**
   - Layer-wise attention heatmaps
   - Head-specific attention distributions
   - Token-to-token attention flows

2. **Statistical Analysis**
   - Mean attention per head
   - Attention entropy
   - Attention sparsity metrics

**Output Format**:
```python
{
    "layer_idx": int,
    "head_idx": int,
    "attention_weights": np.ndarray,  # [seq_len, seq_len]
    "attention_entropy": float,
    "max_attention": float,
    "attention_to_context": float
}
```

### External Context Score Calculator

**Module**: `src/mi/ecs_calculator.py`

**Purpose**: Compute External Context Score following RAGTruth methodology.

**Mathematical Definition**:

For a given token at position `t` in the generated response:

```
ECS(t) = Σ_{i∈Context} Σ_{l=1}^{L} Σ_{h=1}^{H} A_l,h(t, i) / (L × H)

where:
- A_l,h(t, i) = attention weight from token t to context token i at layer l, head h
- L = number of layers (32)
- H = number of heads per layer (32)
- Context = set of tokens between BEGIN_CONTEXT and END_CONTEXT markers
```

**Overall Response ECS**:
```
ECS_response = mean(ECS(t) for all generated tokens t)
```

**Copying Head Identification**:

A head is classified as a "copying head" if:
```
ECS_head > threshold (default: 0.3)
```

**Output Specifications**:
```python
{
    "token_ecs": List[float],  # ECS for each generated token
    "overall_ecs": float,  # Mean ECS across all tokens
    "layer_ecs": List[Dict],  # ECS per layer
    "copying_heads": List[Dict],  # High-ECS heads
    "num_copying_heads": int,
    "context_token_ratio": float  # Proportion of context tokens
}
```

### Parametric Knowledge Score Calculator

**Module**: `src/mi/pks_calculator.py`

**Purpose**: Quantify model's reliance on parametric knowledge.

**Mathematical Definition**:

PKS is computed as the weighted average of three components:

```
PKS = w_conf × Confidence + w_cons × Consistency + w_ent × (1 - Entropy)

where:
- Confidence: max(softmax(logits)) for predicted token
- Consistency: cosine_similarity(hidden_state, previous_hidden_state)
- Entropy: -Σ p(w) log p(w) for token distribution
- w_conf, w_cons, w_ent: weights (default: 0.4, 0.3, 0.3)
```

**Component Definitions**:

1. **Confidence Score**:
   ```
   confidence = max(softmax(final_layer_logits))
   ```
   High confidence suggests strong parametric knowledge.

2. **Consistency Score**:
   ```
   consistency = cosine_similarity(h_t, h_{t-1})
   ```
   High consistency suggests stable internal representations.

3. **Entropy Score**:
   ```
   entropy = -Σ_{w∈V} p(w) log p(w)
   normalized_entropy = entropy / log(|V|)
   low_entropy_score = 1 - normalized_entropy
   ```
   Low entropy suggests confident predictions from internal knowledge.

**Output Specifications**:
```python
{
    "pks_overall": float,  # Overall PKS score
    "confidence_component": float,
    "consistency_component": float,
    "entropy_component": float,
    "token_pks": List[float],  # PKS per generated token
    "layer_pks": List[Dict]  # PKS per layer
}
```

### Logit Lens Implementation

**Module**: `src/mi/logit_lens.py`

**Purpose**: Project hidden states to vocabulary space for layer-wise interpretation.

**Algorithm**:

For each layer `l` from 1 to L:
```
1. Extract hidden state: h_l ∈ R^d
2. Apply layer norm: h_l_norm = LayerNorm(h_l)
3. Project to vocabulary: logits_l = W_unembedding @ h_l_norm
4. Compute probability distribution: p_l = softmax(logits_l)
5. Get top-k predictions: top_k_tokens_l = argsort(p_l)[-k:]
```

**Analysis Metrics**:

1. **Prediction Convergence**:
   - Layer at which top-1 prediction stabilizes
   - Number of prediction changes across layers

2. **Confidence Evolution**:
   - Plot of max(p_l) across layers
   - Identifies when model "decides" on output

3. **Hallucination Indicators**:
   - Late convergence (>layer 25) may indicate hallucination
   - Low final confidence (<0.5) suggests uncertainty
   - Sudden prediction changes indicate instability

**Output Specifications**:
```python
{
    "layer_predictions": List[Dict],  # Top-k predictions per layer
    "convergence_layer": int,  # Layer where top-1 stabilizes
    "confidence_trajectory": List[float],  # Max confidence per layer
    "prediction_changes": int,  # Number of top-1 changes
    "final_confidence": float
}
```

**Visualization**:
- Heatmap: Layer x Top-K predictions
- Line plot: Confidence evolution across layers
- Sankey diagram: Prediction flows between layers

## Annotation Layer

### Annotation Interface

**Module**: `src/annotation/interface.py`

**Purpose**: Integrate with Label Studio for manual hallucination annotation.

**Label Studio Configuration**:
```xml
<View>
  <Header value="Hallucination Annotation - AGORA MI Research"/>
  
  <Text name="question" value="$question"/>
  <Header value="Retrieved Context"/>
  <Text name="context" value="$context"/>
  
  <Header value="Generated Response (annotate hallucination spans)"/>
  <Labels name="hallucination" toName="response">
    <Label value="Evident Baseless Info" background="#FF6B6B"/>
    <Label value="Subtle Conflict" background="#FFA500"/>
    <Label value="Nuanced Misrepresentation" background="#FFD700"/>
    <Label value="Reasoning Error" background="#9370DB"/>
    <Label value="Correct" background="#90EE90"/>
  </Labels>
  <Text name="response" value="$response"/>
  
  <Header value="Additional Metadata"/>
  <Choices name="severity" toName="response" choice="single">
    <Choice value="High"/>
    <Choice value="Medium"/>
    <Choice value="Low"/>
  </Choices>
  
  <TextArea name="notes" toName="response" placeholder="Additional notes"/>
</View>
```

**Data Format**:
```json
{
  "question": "string",
  "context": "string",
  "response": "string",
  "mi_analysis": {
    "overall_ecs": "float",
    "overall_pks": "float",
    "num_copying_heads": "int"
  }
}
```

**Annotation Output**:
```json
{
  "response_id": "string",
  "annotations": [
    {
      "start": "int",
      "end": "int",
      "text": "string",
      "label": "string",
      "severity": "string",
      "annotator_id": "string"
    }
  ],
  "notes": "string",
  "annotation_time": "float"
}
```

### Annotation Validator

**Module**: `src/annotation/validator.py`

**Purpose**: Validate annotation quality and consistency.

**Validation Checks**:

1. **Completeness**:
   - All required fields present
   - No overlapping spans
   - Span boundaries valid

2. **Consistency**:
   - Inter-annotator agreement (Cohen's kappa)
   - Label distribution analysis
   - Severity-label correlation

3. **Quality Metrics**:
   - Annotation time per response
   - Number of annotations per response
   - Annotator-specific statistics

**Quality Thresholds**:
```python
min_inter_annotator_agreement: 0.7  # Cohen's kappa
min_annotation_time: 60  # seconds per response
max_annotation_time: 600  # seconds per response
```

## Evaluation Layer

### Metrics Module

**Module**: `src/evaluation/metrics.py`

**Purpose**: Compute evaluation metrics for hallucination detection.

**Classification Metrics**:

Given hallucination annotations (binary: hallucinated / not hallucinated):

```python
# Confusion matrix
TP: True Positives (correctly identified hallucinations)
TN: True Negatives (correctly identified non-hallucinations)
FP: False Positives (incorrectly flagged as hallucinations)
FN: False Negatives (missed hallucinations)

# Metrics
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Correlation Metrics**:

Correlation between MI metrics and hallucination presence:

```python
# Pearson correlation
r_pearson = pearsonr(ecs_scores, hallucination_binary)

# Spearman correlation (rank-based)
r_spearman = spearmanr(ecs_scores, hallucination_binary)

# Point-biserial correlation (continuous vs binary)
r_pointbiserial = pointbiserialr(ecs_scores, hallucination_binary)
```

**ROC Analysis**:

```python
# Compute ROC curve for ECS as hallucination predictor
fpr, tpr, thresholds = roc_curve(y_true, ecs_scores)
auc = roc_auc_score(y_true, ecs_scores)

# Find optimal threshold (Youden's J statistic)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
```

### Statistical Tests Module

**Module**: `src/evaluation/statistical_tests.py`

**Purpose**: Perform rigorous statistical validation.

**Hypothesis Tests**:

1. **Difference in ECS between hallucinated and non-hallucinated responses**:
   ```python
   # Independent samples t-test
   t_statistic, p_value = ttest_ind(
       ecs_no_hallucination,
       ecs_hallucination,
       equal_var=False  # Welch's t-test
   )
   
   # Effect size (Cohen's d)
   cohens_d = (mean_ecs_no_hall - mean_ecs_hall) / pooled_std
   ```

2. **Correlation significance**:
   ```python
   # Test if correlation is significantly different from zero
   r, p_value = pearsonr(ecs_scores, hallucination_rate)
   ```

3. **Multiple comparison correction**:
   ```python
   # Bonferroni correction
   adjusted_alpha = alpha / num_comparisons
   
   # Benjamini-Hochberg FDR control
   from statsmodels.stats.multitest import multipletests
   reject, p_corrected, _, _ = multipletests(
       p_values, alpha=0.05, method='fdr_bh'
   )
   ```

**Confidence Intervals**:

```python
# Bootstrap confidence intervals
def bootstrap_ci(data, statistic, n_bootstrap=10000, ci=0.95):
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_samples.append(statistic(sample))
    
    lower = np.percentile(bootstrap_samples, (1-ci)/2 * 100)
    upper = np.percentile(bootstrap_samples, (1+ci)/2 * 100)
    return lower, upper
```

## Performance Requirements

### Computational Requirements

**Data Processing**:
- Dataset loading: <5 seconds
- Preprocessing 50 documents: <30 seconds

**RAG Pipeline**:
- Single query retrieval: <100ms
- Single response generation: <10 seconds
- Batch processing (200 questions): <1 hour

**MI Analysis**:
- Attention extraction per response: <5 seconds
- ECS calculation per response: <2 seconds
- PKS calculation per response: <3 seconds
- Logit Lens analysis per response: <5 seconds
- Total MI analysis per response: <15 seconds

### Memory Requirements

**Peak Memory Usage**:
- Data loading: 500MB
- Model loading (4-bit quantized): 6GB GPU memory
- Single inference: 8GB GPU memory
- Batch inference (size 8): 12GB GPU memory
- MI analysis cache: 2GB RAM

### Storage Requirements

- Model weights: 15GB
- AGORA dataset: 150MB
- Processed data: 200MB
- MI analysis results: 500MB per 200 responses
- Annotation data: 50MB
- Total: ~16GB

## Error Handling and Logging

### Error Handling Strategy

**Graceful Degradation**:
- Missing data: Skip with warning, continue processing
- Model errors: Retry with exponential backoff (3 attempts)
- OOM errors: Reduce batch size automatically

**Error Types**:
```python
class DataLoadError(Exception):
    """Raised when data loading fails"""

class ModelInferenceError(Exception):
    """Raised when model inference fails"""

class MIAnalysisError(Exception):
    """Raised when MI analysis fails"""

class AnnotationError(Exception):
    """Raised when annotation validation fails"""
```

### Logging Configuration

```python
# loguru configuration
logger.add(
    "logs/agora_mi_{time}.log",
    rotation="500 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
)
```

**Log Levels**:
- DEBUG: Detailed diagnostic information
- INFO: General progress information
- WARNING: Potential issues (missing data, low scores)
- ERROR: Errors that allow continued execution
- CRITICAL: Fatal errors requiring termination

## Testing Strategy

### Unit Tests

**Coverage Requirements**: >80% code coverage

**Test Categories**:
1. Data processing tests
2. RAG pipeline tests
3. MI analysis tests
4. Annotation tests
5. Evaluation tests

**Example Test Structure**:
```python
def test_ecs_calculator():
    # Test with known attention patterns
    attention = create_test_attention_tensor()
    context_mask = create_test_context_mask()
    
    ecs = calculate_ecs(attention, context_mask)
    
    assert 0 <= ecs <= 1
    assert isinstance(ecs, float)
```

### Integration Tests

Test complete pipeline end-to-end:
```python
def test_full_pipeline():
    # Load test data
    docs = load_test_documents()
    
    # Run RAG
    responses = rag_pipeline.generate(docs)
    
    # Run MI analysis
    mi_results = mi_analyzer.analyze(responses)
    
    # Verify outputs
    assert len(mi_results) == len(responses)
    assert all('ecs' in r for r in mi_results)
```

## Security and Privacy

**Data Handling**:
- No PII collection
- Secure API key storage (environment variables)
- No external data transmission without consent

**Model Security**:
- Model weights verified with checksums
- Sandboxed execution environment
- Resource limits to prevent DoS

## Versioning and Reproducibility

**Version Control**:
- Code: Git with semantic versioning
- Models: HuggingFace model registry
- Data: DVC (Data Version Control)

**Reproducibility**:
- Fixed random seeds (default: 42)
- Deterministic algorithms where possible
- Complete dependency specification (requirements.txt)
- Hardware specification documentation

## API Reference

See `docs/API_REFERENCE.md` for detailed API documentation.

## Changelog

See `CHANGELOG.md` for version history and changes.

