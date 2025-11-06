# IBM Granite Guardian Integration Guide

## Overview

IBM Granite Guardian is a safety model designed to detect various risks in LLM outputs, including hallucinations in RAG scenarios. We integrate it for automated hallucination annotation.

## Model Variants

IBM provides several Granite Guardian variants:

1. **granite-guardian-3.0-8b**: Full RAG hallucination detection (recommended)
2. **granite-guardian-3.0-2b**: Smaller, faster version
3. **granite-guardian-hap-38m**: Harm/Abuse/Profanity detection (not for our use case)

Reference: https://research.ibm.com/publications/granite-guardian

## Installation

### Check Model Availability

```bash
# Check if models are public on HuggingFace
huggingface-cli repo info ibm-granite/granite-guardian-3.0-8b

# If not public, you'll see an error
# In that case, you may need to:
# 1. Request access from IBM Research
# 2. Use alternative approach (see below)
```

### Install Dependencies

```bash
pip install transformers torch
```

## Usage

### Automated Annotation (100 samples)

```bash
cd agora_mi_research

# Run automated annotation
python experiments/phase4_automated_annotation.py
```

This will:
1. Load RAG responses
2. Load MI analyses (for ECS-based stratification)
3. Sample 100 responses (stratified: 30% low ECS, 40% medium, 30% high)
4. Annotate with Granite Guardian
5. Save results to `data/annotations/automated_annotations_granite.json`
6. Export for manual validation

**Output**:
```
data/annotations/
├── automated_annotations_granite.json     # Granite annotations
├── automated_annotation_metadata.json     # Statistics
├── tasks_for_manual_validation.json       # For Label Studio
└── tasks_priority_review.json             # Low-confidence cases
```

### Validation Against Manual

```bash
# After completing manual annotation in Label Studio
python experiments/phase4_validate_annotations.py
```

This computes:
- Accuracy, Precision, Recall, F1
- Cohen's kappa
- Disagreement analysis

**Output**:
```
outputs/phase4_validation/
└── validation_results.json
```

## Programmatic Usage

### Basic Detection

```python
from src.annotation.granite_guardian import GraniteGuardianAnnotator

# Initialize
annotator = GraniteGuardianAnnotator(
    model_variant='rag',     # Use RAG-specific variant
    device='cuda',           # Use GPU
    confidence_threshold=0.5 # Threshold for binary classification
)

# Detect hallucination
result = annotator.detect_hallucination(
    question="What are the requirements for AI systems?",
    context="The EU AI Act requires high-risk AI systems to undergo conformity assessment.",
    response="All AI systems must undergo military certification and defense approval."
)

print(f"Hallucination detected: {result['is_hallucination']}")
print(f"Confidence: {result['confidence']:.3f}")
# Output: Hallucination detected: True, Confidence: 0.87
```

### Batch Annotation

```python
# Load RAG results
with open('outputs/phase2_agora_rag/rag_responses_agora.json', 'r') as f:
    rag_results = json.load(f)

# Annotate all
automated_annotations = annotator.annotate_batch(rag_results)

# Statistics
hallucinated = sum(
    1 for ann in automated_annotations
    if ann['automated_annotation']['is_hallucination']
)

print(f"Hallucination rate: {hallucinated/len(automated_annotations):.1%}")
```

### Validation

```python
# Compare with manual annotations
validation = annotator.validate_against_manual(
    automated_annotations,
    manual_annotations
)

print(f"Agreement metrics:")
print(f"  Accuracy: {validation['accuracy']:.3f}")
print(f"  F1-Score: {validation['f1_score']:.3f}")
```

## Integration with Manual Annotation

### Strategy: Manual + Automated

**Workflow**:
1. **Granite Guardian** annotates all 200 responses (30 min)
2. **Identify low-confidence** cases (confidence < 0.7)
3. **Manual review** of low-confidence cases (priority)
4. **Optional**: Manual validation of high-confidence subset
5. **Final dataset**: Combination of validated automated + manual

**Benefits**:
- Fast initial annotation
- Focus manual effort on difficult cases
- Scale to full dataset
- Validate automated quality

### Example Combined Dataset

```python
# Combine automated (high confidence) + manual (low confidence)

high_conf_auto = [
    ann for ann in automated_annotations
    if ann['automated_annotation']['confidence'] >= 0.7
]

# Manual annotations for low confidence
manual_for_low_conf = [
    ann for ann in manual_annotations
    if ann['question_id'] in low_conf_qids
]

# Create final Gold Dataset
gold_dataset = {
    'high_confidence_automated': high_conf_auto,
    'manually_validated': manual_for_low_conf,
    'total_size': len(high_conf_auto) + len(manual_for_low_conf),
    'annotation_strategy': 'hybrid_automated_manual'
}
```

## Expected Performance

Based on IBM's published results, Granite Guardian typically achieves:

- **RAG Hallucination Detection**: F1 ~0.80-0.85
- **Evident Baseless Info**: High precision (~0.90)
- **Subtle Conflicts**: Lower recall (~0.60-0.70)

Your actual results may vary depending on:
- Domain specificity (legal/policy vs general)
- Annotation guidelines
- Hallucination types distribution

## Troubleshooting

### Issue: Model not found

```
Error: ibm-granite/granite-guardian-3.0-8b not found
```

**Solutions**:

**Option A**: Check HuggingFace
```bash
# Search for public Granite models
https://huggingface.co/models?search=granite+guardian
```

**Option B**: Request access
- Contact IBM Research
- Check if model requires license/agreement

**Option C**: Alternative automated annotator
```python
# Use DistilBERT trained on manual annotations
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# Train on your manual annotations
# Use as automated annotator
```

**Option D**: LLM-as-Judge
```python
# Use GPT-4 or other strong LLM for automated annotation
import openai

def detect_with_gpt4(question, context, response):
    prompt = f"""Question: {question}
    
Context: {context}

Response: {response}

Does the response contain information not supported by the context?
Answer with: Yes or No, followed by confidence (0-1)."""
    
    # Call GPT-4
    # Parse response
    # Return annotation
```

### Issue: CUDA out of memory

```python
# Use smaller model variant
annotator = GraniteGuardianAnnotator(
    model_variant='2b',  # Use 2B instead of 8B
    device='cuda'
)

# Or use CPU
annotator = GraniteGuardianAnnotator(
    model_variant='2b',
    device='cpu'  # Slower but no memory limit
)
```

### Issue: Low F1 score

If Granite Guardian F1 < 0.6 against manual:

**Diagnosis**:
1. Check manual annotation quality (inter-rater kappa)
2. Check if domain is too specialized
3. Check annotation guidelines alignment

**Solutions**:
1. Fine-tune Granite on your manual data
2. Use manual annotations only
3. Increase manual annotation sample

## For Paper Reporting

### If Granite Guardian Works Well (F1 > 0.75)

```latex
We employed IBM Granite Guardian for automated annotation of 
the full dataset (N=200), achieving F1=X.XX (precision=X.XX, 
recall=X.XX) when validated against 50 expert annotations. 
This dual-annotation strategy enables large-scale analysis 
while maintaining quality through manual validation.
```

### If Granite Guardian Not Available

```latex
Due to model availability constraints, we rely solely on 
manual expert annotation of 50 responses (Cohen's kappa=X.XX). 
While this limits sample size, it ensures high annotation 
quality for our mechanistic interpretability analysis.
```

### If Using Alternative

```latex
For automated scaling, we trained a DistilBERT classifier 
on 40 manual annotations and evaluated on 10 held-out examples, 
achieving F1=X.XX. This enabled annotation of the full dataset 
while maintaining quality control through manual validation.
```

## Summary

**Granite Guardian Integration**:
- ✅ Code implemented: `src/annotation/granite_guardian.py`
- ✅ Experiment script: `experiments/phase4_automated_annotation.py`
- ✅ Validation script: `experiments/phase4_validate_annotations.py`
- ✅ Documentation: This file
- ✅ Paper integration: main.tex updated

**Execution**:
```bash
# 1. Automated annotation
python experiments/phase4_automated_annotation.py

# 2. Manual validation (Label Studio)
# ...

# 3. Validation
python experiments/phase4_validate_annotations.py
```

**Time**:
- Automated: 30 min (100 responses)
- Manual validation: 2-5 hours (priority cases)
- Total: 3-6 hours (vs 10+ hours fully manual)

IBM Granite Guardian로 효율적인 gold dataset 구축이 가능합니다! 🚀

