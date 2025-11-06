# Complete Annotation Guide

## Two-Track Annotation Strategy

### Track 1: Manual Annotation (Gold Standard)
**Purpose**: Create high-quality Gold Dataset
**Size**: 50-100 responses
**Time**: 5-10 hours
**Tool**: Label Studio
**Use**: Training, validation, paper results

### Track 2: Automated Annotation (Scaling)
**Purpose**: Scale beyond manual capacity
**Size**: All 200+ responses
**Time**: 30 minutes
**Tool**: IBM Granite Guardian
**Use**: Full dataset analysis, comparison with manual

## Track 1: Manual Annotation (Detailed)

### Setup

See previous message for Label Studio setup. Summary:

```bash
# 1. Install
pip install label-studio

# 2. Start
label-studio start

# 3. Create project
# http://localhost:8080

# 4. Configure labeling interface
# Use XML from src/annotation/interface.py

# 5. Import tasks
# Upload data/annotations/tasks_for_labeling.json
```

### Annotation Protocol

**Time per response**: 10-15 minutes (careful reading required)

**Steps**:
1. Read question
2. Read ALL context segments carefully
3. Read response
4. For each sentence in response:
   - Is it supported by context? → Correct
   - Is it contradicted by context? → Conflict
   - Is it not in context at all? → Baseless
   - Is reasoning flawed? → Reasoning Error
5. Mark spans and assign categories
6. Rate severity and quality
7. Submit

**Quality Standards**:
- Read context twice before annotating
- Check each claim against context
- When in doubt, mark for review
- Be consistent with category definitions

### Gold Dataset Creation

**Selection Criteria**:
```python
# Stratified sample for annotation
sample_for_annotation = {
    'high_ecs': 10,      # ECS > 0.5
    'medium_ecs': 20,    # 0.3 < ECS < 0.5
    'low_ecs': 10,       # ECS < 0.3
    'mixed_task_types': True,
    'diverse_authorities': True
}
```

**Expected Distribution**:
- QA: ~40% (20 responses)
- Data2txt: ~35% (17 responses)
- Summarization: ~25% (13 responses)

## Track 2: Automated Annotation (IBM Granite Guardian)

### Setup

```bash
cd agora_mi_research
```

```python
import sys
sys.path.insert(0, '.')

from src.annotation.granite_guardian import GraniteGuardianAnnotator
import json

# Initialize Granite Guardian
try:
    annotator = GraniteGuardianAnnotator(
        model_variant='rag',  # RAG hallucination detection
        device='cuda',
        confidence_threshold=0.5
    )
    print("✅ Granite Guardian loaded")
    
except Exception as e:
    print(f"❌ Granite Guardian not available: {e}")
    print("Note: Model may not be public yet. Use manual annotation only.")
    exit()

# Load RAG results
with open('outputs/phase2_agora_rag/rag_responses_agora.json', 'r') as f:
    rag_results = json.load(f)

successful = [r for r in rag_results if r.get('success', False)]

print(f"\nAnnotating {len(successful)} responses...")

# Automated annotation
automated_annotations = annotator.annotate_batch(successful)

print(f"\n✅ Automated annotation complete: {len(automated_annotations)} annotated")

# Statistics
hallucinated_count = sum(
    1 for ann in automated_annotations 
    if ann['automated_annotation']['is_hallucination']
)

print(f"\nAutomated Hallucination Rate: {hallucinated_count/len(automated_annotations):.1%}")

# Save
with open('data/annotations/automated_annotations.json', 'w') as f:
    json.dump(automated_annotations, f, indent=2)

print("\nSaved to: data/annotations/automated_annotations.json")
```

### Validation

```python
# After manual annotation is complete
from src.annotation.granite_guardian import GraniteGuardianAnnotator

annotator = GraniteGuardianAnnotator()

# Load both
with open('data/annotations/automated_annotations.json', 'r') as f:
    auto_ann = json.load(f)

with open('data/annotations/parsed_annotations.json', 'r') as f:
    manual_ann = json.load(f)

# Validate
validation = annotator.validate_against_manual(auto_ann, manual_ann)

print("Granite Guardian vs Manual Annotation:")
print(f"  Accuracy: {validation['accuracy']:.3f}")
print(f"  Precision: {validation['precision']:.3f}")
print(f"  Recall: {validation['recall']:.3f}")
print(f"  F1-Score: {validation['f1_score']:.3f}")

# Use in paper:
# "Automated annotation with Granite Guardian achieved 
#  F1=X.XX compared to expert annotations"
```

## Combined Strategy: Manual + Automated

### Workflow

```
1. Manual Annotation (50 responses)
   └─> Gold Dataset
       └─> Training set for analysis
       └─> Validation for Granite Guardian

2. Granite Guardian (200 responses)
   └─> Automated labels for all data
       └─> Compare with manual on overlap
       └─> Use for large-scale patterns

3. Combined Analysis
   └─> Manual: High-quality, limited scale
   └─> Automated: Lower quality, full scale
   └─> Report both in paper
```

### Paper Reporting

**Section 4: Results**

```latex
\subsection{Annotation Results}

\textbf{Manual Annotation.} Expert annotators labeled 50 responses, 
achieving inter-annotator agreement of $\kappa$ = [ACTUAL VALUE]. 
We identified [N] hallucinated responses ([X]\%), with category 
distribution: Evident Baseless Info ([Y]\%), Subtle Conflict ([Z]\%), 
Nuanced Misrepresentation ([W]\%), Reasoning Error ([V]\%).

\textbf{Automated Annotation.} IBM Granite Guardian annotated the full 
dataset of 200 responses. On the 50 manually annotated responses, 
Granite Guardian achieved accuracy = [X.XX], precision = [X.XX], 
recall = [X.XX], F1 = [X.XX], demonstrating [strong/moderate/weak] 
agreement with expert judgments.

This dual-annotation strategy enables: (1) high-quality gold standard 
for MI correlation analysis, and (2) large-scale pattern detection 
across the full dataset.
```

## Practical Considerations

### If Granite Guardian Not Available

**Options**:
1. **Manual only**: Annotate 50-100, report only those
2. **Simple classifier**: Train DistilBERT on manual data
3. **LLM-as-judge**: Use GPT-4 for automated annotation

**Code for Option 2** (Simple classifier):
```python
from transformers import AutoModelForSequenceClassification, Trainer

# Use manual annotations to train simple classifier
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# Train on manual annotations
# Use as "automated" annotator
```

### Time Management

**Efficient annotation**:
- Batch similar task types together
- Take breaks every 10 annotations
- Double-check unclear cases
- Don't rush (quality > speed)

**Scheduling**:
- Week 1: Setup + 20 annotations
- Week 2: 20 annotations
- Week 3: 10 annotations + validation
- Total: 3 weeks at comfortable pace

## Summary

### Manual Annotation
- **Tool**: Label Studio
- **Size**: 50-100 responses
- **Time**: 5-10 hours
- **Quality**: High (gold standard)
- **Use**: Primary evaluation

### Automated Annotation  
- **Tool**: IBM Granite Guardian
- **Size**: All 200+ responses
- **Time**: 30 minutes
- **Quality**: Medium (validated against manual)
- **Use**: Scaling, comparison

### Integration
- Use both in paper
- Report agreement statistics
- Leverage strengths of each approach
- Comprehensive evaluation

이제 Granite Guardian가 완전히 통합되었습니다! 🎯

