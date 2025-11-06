# IBM Granite Guardian Setup Guide

## Official Implementation

This implementation follows the **official Granite Guardian guide** from `detailed_guide_think.ipynb`.

## Key Updates Based on Official Guide

### ✅ Correct Implementation

**Model**: `ibm-granite/granite-guardian-3.3-8b` (NOT 3.0)

**Method**: 
```python
# Correct (official method)
guardian_config = {"criteria_id": "groundedness"}

chat = tokenizer.apply_chat_template(
    messages,
    guardian_config=guardian_config,
    documents=documents,  # Context as documents
    think=True,           # Enable <think> reasoning
    tokenize=False,
    add_generation_prompt=True
)
```

**Output Parsing**:
```python
# Granite outputs:
# <think>reasoning...</think>
# <score>yes/no</score>

score_match = re.findall(r'<score>(.*?)</score>', response, re.DOTALL)
score = score_match[-1].strip().lower()  # 'yes' or 'no'
```

### ❌ Previous Wrong Implementation

```python
# Wrong: Using as classifier
model = AutoModelForSequenceClassification.from_pretrained(...)

# Wrong: Binary logits
probs = torch.softmax(logits, dim=-1)
hallucination_prob = float(probs[0, 1])
```

**Granite Guardian is NOT a classifier**. It's a causal LM that generates reasoning and score.

## Installation

### Step 1: Install Dependencies

```bash
pip install transformers torch

# Optional but HIGHLY recommended for speed
pip install vllm
```

### Step 2: HuggingFace Login

```bash
huggingface-cli login
# Enter your HF token
```

### Step 3: Verify Model Access

```bash
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-guardian-3.3-8b'); print('✅ Model accessible')"
```

If error, check:
1. HuggingFace account
2. Model license agreement (if required)
3. Access permissions

## Usage

### Quick Test

```bash
cd agora_mi_research

# Run test script
python experiments/test_granite_guardian.py
```

This will:
1. Load Granite Guardian 3.3-8b
2. Test 3 cases (clear hallucination, factual, subtle conflict)
3. Verify output format
4. Show expected results

**Expected output**:
```
Test 1 (Clear hallucination): yes
Test 2 (Factual): no
Test 3 (Subtle conflict): yes
```

### Automated Annotation of 100 Samples

```bash
# Main script
python experiments/phase4_automated_annotation.py
```

**Process**:
1. Loads RAG responses
2. Loads MI analyses (for ECS stratification)
3. Samples 100 responses:
   - 30 low ECS (<0.3) - likely hallucinations
   - 40 medium ECS (0.3-0.5) - mixed
   - 30 high ECS (>=0.5) - likely factual
4. Granite Guardian annotates each (~30 min with vLLM)
5. Saves results to `data/annotations/automated_annotations_granite.json`

**Output format**:
```json
{
  "question_id": "q_0042",
  "question": "...",
  "response": "...",
  "automated_annotation": {
    "is_hallucination": true,
    "score": "yes",
    "reasoning": "The response includes claims about military certification that are not supported by the context...",
    "model": "ibm-granite/granite-guardian-3.3-8b",
    "criteria": "groundedness"
  }
}
```

## Performance

### With vLLM (Recommended)

- **Speed**: 2-3 seconds per response
- **100 samples**: ~5-10 minutes
- **Memory**: ~20GB GPU VRAM
- **Batch processing**: Supported

### Without vLLM (Fallback)

- **Speed**: 10-15 seconds per response
- **100 samples**: ~20-30 minutes
- **Memory**: ~12GB GPU VRAM
- **Sequential processing**: One at a time

## Criteria Explanation

### groundedness (RAG Hallucination)

**Definition**: Response includes claims not supported by or contradicted by context.

**Examples**:

**Yes (Hallucination)**:
- Context: "$500M for AI modernization"
- Response: "$500M for AI and military systems" ← adds "military"

**No (Grounded)**:
- Context: "$500M for AI modernization"
- Response: "The funding is $500 million" ← accurate

**Detection Coverage**:
- ✅ Fabricated facts
- ✅ Contradictions
- ✅ Unsupported extrapolations
- ⚠️ May miss subtle semantic shifts

## Comparison with Manual

### Granite Guardian Strengths

- ✅ Fast (2-3 sec vs 10-15 min per response)
- ✅ Consistent (no annotator fatigue)
- ✅ Scalable (100s of responses)
- ✅ Provides reasoning (interpretable)

### Granite Guardian Limitations

- ⚠️ May miss nuanced misrepresentations
- ⚠️ Domain-specific patterns need validation
- ⚠️ Reasoning quality varies
- ⚠️ Not 100% accurate (expect F1 ~0.75-0.85)

### Combined Strategy

**Best Practice**:
1. **Granite Guardian**: Annotate all 200 responses
2. **Manual validation**: Validate 50 subset
3. **Report both**: 
   - Manual: High quality, limited
   - Automated: Scalable, validated

## Troubleshooting

### Issue: Model not found

```
Error: ibm-granite/granite-guardian-3.3-8b not found
```

**Fix**:
```bash
# Check model exists
https://huggingface.co/ibm-granite/granite-guardian-3.3-8b

# If yes, login
huggingface-cli login

# If no, model not public yet
# Use manual annotation only
```

### Issue: vLLM installation fails

```bash
# vLLM is optional
# Code will fall back to transformers automatically

# If you want vLLM:
pip install vllm --no-build-isolation
```

### Issue: CUDA OOM

```python
# Use smaller model
annotator = GraniteGuardianAnnotator(
    model_variant='2b',  # Use 2B instead of 8B
    use_vllm=False       # Disable vLLM
)

# Or use CPU (very slow)
annotator = GraniteGuardianAnnotator(
    model_variant='2b',
    use_vllm=False,
    device='cpu'
)
```

## Integration with Paper

### Section 3.2: Methodology

```latex
We employ IBM Granite Guardian 3.3-8b with groundedness criteria 
for automated hallucination detection. Granite Guardian generates 
structured reasoning via <think> tags and binary judgments (<score>), 
enabling interpretable automated annotation.
```

### Section 5: Results

```latex
Granite Guardian annotated 100 responses, detecting hallucinations 
in [N] cases ([X]\%). When validated against [M] expert annotations, 
Granite achieved F1=[X.XX], demonstrating [strong/moderate/weak] 
alignment with human judgment.
```

## Summary

### Updated Implementation

- ✅ Model: granite-guardian-3.3-8b (correct version)
- ✅ Method: apply_chat_template with groundedness
- ✅ Parsing: <think> and <score> tags
- ✅ vLLM support for speed
- ✅ Follows official guide exactly

### Ready to Run

```bash
# Test
python experiments/test_granite_guardian.py

# Annotate 100 samples
python experiments/phase4_automated_annotation.py
```

### Time Estimate

- Setup: 10 min (model download)
- Test: 1 min
- 100 annotations: 5-10 min (vLLM) or 20-30 min (transformers)
- Total: 15-40 min

**IBM Granite Guardian가 공식 가이드에 맞춰 정확하게 구현되었습니다!** ✅

