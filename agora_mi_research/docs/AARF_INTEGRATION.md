# AARF Integration Guide

## Overview

AARF (Add Attention Reduce FFN) is a real-time hallucination mitigation technique from the ReDEeP framework. This guide explains how AARF is integrated into the AGORA MI Research project.

## Background

### What is AARF?

AARF dynamically modifies model behavior during generation to reduce hallucinations:

1. **Add Attention**: Amplifies attention to context tokens in copying heads
2. **Reduce FFN**: Suppresses Feed-Forward Network output to reduce parametric knowledge injection

### When to Use AARF?

AARF intervenes when:
- Hallucination score exceeds threshold (default: 0.6)
- Hallucination score = α × (1 - ECS) + β × PKS
  - Low ECS (high 1-ECS) indicates weak context usage
  - High PKS indicates strong parametric knowledge reliance
  - Both contribute to hallucination risk

## Implementation

### Core Components

#### 1. HallucinationScoreCalculator

Located in `src/mi/aarf_intervention.py`:

```python
from src.mi.aarf_intervention import HallucinationScoreCalculator

calculator = HallucinationScoreCalculator(
    alpha=0.6,  # ECS weight
    beta=0.4,   # PKS weight
    threshold=0.6  # Intervention trigger threshold
)

should_intervene, h_score = calculator.should_intervene(ecs=0.3, pks=0.8)
```

#### 2. AARFIntervention

Main intervention class that applies hooks to transformer layers:

```python
from src.mi.aarf_intervention import AARFIntervention

intervention = AARFIntervention(
    model=model,
    tokenizer=tokenizer,
    copying_heads=copying_heads_list,
    context_start=context_start_pos,
    context_end=context_end_pos,
    attention_multiplier=1.5,  # Amplify copying head attention
    ffn_suppress=0.7  # Reduce FFN output
)

intervention.apply_intervention()
# Generate response
intervention.remove_intervention()
```

#### 3. ResponseGenerator Integration

The `ResponseGenerator` class supports AARF through `generate_with_aarf()`:

```python
from src.rag.generator import ResponseGenerator

generator = ResponseGenerator(
    enable_aarf=True,
    aarf_threshold=0.6,
    aarf_attention_multiplier=1.5,
    aarf_ffn_suppress=0.7
)

result = generator.generate_with_aarf(
    question=question,
    retrieved_segments=segments
)
```

## Usage

### Basic Usage

```python
from src.rag.generator import ResponseGenerator
from src.mi.ecs_calculator import ECSCalculator
from src.mi.pks_calculator import PKSCalculator

generator = ResponseGenerator(enable_aarf=True)
ecs_calc = ECSCalculator()
pks_calc = PKSCalculator()

result = generator.generate_with_aarf(
    question="What are the requirements?",
    retrieved_segments=[...],
    ecs_calculator=ecs_calc,
    pks_calculator=pks_calc
)

if result.get('aarf_intervention_applied'):
    print("AARF intervention was applied")
    print(f"Baseline response: {result['response']}")
    print(f"AARF response: {result['aarf_response']}")
```

### Configuration

Edit `configs/config.yaml`:

```yaml
aarf_intervention:
  enabled: false
  threshold: 0.6
  attention_multiplier: 1.5
  ffn_suppress: 0.7
  intervention_mode: "token_level"
  alpha: 0.6
  beta: 0.4
```

### Running Experiments

#### Phase 3.3: AARF Intervention

```bash
cd agora_mi_research
python experiments/phase3_3_aarf_intervention.py
```

This script:
1. Loads RAG responses from Phase 2
2. Computes baseline ECS/PKS
3. Applies AARF when threshold exceeded
4. Regenerates responses with AARF
5. Exports results to `outputs/phase3_3_aarf/`

#### Phase 3.3 Evaluation

```bash
python experiments/phase3_3_evaluate_aarf.py
```

This script:
1. Loads AARF intervention results
2. Evaluates baseline and AARF responses with Granite Guardian
3. Computes hallucination rate reduction
4. Performs statistical tests
5. Generates comparison report

## Technical Details

### Intervention Mechanism

AARF uses PyTorch forward hooks to modify model behavior:

1. **Attention Hooks**: Registered on `self_attn` modules
   - Amplifies attention weights for context tokens in copying heads
   - Applied only to heads identified as copying heads

2. **FFN Hooks**: Registered on `mlp` or `feed_forward` modules
   - Suppresses output by multiplying by suppress factor
   - Applied to all layers

### Context Token Identification

Context boundaries are identified from prompt markers:
- `BEGIN_CONTEXT` → `context_start`
- `END_CONTEXT` → `context_end`

These positions are used to:
- Compute ECS (attention to context tokens)
- Identify copying heads
- Apply attention amplification

### Copying Heads Identification

Copying heads are identified by:
1. Computing ECS for each (layer, head) pair
2. Comparing against threshold (default: 0.3)
3. Selecting heads with ECS ≥ threshold

Only copying heads receive attention amplification.

## Results Interpretation

### Output Structure

AARF intervention results include:

```python
{
    'question': "...",
    'response': "...",  # Baseline response
    'aarf_response': "...",  # AARF response (if intervention applied)
    'aarf_analysis': {
        'baseline_ecs': 0.45,
        'baseline_pks': 0.65,
        'hallucination_score': 0.68,
        'should_intervene': True,
        'copying_heads': [...]
    },
    'aarf_intervention_applied': True,
    'aarf_stats': {
        'attention_modifications': 150,
        'ffn_suppressions': 32,
        'total_tokens': 50
    }
}
```

### Evaluation Metrics

Phase 3.3 evaluation provides:

- **Hallucination Rate**: Fraction of responses with hallucinations
- **Baseline Rate**: Hallucination rate without AARF
- **AARF Rate**: Hallucination rate with AARF
- **Improvement**: Relative reduction in hallucination rate
- **Statistical Tests**: Chi-square test, Cohen's d effect size

## Limitations

1. **Computational Overhead**: Requires baseline generation + regeneration
2. **Hook Management**: Hooks must be properly removed to avoid memory leaks
3. **Context Boundaries**: Requires accurate context marker identification
4. **Copying Heads**: Effectiveness depends on correct copying head identification

## Troubleshooting

### Intervention Not Applied

- Check hallucination score: `result['aarf_analysis']['hallucination_score']`
- Verify threshold: `config['aarf_intervention']['threshold']`
- Ensure copying heads identified: `len(result['aarf_analysis']['copying_heads']) > 0`

### Low Improvement

- Adjust `attention_multiplier` (increase for stronger effect)
- Adjust `ffn_suppress` (decrease for stronger suppression)
- Verify copying heads are correctly identified
- Check context boundaries are accurate

### Memory Issues

- Ensure hooks are removed: `intervention.remove_intervention()`
- Use 4-bit quantization: `load_4bit=True`
- Process in smaller batches

## References

- ReDEeP Paper: Sun et al. (2024). "Detecting and Reducing Hallucination via Mechanistic Interpretability"
- ReDEeP Repository: https://github.com/Jeryi-Sun/ReDEeP-ICLR

