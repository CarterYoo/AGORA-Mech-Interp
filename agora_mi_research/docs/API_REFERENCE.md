# API Reference

This document provides a quick reference for key classes and functions in the AGORA MI Research framework.

## Data Processing

### AGORADataLoader

```python
from src.data.loader import AGORADataLoader

loader = AGORADataLoader(data_path="data/raw")
documents_df, segments_df = loader.load_all()
stats = loader.get_statistics()
```

### TextPreprocessor

```python
from src.data.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor(min_length=50, max_length=2000)
processed_df = preprocessor.preprocess_documents(documents_df)
```

### StratifiedSampler

```python
from src.data.sampler import StratifiedSampler

sampler = StratifiedSampler(random_seed=42, target_size=50)
sampled_df, stats = sampler.sample_stratified(documents_df)
```

## RAG Pipeline

### SemanticRetriever

```python
from src.rag.retriever import SemanticRetriever

retriever = SemanticRetriever()
retriever.create_collection()
retriever.index_segments(segments_df)
results = retriever.retrieve(query="What are AI requirements?", top_k=5)
```

### ResponseGenerator

```python
from src.rag.generator import ResponseGenerator

generator = ResponseGenerator(load_4bit=True)
result = generator.generate(
    question="What are the requirements?",
    retrieved_segments=segments
)
```

### RAGPipeline

```python
from src.rag.pipeline import RAGPipeline

pipeline = RAGPipeline(retriever=retriever, generator=generator)
results = pipeline.process_batch(questions, question_ids, task_types)
stats = pipeline.get_statistics(results)
```

## Mechanistic Interpretability

### AttentionAnalyzer

```python
from src.mi.attention_analyzer import AttentionAnalyzer

analyzer = AttentionAnalyzer(num_layers=32, num_heads=32)
stats = analyzer.compute_attention_statistics(attention, token_idx=25)
peaks = analyzer.identify_attention_peaks(attention, layer_idx=15, head_idx=10, token_idx=25)
```

### ECSCalculator

```python
from src.mi.ecs_calculator import ECSCalculator

ecs_calc = ECSCalculator(high_ecs_threshold=0.3)
analysis = ecs_calc.compute_ecs_analysis(
    attention, context_start, context_end, generated_start
)
copying_heads = ecs_calc.identify_copying_heads(attention, context_start, context_end, generated_start)
```

### PKSCalculator

```python
from src.mi.pks_calculator import PKSCalculator

pks_calc = PKSCalculator()
token_pks = pks_calc.compute_token_pks(logits, hidden_current, hidden_previous)
sequence_pks = pks_calc.compute_sequence_pks(logits_sequence, hidden_states_sequence)
p_vs_e = pks_calc.compute_p_vs_e_ratio(pks=0.6, ecs=0.4)
```

### LogitLens

```python
from src.mi.logit_lens import LogitLens

logit_lens = LogitLens(model, tokenizer)
evolution = logit_lens.track_prediction_evolution(hidden_states_by_layer)
indicators = logit_lens.identify_hallucination_indicators(evolution)
```

## Annotation

### AnnotationInterface

```python
from src.annotation.interface import AnnotationInterface

interface = AnnotationInterface(label_studio_url="http://localhost:8080")
interface.export_for_annotation(rag_results, mi_analyses, output_path="tasks.json")
annotations = interface.load_annotations("annotations_export.json")
parsed = interface.parse_all_annotations(annotations)
```

### AnnotationValidator

```python
from src.annotation.validator import AnnotationValidator

validator = AnnotationValidator()
is_complete, issues = validator.validate_annotation_completeness(annotation)
kappa = validator.compute_cohens_kappa(annotations1, annotations2)
agreement = validator.compute_inter_annotator_agreement([annotations1, annotations2])
```

## Evaluation

### EvaluationMetrics

```python
from src.evaluation.metrics import EvaluationMetrics

metrics = EvaluationMetrics()
class_metrics = metrics.compute_classification_metrics(y_true, y_pred)
roc_analysis = metrics.compute_roc_analysis(y_true, y_scores)
correlation = metrics.compute_point_biserial_correlation(scores, labels)
evaluation = metrics.evaluate_hallucination_predictor(y_true, y_scores)
```

### StatisticalTests

```python
from src.evaluation.statistical_tests import StatisticalTests

stat_tests = StatisticalTests(alpha=0.05)
ttest_result = stat_tests.independent_ttest(group1, group2)
ci_result = stat_tests.bootstrap_confidence_interval(data, n_bootstrap=10000)
comparison = stat_tests.compare_groups_comprehensive(group1, group2)
```

## Common Workflows

### Complete RAG Pipeline

```python
from src.data.loader import AGORADataLoader
from src.rag.retriever import SemanticRetriever
from src.rag.generator import ResponseGenerator
from src.rag.pipeline import RAGPipeline

loader = AGORADataLoader(data_path="data/raw")
documents_df, segments_df = loader.load_all()

retriever = SemanticRetriever()
retriever.create_collection()
retriever.index_segments(segments_df)

generator = ResponseGenerator(load_4bit=True)

pipeline = RAGPipeline(retriever=retriever, generator=generator)
results = pipeline.process_batch(questions)
```

### MI Analysis Workflow

```python
from src.mi.ecs_calculator import ECSCalculator
from src.mi.pks_calculator import PKSCalculator
from src.mi.logit_lens import LogitLens

ecs_calc = ECSCalculator()
pks_calc = PKSCalculator()
logit_lens = LogitLens(model, tokenizer)

ecs_analysis = ecs_calc.compute_ecs_analysis(attention, context_start, context_end, generated_start)

sequence_pks = pks_calc.compute_sequence_pks(logits_sequence, hidden_states_sequence)

evolution = logit_lens.track_prediction_evolution(hidden_states_by_layer)
indicators = logit_lens.identify_hallucination_indicators(evolution)
```

### Evaluation Workflow

```python
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.statistical_tests import StatisticalTests

metrics = EvaluationMetrics()
stat_tests = StatisticalTests()

evaluation = metrics.evaluate_hallucination_predictor(y_true, ecs_scores)

ttest_result = stat_tests.independent_ttest(
    ecs_no_hallucination,
    ecs_hallucination
)

ci = stat_tests.bootstrap_confidence_interval(ecs_scores)
```

## Configuration

All experiments use `configs/config.yaml`:

```yaml
random_seed: 42

data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  num_documents: 50

model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
  quantization: "4bit"
  max_length: 2048
  temperature: 0.7

rag:
  retriever:
    top_k: 5
    similarity_threshold: 0.7
  vector_store:
    type: "chromadb"
    collection_name: "agora_documents"

mi_analysis:
  ecs:
    high_ecs_threshold: 0.3
  pks:
    confidence_weight: 0.4
    consistency_weight: 0.3
    entropy_weight: 0.3
```

## Error Handling

All modules use custom exceptions:

```python
from src.data.loader import DataLoadError
from src.rag.generator import ModelInferenceError

try:
    documents_df, segments_df = loader.load_all()
except DataLoadError as e:
    logger.error(f"Failed to load data: {e}")

try:
    result = generator.generate(question, segments)
except ModelInferenceError as e:
    logger.error(f"Generation failed: {e}")
```

## Logging

All modules use loguru for logging:

```python
from loguru import logger

logger.add(
    "logs/experiment_{time}.log",
    rotation="500 MB",
    retention="30 days",
    level="INFO"
)

logger.info("Processing started")
logger.debug("Detailed information")
logger.warning("Potential issue")
logger.error("Error occurred")
```

## For More Information

- **Technical Specification**: See `docs/TECHNICAL_SPECIFICATION.md`
- **Methodology**: See `docs/METHODOLOGY.md`
- **Example Scripts**: See `experiments/` directory
- **README**: See project root `README.md`

