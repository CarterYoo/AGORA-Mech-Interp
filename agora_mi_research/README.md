# AGORA Mechanistic Interpretability Research

A research framework for analyzing and mitigating hallucinations in Retrieval-Augmented Generation (RAG) systems using mechanistic interpretability techniques, specifically designed for AI governance and policy documents.

## Overview

This project implements a comprehensive mechanistic interpretability pipeline for detecting and understanding hallucinations in RAG systems. It follows established methodologies from:

- **RAGTruth**: Word-level hallucination annotation framework ([Wu et al., 2024](https://arxiv.org/abs/2401.00396))
- **Logit Lens**: Layer-wise interpretation technique ([nostalgebraist, 2020](https://github.com/arnab-api/Logit-Lens-Interpreting-GPT-2))
- **ReDEeP**: Mechanistic interpretability for hallucination detection ([Sun et al., 2024](https://github.com/Jeryi-Sun/ReDEeP-ICLR))
- **AGORA Q&A**: ColBERT retrieval and DPO fine-tuning for factually grounded generation (https://github.com/rrittner1/agora)

## Project Structure

```
agora_mi_research/
├── data/                     # Dataset storage
│   ├── raw/                  # Original AGORA dataset
│   ├── processed/            # Filtered and preprocessed data
│   └── annotations/          # Manual hallucination annotations
├── src/                      # Source code modules
│   ├── data/                 # Data processing
│   ├── rag/                  # RAG pipeline
│   ├── mi/                   # Mechanistic interpretability
│   ├── annotation/           # Annotation tools
│   ├── evaluation/           # Metrics and statistical tests
│   └── utils/                # Utilities
├── experiments/              # Experimental scripts
├── docs/                     # Technical documentation
├── tests/                    # Unit tests
├── configs/                  # Configuration files
└── outputs/                  # Results and checkpoints
```

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU with at least 24GB VRAM (recommended)
- 32GB system RAM (minimum)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd agora_mi_research
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place the AGORA dataset in `data/raw/`:
```bash
cp <path-to-agora>/documents.csv data/raw/
cp <path-to-agora>/segments.csv data/raw/
```

## Key Features

### Multiple RAG Configurations

The framework supports four configurations for comparative analysis:

1. **Baseline**: Sentence-Transformers + Vanilla Mistral-7B
2. **ColBERT Only**: ColBERT retrieval + Vanilla Mistral-7B
3. **DPO Only**: Sentence-Transformers + DPO fine-tuned model
4. **Full Integration**: ColBERT retrieval + DPO fine-tuned model

This enables research on:
- Impact of retrieval precision on hallucinations
- Effect of human alignment (DPO) on mechanistic patterns
- Optimal configuration for policy Q&A systems

## Usage

### Configuration Selection

Edit `configs/config.yaml` to choose retriever and model type:

```yaml
# Choose retriever type
rag:
  retriever:
    type: "sentence_transformer"  # or "colbert"

# Choose model type
model:
  type: "vanilla"  # or "dpo"
```

### Phase 1: Data Preparation

Filter and prepare the AGORA dataset using stratified sampling:

```bash
python experiments/phase1_data_preparation.py
```

This will:
- Load 650 AGORA documents
- Apply stratified sampling to select 50 documents
- Ensure balanced coverage of policy tags
- Export filtered dataset to `data/processed/`

### Phase 2: RAG Response Generation

Generate RAG responses for analysis:

```bash
python experiments/phase2_rag_generation.py
```

This will:
- Build ChromaDB vector store
- Generate 200-250 questions across task types
- Generate RAG responses with context markers
- Export responses for MI analysis

### Phase 3: Mechanistic Interpretability Analysis

Perform MI analysis on RAG responses:

```bash
python experiments/phase3_mi_analysis.py
```

This will:
- Extract attention patterns and hidden states
- Compute External Context Score (ECS)
- Compute Parametric Knowledge Score (PKS)
- Apply Logit Lens analysis
- Identify copying heads and knowledge FFNs

### Phase 3.3: AARF Intervention

Apply AARF (Add Attention Reduce FFN) intervention for real-time hallucination mitigation:

```bash
python experiments/phase3_3_aarf_intervention.py
```

This will:
- Load RAG responses from Phase 2
- Compute baseline ECS/PKS and hallucination scores
- Apply AARF intervention when threshold exceeded
- Regenerate responses with AARF
- Export comparison results

Evaluate AARF effectiveness:

```bash
python experiments/phase3_3_evaluate_aarf.py
```

This will:
- Compare baseline vs AARF responses using Granite Guardian
- Compute hallucination rate reduction
- Perform statistical tests (chi-square, Cohen's d)
- Generate evaluation report

### Phase 4: Annotation and Evaluation

Create Gold Dataset and evaluate:

```bash
python experiments/phase4_annotation_evaluation.py
```

This will:
- Set up Label Studio annotation interface
- Load manual annotations
- Correlate MI metrics with hallucinations
- Perform statistical validation

### Phase 5: Generate Publication Figures

Create publication-quality visualizations:

```bash
python experiments/generate_publication_figures.py
```

### Configuration Comparison

Compare all four configurations:

```bash
python experiments/compare_configurations.py
```

This will:
- Test all configurations with same questions
- Generate responses and collect MI data
- Save results for comparative analysis
- Enable research on impact of ColBERT and DPO

## Methodology

### RAGTruth Framework

We follow the RAGTruth annotation schema with four hallucination categories:

1. **Evident Baseless Info**: Clear fabrications without source support
2. **Subtle Conflict**: Minor inconsistencies with retrieved context
3. **Nuanced Misrepresentation**: Subtle distortions of source information
4. **Reasoning Error**: Incorrect logical inferences from context

### Mechanistic Interpretability Metrics

#### External Context Score (ECS)

Measures the proportion of attention mass directed to retrieved context tokens:

```
ECS = Σ(attention_to_context_tokens) / Σ(total_attention)
```

High ECS indicates strong reliance on external context, while low ECS suggests parametric knowledge usage.

#### Parametric Knowledge Score (PKS)

Quantifies model's reliance on internal knowledge:

```
PKS = (confidence_score + consistency_score + entropy_score) / 3
```

High PKS indicates strong parametric knowledge influence.

#### Logit Lens Analysis

Projects hidden states at each layer to vocabulary space to track prediction evolution:

```
logits_layer_i = W_unembedding @ hidden_state_layer_i
```

Reveals when and how the model "decides" on output tokens.

## Configuration

Edit `configs/config.yaml` to customize:

- Model parameters (temperature, max_length, etc.)
- RAG retriever settings (top_k, similarity threshold)
- MI analysis parameters (ECS threshold, copying heads)
- Annotation categories and Gold Dataset size

## Results

All experimental results are saved to `outputs/`:

- `outputs/phase1/`: Filtered dataset and statistics
- `outputs/phase2/`: RAG responses and metadata
- `outputs/phase3/`: MI analysis results
- `outputs/phase4/`: Annotations and evaluation metrics
- `outputs/figures/`: Publication-quality visualizations

## Citation

If you use this code in your research, please cite:

```bibtex
@software{NA,
  title={NA},
  author={NA},
  year={2025},
  note={NA}
}
```

## References

1. Wu, Y., et al. (2024). RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. arXiv:2401.00396.
2. nostalgebraist. (2020). Interpreting GPT: The Logit Lens. [GitHub Repository](https://github.com/arnab-api/Logit-Lens-Interpreting-GPT-2)
3. Sun, J., et al. (2024). ReDEeP: Detecting and Reducing Hallucination in Large Language Models via Mechanistic Interpretability. ICLR 2024.
4. Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR 2020.
5. Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS 2023.
6. Rittner et al. (2024). Factually Grounded and Human-Aligned RAG Systems for AI Governance and Policy. [GitHub Repository](https://github.com/rrittner1/agora)

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions or collaboration opportunities, please open an issue on the repository.



