# Methodology

## Research Objective

This research aims to develop and validate mechanistic interpretability techniques for detecting and understanding hallucinations in Retrieval-Augmented Generation (RAG) systems, with specific focus on AI governance and policy documents from the AGORA dataset.

## Research Questions

1. **RQ1**: Can External Context Score (ECS) effectively predict hallucination likelihood in RAG responses?

2. **RQ2**: What is the relationship between Parametric Knowledge Score (PKS) and hallucination occurrence?

3. **RQ3**: Do specific attention heads ("copying heads") consistently attend to retrieved context in factual responses?

4. **RQ4**: Can Logit Lens analysis reveal layer-wise prediction patterns that distinguish hallucinated from factual content?

## Methodological Framework

This research integrates three established methodologies:

### 1. RAGTruth Framework

**Source**: Wu et al. (2024) - "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models"

**Key Contributions**:
- Word-level hallucination annotation schema
- Four-category hallucination taxonomy
- Task-based evaluation (QA, Data-to-text, Summarization)

**Adaptation for AGORA**:
- Applied to AI governance domain (legal/policy texts)
- Focus on high-stakes scenarios where accuracy is critical
- Domain-specific hallucination patterns (e.g., misinterpreting regulatory requirements)

### 2. Logit Lens Technique

**Source**: nostalgebraist (2020) - "Interpreting GPT: The Logit Lens"

**Key Contributions**:
- Layer-wise projection of hidden states to vocabulary space
- Tracking prediction evolution across model depth
- Identifying "decision points" in generation process

**Adaptation for AGORA**:
- Applied to Mistral-7B architecture (32 layers)
- Focused on hallucination-prone tokens
- Correlated with ECS/PKS metrics

### 3. ReDEeP Framework

**Source**: Sun et al. (2024) - "Detecting and Reducing Hallucination via Mechanistic Interpretability"

**Key Contributions**:
- Copying heads identification
- Knowledge FFN analysis
- Parametric vs External knowledge decomposition

**Adaptation for AGORA**:
- ECS calculation for context attribution
- PKS formulation for parametric knowledge quantification
- P vs E ratio as hallucination indicator

## Experimental Design

### Phase 1: Dataset Preparation

**Objective**: Create balanced, representative subset of AGORA corpus.

**Procedure**:

1. **Data Collection**
   - Source: AGORA dataset (650 documents, 5,441 segments)
   - Documents: AI governance policies from multiple jurisdictions
   - Time period: 2020-2024

2. **Stratified Sampling**
   - Sample size: 50 documents
   - Stratification factors:
     - Authority distribution (EU, US, UK, Canada, Others)
     - Policy tags (Applications, Harms, Risk factors, Strategies)
     - Document length (Short, Medium, Long)
   - Random seed: 42

3. **Quality Control**
   - Minimum document length: 50 tokens
   - Language filtering: English only
   - Completeness check: No missing critical fields

**Expected Outcome**: Filtered dataset of 50 documents with balanced representation.

### Phase 2: RAG Response Generation

**Objective**: Generate RAG responses with full attention/hidden state capture.

**Procedure**:

1. **Vector Store Construction**
   - Embedding model: sentence-transformers/all-MiniLM-L6-v2
   - Index: 1,247 document segments
   - Storage: ChromaDB with HNSW indexing

2. **Question Generation**
   - Source: Existing 200 questions from `outputs/phase1_2/generated_questions.json`
   - Task distribution:
     - QA (Question Answering): 40%
     - Data2txt (Data-to-text): 35%
     - Summarization: 25%
   - Complexity: Mixed (simple factual to complex reasoning)

3. **Response Generation**
   - Model: Mistral-7B-Instruct-v0.2 (4-bit quantized)
   - Generation parameters:
     - Temperature: 0.7
     - Top-p: 0.9
     - Max new tokens: 512
   - Context formatting: Use BEGIN_CONTEXT/END_CONTEXT markers
   - Capture: Full attention tensors and hidden states

4. **Metadata Collection**
   - Retrieved segment IDs
   - Retrieval scores
   - Context boundaries (token positions)
   - Generation time
   - Token counts

**Expected Outcome**: 200-250 RAG responses with complete MI data.

### Phase 3: Mechanistic Interpretability Analysis

**Objective**: Extract and compute MI metrics for hallucination detection.

**Procedure**:

#### 3.1 Attention Pattern Extraction

1. **Data Extraction**
   - Extract attention tensors from all 32 layers
   - Shape: [batch_size=1, num_heads=32, seq_len, seq_len]
   - Storage format: HDF5 for efficient access

2. **Attention Statistics**
   - Per-head attention distributions
   - Layer-wise attention patterns
   - Token-to-token attention flows

#### 3.2 External Context Score Calculation

1. **Context Token Identification**
   - Locate BEGIN_CONTEXT and END_CONTEXT markers
   - Extract context token positions
   - Compute context ratio

2. **ECS Computation**
   - For each generated token:
     ```
     ECS(token_t) = (Σ_{layer} Σ_{head} Σ_{i∈context} attention[layer, head, t, i]) / (num_layers × num_heads)
     ```
   - Aggregate across response:
     ```
     ECS_response = mean(ECS(token_t) for all generated tokens)
     ```

3. **Copying Head Identification**
   - Compute ECS per head
   - Threshold: ECS_head > 0.3
   - Rank by ECS score

**Expected Metrics**:
- ECS per token
- ECS per response
- ECS per layer
- List of copying heads with scores

#### 3.3 Parametric Knowledge Score Calculation

1. **Hidden State Extraction**
   - Extract final hidden states for each token
   - Extract intermediate hidden states for consistency

2. **Confidence Component**
   ```
   confidence = max(softmax(final_logits))
   ```

3. **Consistency Component**
   ```
   consistency = cosine_similarity(hidden_state[t], hidden_state[t-1])
   ```

4. **Entropy Component**
   ```
   entropy = -Σ p(token) log(p(token))
   normalized_entropy = entropy / log(vocab_size)
   low_entropy_score = 1 - normalized_entropy
   ```

5. **PKS Aggregation**
   ```
   PKS = 0.4 × confidence + 0.3 × consistency + 0.3 × low_entropy_score
   ```

**Expected Metrics**:
- PKS per token
- PKS per response
- Component scores
- Layer-wise PKS evolution

#### 3.4 Logit Lens Analysis

1. **Layer-wise Projection**
   - For each layer l = 1 to 32:
     - Extract hidden state: h_l
     - Apply layer normalization
     - Project to vocabulary: logits_l = W_unembedding @ h_l_norm
     - Compute top-k predictions

2. **Convergence Analysis**
   - Track when top-1 prediction stabilizes
   - Count prediction changes across layers
   - Compute confidence trajectory

3. **Hallucination Indicators**
   - Late convergence (layer > 25)
   - Low final confidence (< 0.5)
   - Frequent prediction changes (> 5)

**Expected Metrics**:
- Layer-wise top-k predictions
- Convergence layer
- Prediction stability score
- Confidence trajectory

**Expected Outcome**: Complete MI analysis for 200-250 responses.

### Phase 3.3: AARF Intervention

**Objective**: Apply real-time hallucination mitigation using AARF intervention.

**Procedure**:

#### 3.3.1 Hallucination Score Calculation

1. **Score Formula**
   ```
   h_score = α × (1 - ECS) + β × PKS
   where α + β = 1 (default: α=0.6, β=0.4)
   ```

2. **Interpretation**
   - High (1-ECS): Low context usage → hallucination risk
   - High PKS: Strong parametric knowledge → hallucination risk
   - Combined score indicates overall hallucination probability

3. **Intervention Trigger**
   - Threshold: h_score ≥ 0.6 (configurable)
   - When triggered, apply AARF intervention

#### 3.3.2 AARF Intervention Application

1. **Copying Head Amplification**
   - Identify copying heads from ECS analysis
   - Register attention hooks on identified heads
   - Amplify attention weights to context tokens
   - Multiplier: 1.5× (configurable)

2. **FFN Suppression**
   - Register hooks on all FFN layers (mlp/feed_forward)
   - Suppress output by multiplying by suppress factor
   - Suppress factor: 0.7× (configurable)

3. **Regeneration**
   - Regenerate response with intervention active
   - Compare baseline vs AARF responses
   - Track intervention statistics

**Expected Outcomes**:
- Baseline vs AARF response pairs
- Intervention frequency statistics
- Hallucination rate comparison
- Response quality metrics

#### 3.3.3 Evaluation

1. **Granite Guardian Annotation**
   - Evaluate baseline responses
   - Evaluate AARF responses
   - Compare hallucination rates

2. **Statistical Analysis**
   - Chi-square test for hallucination rate difference
   - Cohen's d for effect size
   - Confidence intervals

3. **Quality Metrics**
   - Response length comparison
   - Semantic similarity (if applicable)
   - Intervention overhead

### Phase 4: Annotation and Evaluation

**Objective**: Create Gold Dataset and validate MI metrics as hallucination predictors.

**Procedure**:

#### 4.1 Manual Annotation

1. **Annotation Setup**
   - Platform: Label Studio
   - Annotators: 2-3 expert annotators
   - Training: Practice on 10 examples with feedback
   - Guidelines: Detailed annotation manual

2. **Annotation Categories** (from RAGTruth):

   **Evident Baseless Info**:
   - Definition: Clear fabrications without any support in context
   - Example: "The EU AI Act bans all facial recognition" (when context discusses regulation, not bans)
   - Severity: High

   **Subtle Conflict**:
   - Definition: Minor contradictions with retrieved context
   - Example: "The act applies to all AI systems" (when context specifies high-risk systems)
   - Severity: Medium

   **Nuanced Misrepresentation**:
   - Definition: Subtle distortions of source information
   - Example: Changing "should consider" to "must implement"
   - Severity: Medium

   **Reasoning Error**:
   - Definition: Incorrect logical inferences
   - Example: Concluding cause from correlation in context
   - Severity: Low to Medium

3. **Annotation Process**
   - Sample: 50-100 responses (stratified by task type)
   - Word-level annotation: Mark span, assign category, rate severity
   - Metadata: Annotator ID, time spent, confidence
   - Quality control: Dual annotation for 20% of data

4. **Inter-Annotator Agreement**
   - Metric: Cohen's kappa
   - Target: κ > 0.7
   - Adjudication: Resolve disagreements through discussion

**Expected Outcome**: Gold Dataset of 50-100 annotated responses.

#### 4.2 Statistical Validation

1. **Correlation Analysis**
   
   **Hypothesis 1**: ECS negatively correlates with hallucination rate
   ```
   H0: ρ(ECS, hallucination) = 0
   H1: ρ(ECS, hallucination) < 0
   Test: Pearson correlation with one-tailed test
   Significance level: α = 0.05
   ```

   **Hypothesis 2**: PKS positively correlates with hallucination rate
   ```
   H0: ρ(PKS, hallucination) = 0
   H1: ρ(PKS, hallucination) > 0
   Test: Pearson correlation with one-tailed test
   Significance level: α = 0.05
   ```

2. **Group Comparison**
   
   **Hypothesis 3**: ECS differs between hallucinated and non-hallucinated responses
   ```
   H0: μ_ECS(no_hall) = μ_ECS(hall)
   H1: μ_ECS(no_hall) ≠ μ_ECS(hall)
   Test: Independent samples t-test (Welch's)
   Significance level: α = 0.05
   Effect size: Cohen's d
   ```

3. **Predictive Modeling**
   
   **Model**: Logistic regression
   ```
   P(hallucination | ECS, PKS) = sigmoid(β0 + β1·ECS + β2·PKS + β3·num_copying_heads)
   ```
   
   **Evaluation**:
   - ROC curve and AUC
   - Precision-Recall curve
   - Confusion matrix
   - Cross-validation (5-fold)

4. **Confidence Intervals**
   - Bootstrap method (10,000 iterations)
   - 95% confidence intervals for all metrics
   - Report: point estimate [CI_lower, CI_upper]

5. **Multiple Comparison Correction**
   - Method: Benjamini-Hochberg FDR control
   - Adjusted significance level
   - Report corrected p-values

**Expected Outcome**: Statistical validation of MI metrics as hallucination predictors.

### Phase 5: Analysis and Visualization

**Objective**: Synthesize findings and create publication-ready materials.

**Analyses**:

1. **Descriptive Statistics**
   - Hallucination rate by task type
   - ECS/PKS distributions
   - Copying head statistics

2. **Inferential Statistics**
   - Correlation matrices
   - Group comparisons with effect sizes
   - Regression model results

3. **Pattern Analysis**
   - Layer-wise ECS evolution
   - Logit Lens trajectories for hallucinated vs non-hallucinated
   - Attention pattern visualizations

**Visualizations** (Publication Quality):

1. **Figure 1**: Hallucination Example
   - Side-by-side: Context vs Response
   - Highlighted: Hallucinated spans with categories

2. **Figure 2**: ECS vs PKS Scatter Plot
   - X-axis: ECS, Y-axis: PKS
   - Colors: Hallucinated (red) vs Non-hallucinated (blue)
   - Decision boundary from logistic regression

3. **Figure 3**: ROC Curve
   - Multiple curves: ECS alone, PKS alone, Combined model
   - AUC values annotated
   - Optimal threshold marked

4. **Figure 4**: Attention Heatmap
   - Example response with high/low ECS
   - Layer-wise attention to context vs parametric tokens
   - Copying heads highlighted

5. **Figure 5**: Logit Lens Trajectory
   - Line plots: Confidence evolution across layers
   - Separate plots: Hallucinated vs Non-hallucinated
   - Convergence points marked

6. **Figure 6**: Layer-wise Analysis
   - Bar plots: ECS per layer
   - Grouped by: Hallucinated vs Non-hallucinated
   - Error bars: 95% CI

**Expected Outcome**: Complete analysis with publication-ready figures.

## Evaluation Criteria

### Validity

**Internal Validity**:
- Fixed random seeds for reproducibility
- Blinded annotation (annotators unaware of MI metrics)
- Multiple annotators for reliability
- Systematic data collection procedures

**External Validity**:
- Stratified sampling for representativeness
- Multiple jurisdictions represented
- Various task types included
- Results generalizable to AI governance domain

**Construct Validity**:
- ECS measures context attribution (construct: external knowledge usage)
- PKS measures parametric knowledge (construct: internal knowledge usage)
- Hallucination categories aligned with established taxonomy (RAGTruth)

**Statistical Conclusion Validity**:
- Adequate sample size (power analysis)
- Appropriate statistical tests
- Multiple comparison corrections
- Effect sizes reported
- Confidence intervals provided

### Reliability

**Inter-Rater Reliability**:
- Cohen's kappa > 0.7
- Percentage agreement > 80%
- Adjudication protocol for disagreements

**Test-Retest Reliability**:
- Consistent MI metric computation
- Deterministic algorithms (fixed seeds)
- Verification with repeated runs

**Internal Consistency**:
- PKS components should correlate
- ECS across layers should show patterns
- Multiple indicators of same construct should agree

### Limitations

1. **Sample Size**:
   - Limited to 50 documents (computational constraints)
   - Gold Dataset: 50-100 responses (annotation cost)
   - May limit generalizability

2. **Domain Specificity**:
   - Focused on AI governance/policy domain
   - Results may not generalize to other domains (medical, scientific, etc.)
   - Legal language has unique characteristics

3. **Model Specificity**:
   - Analysis limited to Mistral-7B
   - Results may differ for other architectures (Llama, GPT, etc.)
   - Findings may not hold for larger models (70B+)

4. **Annotation Subjectivity**:
   - Hallucination boundaries can be ambiguous
   - Severity ratings subjective
   - Category assignment sometimes unclear

5. **Computational Constraints**:
   - 4-bit quantization may affect results
   - Limited to single GPU inference
   - Batch size restrictions

6. **Temporal Validity**:
   - AGORA documents from 2020-2024
   - Rapidly evolving policy landscape
   - Results may become dated

### Mitigation Strategies

1. **For sample size**:
   - Maximize within-sample diversity through stratification
   - Report confidence intervals to acknowledge uncertainty
   - Frame conclusions appropriately

2. **For domain specificity**:
   - Clearly state domain scope in conclusions
   - Discuss domain-specific characteristics
   - Suggest cross-domain validation as future work

3. **For model specificity**:
   - Document model architecture and parameters
   - Compare with related work on other models
   - Discuss architectural generalizability

4. **For annotation subjectivity**:
   - Detailed annotation guidelines
   - Training and calibration sessions
   - Multiple annotators with disagreement resolution
   - Report inter-rater reliability

5. **For computational constraints**:
   - Document hardware specifications
   - Compare quantized vs full-precision on subset
   - Acknowledge as limitation

## Ethical Considerations

### Data Ethics

**Source Data**:
- AGORA dataset is publicly available
- No personal information in policy documents
- Proper attribution and citation

**Generated Data**:
- RAG responses are synthetic (model-generated)
- No privacy concerns
- Hallucinations flagged and not presented as fact

### Research Ethics

**Transparency**:
- Open methodology
- Code and data availability (where permitted)
- Clear reporting of limitations

**Responsible AI**:
- Research aims to improve AI safety
- Findings contribute to trustworthy AI
- Results may inform policy and regulation

**Potential Misuse**:
- Methods could theoretically be used to improve deceptive AI
- Mitigation: Emphasize hallucination detection, not generation
- Focus on defensive applications

## Timeline

**Phase 1** (Week 1-2): Dataset preparation and filtering

**Phase 2** (Week 3-4): RAG response generation and data collection

**Phase 3** (Week 5-6): MI analysis and metric computation

**Phase 4** (Week 7-10): Manual annotation and statistical validation

**Phase 5** (Week 11-12): Analysis synthesis and visualization

**Total Duration**: 12 weeks

## Expected Contributions

### Theoretical Contributions

1. **Mechanistic understanding of RAG hallucinations**:
   - Quantify role of external vs parametric knowledge
   - Identify attention patterns associated with factuality

2. **Extension of RAGTruth to specialized domain**:
   - Demonstrate applicability to legal/policy texts
   - Domain-specific hallucination patterns

3. **Integration of multiple MI techniques**:
   - Combined ECS, PKS, and Logit Lens analysis
   - Holistic framework for hallucination detection

### Methodological Contributions

1. **Reusable MI analysis pipeline**:
   - Modular Python codebase
   - Applicable to other RAG systems and domains

2. **Validated metrics for hallucination prediction**:
   - ECS and PKS as quantitative indicators
   - ROC curves and optimal thresholds

3. **Annotation framework for AI governance domain**:
   - Adapted RAGTruth schema
   - Guidelines for policy document hallucinations

### Practical Contributions

1. **Improved RAG trustworthiness**:
   - Identify when to trust vs distrust RAG outputs
   - Real-time hallucination monitoring

2. **Policy implications**:
   - Inform deployment of RAG in high-stakes domains
   - Guide AI governance policy development

3. **Tool for AI safety research**:
   - Open-source implementation
   - Foundation for future research

## References

1. Wu, Y., et al. (2024). RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. arXiv:2401.00396.

2. nostalgebraist. (2020). Interpreting GPT: The Logit Lens. Available: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

3. Sun, J., et al. (2024). Detecting and Reducing Hallucination in Large Language Models via Mechanistic Interpretability. ICLR 2024.

4. Jiang, A. Q., et al. (2023). Mistral 7B. arXiv:2310.06825.

5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.

6. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.

