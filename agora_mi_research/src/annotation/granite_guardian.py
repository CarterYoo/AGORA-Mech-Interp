"""
IBM Granite Guardian integration for automated hallucination detection.

This module integrates IBM's Granite Guardian model for automated annotation
of hallucination spans, enabling scalable analysis beyond manual annotation capacity.

Reference: 
- IBM Granite Guardian: https://research.ibm.com/publications/granite-guardian
- Model: ibm-granite/granite-guardian-3.3-8b
- Official Guide: detailed_guide_think.ipynb
"""

import re
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import numpy as np

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.info("vLLM not available (Windows or not installed). Using transformers instead.")

import platform
if platform.system() == "Windows":
    VLLM_AVAILABLE = False
    logger.info("Running on Windows - vLLM not supported. Using transformers.")


class GraniteGuardianAnnotator:
    """
    Automated annotator using IBM Granite Guardian.
    
    Granite Guardian is designed to detect risks including hallucinations
    in RAG scenarios using the 'groundedness' criteria. We use it to:
    1. Automatically annotate responses at scale
    2. Validate manual annotations
    3. Pre-filter responses for manual review
    
    Based on official Granite Guardian guide with proper usage of:
    - criteria_id: "groundedness" for RAG hallucination detection
    - documents parameter for context
    - <think> and <score> tag parsing
    """
    
    GRANITE_GUARDIAN_MODELS = {
        'latest': 'ibm-granite/granite-guardian-3.3-8b',  # Latest (recommended)
        'hap': 'ibm-granite/granite-guardian-hap-38m',    # Harm/Abuse/Profanity
        '8b': 'ibm-granite/granite-guardian-3.3-8b',      # Full model
        '2b': 'ibm-granite/granite-guardian-3.3-2b'       # Smaller version
    }
    
    def __init__(
        self,
        model_variant: str = 'latest',
        use_vllm: bool = True,
        device: str = 'cuda',
        temperature: float = 0.0,
        max_tokens: int = 2048
    ):
        """
        Initialize Granite Guardian annotator following official guide.
        
        Args:
            model_variant: Which Granite Guardian variant ('latest', '8b', '2b', 'hap')
            use_vllm: Whether to use vLLM for faster inference
            device: Device for inference ('cuda' or 'cpu')
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens for reasoning
        """
        if model_variant not in self.GRANITE_GUARDIAN_MODELS:
            raise ValueError(
                f"Unknown variant: {model_variant}. "
                f"Choose from: {list(self.GRANITE_GUARDIAN_MODELS.keys())}"
            )
        
        self.model_name = self.GRANITE_GUARDIAN_MODELS[model_variant]
        self.model_variant = model_variant
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Loading Granite Guardian: {self.model_name}")
        logger.info(f"Using vLLM: {self.use_vllm}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.use_vllm:
                self.sampling_params = SamplingParams(
                    temperature=temperature,
                    logprobs=20,
                    max_tokens=max_tokens
                )
                self.model = LLM(
                    model=self.model_name,
                    tensor_parallel_size=1
                )
                logger.info("Loaded with vLLM for fast inference")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                    device_map='auto' if device == 'cuda' else None
                )
                
                if device == 'cpu':
                    self.model = self.model.to('cpu')
                
                self.model.eval()
                logger.info(f"Loaded with transformers on {device}")
            
            logger.info("Granite Guardian loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Granite Guardian: {e}")
            logger.error(
                "Please check:\n"
                "1. Model access: https://huggingface.co/ibm-granite/granite-guardian-3.3-8b\n"
                "2. Install vLLM: pip install vllm (optional, for speed)\n"
                "3. GPU availability for large model"
            )
            raise
    
    def parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse Granite Guardian response to extract score and reasoning.
        
        Granite outputs:
        <think>reasoning...</think>
        <score>yes/no</score>
        
        Args:
            response: Raw model output
        
        Returns:
            Tuple of (score, trace/reasoning)
        """
        trace_match = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
        score_match = re.findall(r'<score>(.*?)</score>', response, re.DOTALL)
        
        score = score_match[-1].strip().lower() if score_match else None
        trace = trace_match[-1].strip() if trace_match else None
        
        return score, trace
    
    def detect_hallucination(
        self,
        question: str,
        context: str,
        response: str
    ) -> Dict:
        """
        Detect whether response contains hallucination using groundedness criteria.
        
        Following official Granite Guardian guide:
        - Uses 'groundedness' criteria_id for RAG hallucination
        - Passes context as 'documents' parameter
        - Parses <think> and <score> tags
        
        Args:
            question: Question text
            context: Retrieved context
            response: Generated response
        
        Returns:
            Dictionary with detection results and reasoning
        """
        # Format as messages (official format)
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
        
        # Guardian config with groundedness criteria
        guardian_config = {"criteria_id": "groundedness"}
        
        # Format documents (context)
        documents = [{"content": context}]
        
        # Apply chat template (official method)
        chat = self.tokenizer.apply_chat_template(
            messages,
            guardian_config=guardian_config,
            documents=documents,
            think=True,  # Enable reasoning
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate with model
        if self.use_vllm:
            # vLLM path (faster)
            output = self.model.generate(chat, self.sampling_params, use_tqdm=False)
            raw_response = output[0].outputs[0].text.strip()
        else:
            # Standard transformers path
            inputs = self.tokenizer(
                chat,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0
                )
            
            raw_response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        
        # Parse response
        score, trace = self.parse_response(raw_response)
        
        # Convert score to boolean
        is_hallucination = (score == 'yes') if score else None
        
        result = {
            'is_hallucination': is_hallucination,
            'score': score,  # 'yes' or 'no'
            'reasoning': trace,  # Full reasoning from <think> tag
            'raw_response': raw_response,
            'model': self.model_name,
            'criteria': 'groundedness'
        }
        
        logger.debug(
            f"Hallucination detected: {is_hallucination} "
            f"(score: {score})"
        )
        
        return result
    
    def annotate_batch(
        self,
        rag_results: List[Dict]
    ) -> List[Dict]:
        """
        Automatically annotate batch of RAG results.
        
        Args:
            rag_results: List of RAG result dictionaries
        
        Returns:
            List of annotation dictionaries
        """
        logger.info(f"Annotating {len(rag_results)} responses with Granite Guardian")
        
        annotations = []
        
        for i, rag_result in enumerate(rag_results):
            try:
                question = rag_result.get('question', '')
                response = rag_result.get('response', '')
                
                # Format context from retrieved segments
                context_parts = []
                for seg in rag_result.get('retrieved_segments', []):
                    context_parts.append(seg['text'])
                context = '\n\n'.join(context_parts)
                
                # Detect hallucination
                detection = self.detect_hallucination(question, context, response)
                
                annotation = {
                    'question_id': rag_result.get('question_id', ''),
                    'question': question,
                    'response': response,
                    'automated_annotation': {
                        'is_hallucination': detection['is_hallucination'],
                        'score': detection['score'],
                        'reasoning': detection['reasoning'],
                        'model': self.model_name,
                        'criteria': detection['criteria']
                    },
                    'requires_manual_review': detection['score'] is None
                }
                
                annotations.append(annotation)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(rag_results)}")
                
            except Exception as e:
                logger.error(f"Error annotating {rag_result.get('question_id', i)}: {e}")
                continue
        
        logger.info(f"Automated annotation complete: {len(annotations)} annotated")
        
        return annotations
    
    def validate_against_manual(
        self,
        automated_annotations: List[Dict],
        manual_annotations: List[Dict]
    ) -> Dict:
        """
        Validate automated annotations against manual annotations.
        
        Args:
            automated_annotations: Annotations from Granite Guardian
            manual_annotations: Manual annotations (gold standard)
        
        Returns:
            Validation metrics dictionary
        """
        from src.evaluation.metrics import EvaluationMetrics
        
        metrics_calc = EvaluationMetrics()
        
        # Align annotations by question_id
        auto_dict = {
            ann['question_id']: ann['automated_annotation']['is_hallucination']
            for ann in automated_annotations
        }
        
        manual_dict = {
            ann['question_id']: len(ann.get('hallucination_spans', [])) > 0
            for ann in manual_annotations
        }
        
        # Get common question_ids
        common_ids = set(auto_dict.keys()) & set(manual_dict.keys())
        
        y_true = [manual_dict[qid] for qid in common_ids]
        y_pred = [auto_dict[qid] for qid in common_ids]
        
        # Compute metrics
        validation = metrics_calc.compute_classification_metrics(y_true, y_pred)
        
        logger.info(
            f"Validation against {len(common_ids)} manual annotations: "
            f"Accuracy={validation['accuracy']:.3f}, "
            f"F1={validation['f1_score']:.3f}"
        )
        
        return validation
    
    def get_model_info(self) -> Dict:
        """
        Get Granite Guardian model information.
        
        Returns:
            Dictionary with model info
        """
        if self.use_vllm:
            num_params = "N/A (vLLM model)"
        else:
            num_params = sum(p.numel() for p in self.model.parameters())
        
        info = {
            'model_name': self.model_name,
            'model_variant': self.model_variant,
            'num_parameters': num_params,
            'device': self.device,
            'use_vllm': self.use_vllm,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
        
        return info


class GraniteGuardianFinetuner:
    """
    Fine-tune Granite Guardian on domain-specific data.
    
    Uses manual annotations to fine-tune Granite Guardian for
    AGORA-specific hallucination patterns.
    """
    
    def __init__(
        self,
        base_model: str = 'ibm-granite/granite-guardian-3.0-2b',
        output_dir: str = 'outputs/granite_guardian_finetuned'
    ):
        """
        Initialize fine-tuner.
        
        Args:
            base_model: Base Granite Guardian model
            output_dir: Directory to save fine-tuned model
        """
        self.base_model = base_model
        self.output_dir = output_dir
        
        logger.info(f"Initialized GraniteGuardianFinetuner: {base_model}")
    
    def prepare_training_data(
        self,
        manual_annotations: List[Dict],
        rag_results: List[Dict]
    ) -> Dict:
        """
        Prepare training data from manual annotations.
        
        Args:
            manual_annotations: Manual annotations (gold standard)
            rag_results: Corresponding RAG results
        
        Returns:
            DatasetDict with train/test splits
        """
        logger.info("Preparing training data from manual annotations...")
        
        # Create annotation lookup
        ann_dict = {
            ann['question_id']: ann
            for ann in manual_annotations
        }
        
        training_examples = []
        
        for rag_result in rag_results:
            qid = rag_result.get('question_id', '')
            
            if qid not in ann_dict:
                continue
            
            annotation = ann_dict[qid]
            
            # Format context
            context_parts = []
            for seg in rag_result.get('retrieved_segments', []):
                context_parts.append(seg['text'])
            context = '\n\n'.join(context_parts)
            
            # Create training example
            example = {
                'question': rag_result.get('question', ''),
                'context': context,
                'response': rag_result.get('response', ''),
                'label': 1 if len(annotation.get('hallucination_spans', [])) > 0 else 0
            }
            
            training_examples.append(example)
        
        logger.info(f"Created {len(training_examples)} training examples")
        
        # Split train/test (80/20)
        n_train = int(len(training_examples) * 0.8)
        
        import random
        random.seed(42)
        random.shuffle(training_examples)
        
        train_data = training_examples[:n_train]
        test_data = training_examples[n_train:]
        
        from datasets import Dataset, DatasetDict
        
        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
        
        logger.info(f"Dataset: {len(train_data)} train, {len(test_data)} test")
        
        return dataset
    
    def fine_tune(
        self,
        dataset: Dict,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        per_device_batch_size: int = 8
    ) -> str:
        """
        Fine-tune Granite Guardian on AGORA data.
        
        Args:
            dataset: DatasetDict with train/test
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            per_device_batch_size: Batch size per device
        
        Returns:
            Path to fine-tuned model
        """
        logger.info("Starting Granite Guardian fine-tuning...")
        
        # This is a placeholder - actual fine-tuning requires:
        # 1. Sufficient training data (50+ examples)
        # 2. GPU resources
        # 3. Granite Guardian model access
        
        logger.warning(
            "Fine-tuning not implemented in this version. "
            "Use pre-trained Granite Guardian for now."
        )
        
        return self.base_model


def main():
    """
    Example usage of GraniteGuardianAnnotator (official method).
    """
    print("IBM Granite Guardian Annotator")
    print("="*60)
    print()
    print("Official Implementation following detailed_guide_think.ipynb")
    print()
    print("Model: ibm-granite/granite-guardian-3.3-8b")
    print("Criteria: groundedness (RAG hallucination detection)")
    print()
    print("Usage:")
    print()
    print("from src.annotation.granite_guardian import GraniteGuardianAnnotator")
    print()
    print("# Initialize (with vLLM for speed)")
    print("annotator = GraniteGuardianAnnotator(")
    print("    model_variant='latest',")
    print("    use_vllm=True,")
    print("    device='cuda'")
    print(")")
    print()
    print("# Detect hallucination")
    print("result = annotator.detect_hallucination(")
    print("    question='What funding is allocated?',")
    print("    context='The Act appropriates $500 million for AI modernization.',")
    print("    response='The Act provides $500 million for AI and military applications.'")
    print(")")
    print()
    print("print(f'Hallucination: {result[\"is_hallucination\"]}')")
    print("print(f'Score: {result[\"score\"]}')")
    print("print(f'Reasoning: {result[\"reasoning\"][:200]}...')")
    print()
    print("Output format:")
    print("  is_hallucination: True/False")
    print("  score: 'yes' or 'no'")
    print("  reasoning: Full explanation from <think> tag")
    print()
    print("Requirements:")
    print("  1. HuggingFace access: huggingface-cli login")
    print("  2. Model: https://huggingface.co/ibm-granite/granite-guardian-3.3-8b")
    print("  3. Optional: pip install vllm (for 5-10x speedup)")
    print("  4. GPU with 16GB+ VRAM (24GB for vLLM)")


if __name__ == "__main__":
    main()

