"""
Response generation module for RAG pipeline.

This module implements response generation using Mistral-7B with full attention
and hidden state extraction for mechanistic interpretability analysis.
"""

import torch
from typing import List, Dict, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from loguru import logger
import numpy as np

from ..mi.aarf_intervention import AARFIntervention, HallucinationScoreCalculator
from ..mi.ecs_calculator import ECSCalculator
from ..mi.pks_calculator import PKSCalculator
from ..mi.data_structures import MIAnalysisData, TrajectoryData


class ModelInferenceError(Exception):
    """Raised when model inference fails."""
    pass


class ResponseGenerator:
    """
    Response generator using Mistral-7B with MI capabilities.
    
    Generates RAG responses with full attention tensor and hidden state extraction
    for subsequent mechanistic interpretability analysis.
    """
    
    CONTEXT_START_MARKER = "BEGIN_CONTEXT"
    CONTEXT_END_MARKER = "END_CONTEXT"
    
    DEFAULT_PROMPT_TEMPLATE = """You are an AI assistant answering questions about AI governance policies.

Context:
{context_marker_start}
{context}
{context_marker_end}

Question: {question}

Answer based strictly on the provided context. If the context does not contain enough information to answer the question, state this clearly.

Answer:"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        load_4bit: bool = True,
        device_map: str = "auto",
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        enable_aarf: bool = False,
        aarf_threshold: float = 0.6,
        aarf_attention_multiplier: float = 1.5,
        aarf_ffn_suppress: float = 0.7
    ):
        """
        Initialize response generator.
        
        Args:
            model_name: HuggingFace model name
            load_4bit: Whether to use 4-bit quantization
            device_map: Device mapping strategy
            max_length: Maximum context length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty factor
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        
        self.enable_aarf = enable_aarf
        self.aarf_threshold = aarf_threshold
        self.aarf_attention_multiplier = aarf_attention_multiplier
        self.aarf_ffn_suppress = aarf_ffn_suppress
        self.aarf_intervention = None
        self.hallucination_calculator = None
        
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": device_map if torch.cuda.is_available() else None,
            "output_attentions": True,
            "output_hidden_states": True,
            "trust_remote_code": True
        }
        
        if load_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            logger.info("Using 4-bit quantization")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.eval()
        
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        logger.info(f"Model loaded: device={device}, dtype={dtype}")
        
        if self.enable_aarf:
            self.hallucination_calculator = HallucinationScoreCalculator(
                threshold=self.aarf_threshold
            )
            logger.info("AARF enabled, hallucination calculator initialized")
    
    def format_context(
        self,
        retrieved_segments: List[Dict],
        max_context_length: int = 1000
    ) -> Tuple[str, List[int]]:
        """
        Format retrieved segments into context string.
        
        Args:
            retrieved_segments: List of retrieved segment dictionaries
            max_context_length: Maximum tokens in context
        
        Returns:
            Tuple of (formatted_context, segment_token_counts)
        """
        context_parts = []
        token_counts = []
        total_tokens = 0
        
        for i, segment in enumerate(retrieved_segments, 1):
            segment_text = segment['text']
            segment_tokens = len(self.tokenizer.encode(segment_text, add_special_tokens=False))
            
            if total_tokens + segment_tokens > max_context_length and context_parts:
                break
            
            context_parts.append(f"[{i}] {segment_text}")
            token_counts.append(segment_tokens)
            total_tokens += segment_tokens
        
        formatted_context = "\n\n".join(context_parts)
        
        logger.debug(
            f"Formatted context: {len(context_parts)} segments, "
            f"{total_tokens} tokens"
        )
        
        return formatted_context, token_counts
    
    def create_prompt(
        self,
        question: str,
        context: str,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Create prompt from question and context.
        
        Args:
            question: Question text
            context: Formatted context string
            prompt_template: Optional custom prompt template
        
        Returns:
            Formatted prompt string
        """
        if prompt_template is None:
            prompt_template = self.DEFAULT_PROMPT_TEMPLATE
        
        prompt = prompt_template.format(
            context_marker_start=self.CONTEXT_START_MARKER,
            context=context,
            context_marker_end=self.CONTEXT_END_MARKER,
            question=question
        )
        
        return prompt
    
    def identify_context_boundaries(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Identify token positions of context markers.
        
        Args:
            input_ids: Tokenized input tensor [batch_size, seq_len]
        
        Returns:
            Tuple of (context_start_pos, context_end_pos) or (None, None)
        """
        input_ids_list = input_ids[0].tolist()
        
        start_marker_ids = self.tokenizer.encode(
            self.CONTEXT_START_MARKER,
            add_special_tokens=False
        )
        end_marker_ids = self.tokenizer.encode(
            self.CONTEXT_END_MARKER,
            add_special_tokens=False
        )
        
        start_pos = None
        end_pos = None
        
        for i in range(len(input_ids_list) - len(start_marker_ids) + 1):
            if input_ids_list[i:i+len(start_marker_ids)] == start_marker_ids:
                start_pos = i + len(start_marker_ids)
                break
        
        for i in range(len(input_ids_list) - len(end_marker_ids) + 1):
            if input_ids_list[i:i+len(end_marker_ids)] == end_marker_ids:
                end_pos = i
                break
        
        if start_pos and end_pos:
            logger.debug(
                f"Context boundaries: start={start_pos}, end={end_pos}, "
                f"length={end_pos - start_pos}"
            )
        else:
            logger.warning("Could not identify context boundaries")
        
        return start_pos, end_pos
    
    def generate(
        self,
        question: str,
        retrieved_segments: List[Dict],
        max_new_tokens: int = 512,
        return_attention: bool = True,
        return_hidden_states: bool = True
    ) -> Dict:
        """
        Generate response with full MI data.
        
        Args:
            question: Question text
            retrieved_segments: List of retrieved segments
            max_new_tokens: Maximum tokens to generate
            return_attention: Whether to return attention tensors
            return_hidden_states: Whether to return hidden states
        
        Returns:
            Dictionary containing response and MI data
        """
        try:
            context, segment_token_counts = self.format_context(retrieved_segments)
            
            prompt = self.create_prompt(question, context)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            context_start, context_end = self.identify_context_boundaries(
                inputs['input_ids']
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True,
                    output_attentions=return_attention,
                    output_hidden_states=return_hidden_states,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated_ids = outputs.sequences[0]
            input_length = inputs['input_ids'].shape[1]
            
            response = self.tokenizer.decode(
                generated_ids[input_length:],
                skip_special_tokens=True
            )
            
            result = {
                'question': question,
                'response': response,
                'retrieved_segments': retrieved_segments,
                'num_segments': len(retrieved_segments),
                'context_token_count': sum(segment_token_counts),
                'input_length': input_length,
                'output_length': len(generated_ids) - input_length,
                'total_length': len(generated_ids),
                'context_start_pos': context_start,
                'context_end_pos': context_end,
                'prompt': prompt,
                'full_text': self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            }
            
            if return_attention and hasattr(outputs, 'attentions') and outputs.attentions:
                logger.debug(f"Captured attention tensors: {len(outputs.attentions)} generation steps")
                result['has_attention'] = True
                result['num_attention_steps'] = len(outputs.attentions)
            else:
                result['has_attention'] = False
            
            if return_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                logger.debug(f"Captured hidden states: {len(outputs.hidden_states)} generation steps")
                result['has_hidden_states'] = True
                result['num_hidden_state_steps'] = len(outputs.hidden_states)
            else:
                result['has_hidden_states'] = False
            
            logger.info(
                f"Generated response: {result['output_length']} tokens, "
                f"attention={result['has_attention']}, "
                f"hidden_states={result['has_hidden_states']}"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            raise ModelInferenceError(error_msg) from e
    
    def generate_batch(
        self,
        questions: List[str],
        retrieved_segments_list: List[List[Dict]],
        max_new_tokens: int = 512,
        return_attention: bool = True,
        return_hidden_states: bool = True
    ) -> List[Dict]:
        """
        Generate responses for multiple questions.
        
        Args:
            questions: List of questions
            retrieved_segments_list: List of retrieved segments for each question
            max_new_tokens: Maximum tokens to generate per response
            return_attention: Whether to return attention tensors
            return_hidden_states: Whether to return hidden states
        
        Returns:
            List of result dictionaries
        """
        if len(questions) != len(retrieved_segments_list):
            raise ValueError(
                f"Mismatch: {len(questions)} questions, "
                f"{len(retrieved_segments_list)} segment lists"
            )
        
        results = []
        for i, (question, segments) in enumerate(zip(questions, retrieved_segments_list)):
            try:
                result = self.generate(
                    question=question,
                    retrieved_segments=segments,
                    max_new_tokens=max_new_tokens,
                    return_attention=return_attention,
                    return_hidden_states=return_hidden_states
                )
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(questions)} responses")
                    
            except ModelInferenceError as e:
                logger.error(f"Failed to generate response for question {i}: {e}")
                results.append({
                    'question': question,
                    'error': str(e),
                    'success': False
                })
        
        logger.info(f"Batch generation complete: {len(results)} responses")
        successful = sum(1 for r in results if 'error' not in r)
        logger.info(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.model_name,
            'device': str(device),
            'dtype': str(dtype),
            'num_parameters': num_params,
            'num_trainable_parameters': num_trainable,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repetition_penalty': self.repetition_penalty
        }
        
        logger.info(f"Model info: {info}")
        return info
    
    def generate_with_aarf(
        self,
        question: str,
        retrieved_segments: List[Dict],
        max_new_tokens: int = 512,
        ecs_calculator: Optional[ECSCalculator] = None,
        pks_calculator: Optional[PKSCalculator] = None,
        return_attention: bool = True,
        return_hidden_states: bool = True
    ) -> Dict:
        """
        Generate response with AARF intervention when hallucination risk detected.
        
        This method first generates a baseline response, computes ECS/PKS,
        calculates hallucination score, and applies AARF if threshold exceeded.
        
        Args:
            question: Question text
            retrieved_segments: List of retrieved segments
            max_new_tokens: Maximum tokens to generate
            ecs_calculator: ECSCalculator instance (creates default if None)
            pks_calculator: PKSCalculator instance (creates default if None)
            return_attention: Whether to return attention tensors
            return_hidden_states: Whether to return hidden states
        
        Returns:
            Dictionary containing response, MI data, and AARF intervention info
        """
        if not self.enable_aarf:
            logger.warning("AARF not enabled, falling back to standard generation")
            return self.generate(
                question=question,
                retrieved_segments=retrieved_segments,
                max_new_tokens=max_new_tokens,
                return_attention=return_attention,
                return_hidden_states=return_hidden_states
            )
        
        if ecs_calculator is None:
            ecs_calculator = ECSCalculator()
        if pks_calculator is None:
            pks_calculator = PKSCalculator()
        
        logger.info("Generating baseline response for AARF analysis...")
        
        baseline_result = self.generate(
            question=question,
            retrieved_segments=retrieved_segments,
            max_new_tokens=max_new_tokens,
            return_attention=True,
            return_hidden_states=True
        )
        
        if not baseline_result.get('has_attention') or not baseline_result.get('has_hidden_states'):
            logger.warning("Baseline generation missing attention/hidden states, skipping AARF")
            return baseline_result
        
        context_start = baseline_result.get('context_start_pos')
        context_end = baseline_result.get('context_end_pos')
        
        if context_start is None or context_end is None:
            logger.warning("Could not identify context boundaries, skipping AARF")
            return baseline_result
        
        input_length = baseline_result['input_length']
        generated_start = input_length
        
        logger.info("Computing ECS and PKS for baseline response...")
        
        try:
            attention_tensors = []
            hidden_states_list = []
            logits_list = []
            
            if hasattr(baseline_result, 'attentions'):
                for step_attentions in baseline_result.attentions:
                    if step_attentions:
                        attention_tensors.append(torch.stack(step_attentions))
            
            mean_ecs = 0.0
            mean_pks = 0.0
            copying_heads = []
            
            if attention_tensors:
                full_attention = torch.stack(attention_tensors)
                ecs_analysis = ecs_calculator.compute_sequence_ecs(
                    attention=full_attention,
                    context_start=context_start,
                    context_end=context_end,
                    generated_start=generated_start
                )
                
                copying_heads = ecs_calculator.identify_copying_heads(
                    attention=full_attention,
                    context_start=context_start,
                    context_end=context_end,
                    generated_start=generated_start
                )
                
                if ecs_analysis.get('token_ecs'):
                    mean_ecs = np.mean([step['mean_ecs'] for step in ecs_analysis['token_ecs']])
            
            if hidden_states_list:
                pks_analysis = pks_calculator.compute_sequence_pks(
                    logits_list=logits_list,
                    hidden_states_list=hidden_states_list
                )
                
                if pks_analysis.get('token_pks'):
                    mean_pks = np.mean([step['pks'] for step in pks_analysis['token_pks']])
            
            should_intervene, h_score = self.hallucination_calculator.should_intervene(
                ecs=float(mean_ecs),
                pks=float(mean_pks)
            )
            
            logger.info(
                f"Baseline analysis: ECS={mean_ecs:.3f}, PKS={mean_pks:.3f}, "
                f"Hallucination score={h_score:.3f}, "
                f"Intervention needed={should_intervene}"
            )
            
            result = baseline_result.copy()
            result['aarf_analysis'] = {
                'baseline_ecs': float(mean_ecs),
                'baseline_pks': float(mean_pks),
                'hallucination_score': float(h_score),
                'should_intervene': should_intervene,
                'copying_heads': copying_heads
            }
            
            if should_intervene and copying_heads:
                logger.info("Applying AARF intervention and regenerating...")
                
                self.aarf_intervention = AARFIntervention(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    copying_heads=copying_heads,
                    context_start=context_start,
                    context_end=context_end,
                    attention_multiplier=self.aarf_attention_multiplier,
                    ffn_suppress=self.aarf_ffn_suppress
                )
                
                self.aarf_intervention.apply_intervention()
                
                try:
                    aarf_result = self.generate(
                        question=question,
                        retrieved_segments=retrieved_segments,
                        max_new_tokens=max_new_tokens,
                        return_attention=return_attention,
                        return_hidden_states=return_hidden_states
                    )
                    
                    aarf_stats = self.aarf_intervention.get_stats()
                    
                    result['aarf_response'] = aarf_result['response']
                    result['aarf_intervention_applied'] = True
                    result['aarf_stats'] = aarf_stats
                    
                    logger.info(
                        f"AARF intervention applied: "
                        f"{aarf_stats['attention_modifications']} attention mods, "
                        f"{aarf_stats['ffn_suppressions']} FFN suppressions"
                    )
                    
                finally:
                    self.aarf_intervention.remove_intervention()
                    self.aarf_intervention = None
            else:
                result['aarf_intervention_applied'] = False
                logger.info("AARF intervention not needed (threshold not exceeded)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AARF analysis: {e}")
            if self.aarf_intervention is not None:
                try:
                    self.aarf_intervention.remove_intervention()
                except:
                    pass
                self.aarf_intervention = None
            return baseline_result


def main():
    """
    Example usage of ResponseGenerator.
    """
    logger.info("Initializing ResponseGenerator...")
    
    generator = ResponseGenerator(
        load_4bit=True,
        temperature=0.7
    )
    
    sample_segments = [
        {
            'segment_id': 'seg_1',
            'text': 'The EU AI Act requires high-risk AI systems to undergo conformity assessment.',
            'similarity': 0.95
        },
        {
            'segment_id': 'seg_2',
            'text': 'Providers must implement appropriate data governance and management practices.',
            'similarity': 0.88
        }
    ]
    
    question = "What are the requirements for high-risk AI systems?"
    
    logger.info(f"Generating response for: {question}")
    result = generator.generate(
        question=question,
        retrieved_segments=sample_segments,
        max_new_tokens=100
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nResponse: {result['response']}")
    print(f"\nMetadata:")
    print(f"  Input length: {result['input_length']} tokens")
    print(f"  Output length: {result['output_length']} tokens")
    print(f"  Context boundaries: {result['context_start_pos']}-{result['context_end_pos']}")
    print(f"  Has attention: {result['has_attention']}")
    print(f"  Has hidden states: {result['has_hidden_states']}")
    
    model_info = generator.get_model_info()
    print(f"\nModel info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

