"""
DPO (Direct Preference Optimization) enhanced response generator.

This module extends the base ResponseGenerator to support DPO fine-tuned models
for human-aligned, factually grounded generation as described in the AGORA Q&A system.

Reference: 
- AGORA Q&A: https://github.com/rrittner1/agora
- DPO Paper: Rafailov et al. (2023), "Direct Preference Optimization"
"""

import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from .generator import ResponseGenerator, ModelInferenceError


class DPOResponseGenerator(ResponseGenerator):
    """
    Response generator with DPO (Direct Preference Optimization) fine-tuning.
    
    DPO fine-tuning aligns the model with human preferences for factual accuracy
    and reduces hallucinations without requiring explicit reward modeling.
    
    This class extends ResponseGenerator to support:
    1. Loading DPO fine-tuned models
    2. Loading LoRA adapters for DPO
    3. Maintaining compatibility with MI analysis
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        adapter_path: Optional[str] = None,
        beta: float = 0.1,
        load_4bit: bool = True,
        device_map: str = "auto",
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ):
        """
        Initialize DPO response generator.
        
        Args:
            model_name: Base model or DPO fine-tuned model name
            adapter_path: Path to DPO LoRA adapter (optional)
            beta: DPO temperature parameter
            load_4bit: Whether to use 4-bit quantization
            device_map: Device mapping strategy
            max_length: Maximum context length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty factor
        """
        self.adapter_path = adapter_path
        self.beta = beta
        self.use_adapter = adapter_path is not None
        
        super().__init__(
            model_name=model_name,
            load_4bit=load_4bit,
            device_map=device_map,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        
        if self.use_adapter:
            self._load_dpo_adapter()
        
        logger.info(
            f"Initialized DPOResponseGenerator: "
            f"adapter={'loaded' if self.use_adapter else 'none'}, "
            f"beta={beta}"
        )
    
    def _load_dpo_adapter(self) -> None:
        """
        Load DPO LoRA adapter onto the base model.
        
        This method uses PEFT (Parameter-Efficient Fine-Tuning) to load
        a LoRA adapter trained with DPO.
        """
        try:
            from peft import PeftModel
            
            logger.info(f"Loading DPO adapter from {self.adapter_path}")
            
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                torch_dtype=self.model.dtype,
                device_map=self.model.device
            )
            
            self.model = self.model.merge_and_unload()
            
            logger.info("DPO adapter loaded and merged successfully")
            
        except ImportError:
            logger.error(
                "PEFT library not available. Install with: pip install peft\n"
                "DPO adapter cannot be loaded without PEFT."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load DPO adapter: {e}")
            raise ModelInferenceError(f"DPO adapter loading failed: {e}") from e
    
    def generate(
        self,
        question: str,
        retrieved_segments: List[Dict],
        max_new_tokens: int = 512,
        return_attention: bool = True,
        return_hidden_states: bool = True,
        use_dpo_temperature: bool = False
    ) -> Dict:
        """
        Generate response with DPO-enhanced model.
        
        Args:
            question: Question text
            retrieved_segments: List of retrieved segments
            max_new_tokens: Maximum tokens to generate
            return_attention: Whether to return attention tensors
            return_hidden_states: Whether to return hidden states
            use_dpo_temperature: Whether to apply DPO beta temperature
        
        Returns:
            Dictionary containing response and MI data
        """
        original_temp = self.temperature
        
        if use_dpo_temperature:
            self.temperature = self.beta
            logger.debug(f"Using DPO temperature: {self.beta}")
        
        try:
            result = super().generate(
                question=question,
                retrieved_segments=retrieved_segments,
                max_new_tokens=max_new_tokens,
                return_attention=return_attention,
                return_hidden_states=return_hidden_states
            )
            
            result['model_type'] = 'dpo'
            result['dpo_beta'] = self.beta
            result['used_adapter'] = self.use_adapter
            
            return result
            
        finally:
            if use_dpo_temperature:
                self.temperature = original_temp
    
    def compare_with_vanilla(
        self,
        question: str,
        retrieved_segments: List[Dict],
        vanilla_generator: ResponseGenerator,
        max_new_tokens: int = 512
    ) -> Dict:
        """
        Generate responses from both DPO and vanilla models for comparison.
        
        This is useful for analyzing the impact of DPO fine-tuning on
        response quality and mechanistic interpretability patterns.
        
        Args:
            question: Question text
            retrieved_segments: List of retrieved segments
            vanilla_generator: Vanilla (non-DPO) generator instance
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Dictionary with both responses and comparison metadata
        """
        logger.info("Generating comparison: DPO vs Vanilla")
        
        dpo_result = self.generate(
            question=question,
            retrieved_segments=retrieved_segments,
            max_new_tokens=max_new_tokens
        )
        
        vanilla_result = vanilla_generator.generate(
            question=question,
            retrieved_segments=retrieved_segments,
            max_new_tokens=max_new_tokens
        )
        
        comparison = {
            'question': question,
            'retrieved_segments': retrieved_segments,
            'dpo_response': {
                'text': dpo_result['response'],
                'output_length': dpo_result['output_length'],
                'has_attention': dpo_result['has_attention'],
                'has_hidden_states': dpo_result['has_hidden_states']
            },
            'vanilla_response': {
                'text': vanilla_result['response'],
                'output_length': vanilla_result['output_length'],
                'has_attention': vanilla_result['has_attention'],
                'has_hidden_states': vanilla_result['has_hidden_states']
            },
            'response_similarity': self._compute_similarity(
                dpo_result['response'],
                vanilla_result['response']
            )
        }
        
        logger.info(
            f"Comparison complete: "
            f"DPO={dpo_result['output_length']} tokens, "
            f"Vanilla={vanilla_result['output_length']} tokens"
        )
        
        return comparison
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple word-level similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Jaccard similarity score
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_model_info(self) -> Dict:
        """
        Get DPO model information.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        info.update({
            'model_type': 'dpo',
            'dpo_beta': self.beta,
            'adapter_path': self.adapter_path,
            'use_adapter': self.use_adapter
        })
        
        return info


def create_generator_from_config(config: Dict) -> ResponseGenerator:
    """
    Factory function to create appropriate generator from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        ResponseGenerator or DPOResponseGenerator instance
    """
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'vanilla')
    
    if model_type == 'dpo':
        dpo_config = model_config.get('dpo', {})
        
        return DPOResponseGenerator(
            model_name=dpo_config.get('base_model', model_config.get('name')),
            adapter_path=dpo_config.get('adapter_path'),
            beta=dpo_config.get('beta', 0.1),
            load_4bit=model_config.get('quantization') == '4bit',
            max_length=model_config.get('max_length', 2048),
            temperature=model_config.get('temperature', 0.7),
            top_p=model_config.get('top_p', 0.9)
        )
    else:
        return ResponseGenerator(
            model_name=model_config.get('name'),
            load_4bit=model_config.get('quantization') == '4bit',
            max_length=model_config.get('max_length', 2048),
            temperature=model_config.get('temperature', 0.7),
            top_p=model_config.get('top_p', 0.9)
        )


def main():
    """
    Example usage of DPOResponseGenerator.
    """
    print("DPOResponseGenerator implementation complete.")
    print("\nTo use DPO model:")
    print()
    print("# Option 1: DPO fine-tuned model directly")
    print("generator = DPOResponseGenerator(")
    print("    model_name='path/to/dpo-model',")
    print("    beta=0.1")
    print(")")
    print()
    print("# Option 2: Base model + LoRA adapter")
    print("generator = DPOResponseGenerator(")
    print("    model_name='mistralai/Mistral-7B-Instruct-v0.2',")
    print("    adapter_path='path/to/dpo-adapter',")
    print("    beta=0.1")
    print(")")
    print()
    print("# Option 3: From config file")
    print("import yaml")
    print("with open('configs/config.yaml', 'r') as f:")
    print("    config = yaml.safe_load(f)")
    print("generator = create_generator_from_config(config)")


if __name__ == "__main__":
    main()

