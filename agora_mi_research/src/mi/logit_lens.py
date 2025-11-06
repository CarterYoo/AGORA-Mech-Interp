"""
Logit Lens implementation for layer-wise interpretation.

This module implements the Logit Lens technique for projecting hidden states
at each layer to vocabulary space, revealing how predictions evolve across
model depth.

Based on: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger
from transformers import PreTrainedModel, PreTrainedTokenizer


class LogitLens:
    """
    Logit Lens analyzer for transformer models.
    
    Projects hidden states from intermediate layers to vocabulary space
    to understand how the model's predictions evolve across layers.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_layers: int = 32
    ):
        """
        Initialize Logit Lens analyzer.
        
        Args:
            model: HuggingFace transformer model
            tokenizer: Corresponding tokenizer
            num_layers: Number of transformer layers
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        
        if hasattr(model, 'lm_head'):
            self.lm_head = model.lm_head
        elif hasattr(model, 'embed_out'):
            self.lm_head = model.embed_out
        else:
            logger.warning("Could not find language modeling head")
            self.lm_head = None
        
        if hasattr(model.model, 'norm') or hasattr(model, 'norm'):
            self.final_norm = model.model.norm if hasattr(model.model, 'norm') else model.norm
        else:
            logger.warning("Could not find final layer normalization")
            self.final_norm = None
        
        logger.info(
            f"Initialized LogitLens: "
            f"layers={num_layers}, "
            f"lm_head={'found' if self.lm_head else 'not found'}"
        )
    
    def project_hidden_state(
        self,
        hidden_state: torch.Tensor,
        apply_norm: bool = True
    ) -> torch.Tensor:
        """
        Project hidden state to vocabulary space.
        
        Args:
            hidden_state: Hidden state tensor [hidden_dim] or [batch, hidden_dim]
            apply_norm: Whether to apply final layer normalization
        
        Returns:
            Logits tensor [vocab_size] or [batch, vocab_size]
        """
        if self.lm_head is None:
            logger.error("Language modeling head not available")
            return torch.zeros(self.tokenizer.vocab_size)
        
        if apply_norm and self.final_norm is not None:
            hidden_state = self.final_norm(hidden_state)
        
        with torch.no_grad():
            logits = self.lm_head(hidden_state)
        
        return logits
    
    def get_top_k_predictions(
        self,
        logits: torch.Tensor,
        k: int = 10
    ) -> List[Dict]:
        """
        Get top-k predictions from logits.
        
        Args:
            logits: Logits tensor [vocab_size]
            k: Number of top predictions
        
        Returns:
            List of dictionaries with token, probability, and rank
        """
        if logits.dim() > 1:
            logits = logits[0]
        
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, k=min(k, len(probs)))
        
        predictions = []
        for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
            token_id = int(idx)
            token_str = self.tokenizer.decode([token_id])
            
            predictions.append({
                'rank': rank,
                'token_id': token_id,
                'token': token_str,
                'probability': float(prob)
            })
        
        return predictions
    
    def analyze_layer_predictions(
        self,
        hidden_states_by_layer: List[torch.Tensor],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Analyze predictions at each layer.
        
        Args:
            hidden_states_by_layer: List of hidden states for each layer
            top_k: Number of top predictions to track
        
        Returns:
            List of layer prediction dictionaries
        """
        layer_predictions = []
        
        for layer_idx, hidden_state in enumerate(hidden_states_by_layer):
            logits = self.project_hidden_state(hidden_state, apply_norm=True)
            
            top_predictions = self.get_top_k_predictions(logits, k=top_k)
            
            confidence = top_predictions[0]['probability'] if top_predictions else 0.0
            
            layer_predictions.append({
                'layer': layer_idx,
                'top_predictions': top_predictions,
                'top_1_token': top_predictions[0]['token'] if top_predictions else '',
                'top_1_confidence': confidence
            })
        
        logger.debug(
            f"Analyzed predictions for {len(hidden_states_by_layer)} layers"
        )
        
        return layer_predictions
    
    def track_prediction_evolution(
        self,
        hidden_states_by_layer: List[torch.Tensor],
        true_token_id: Optional[int] = None
    ) -> Dict:
        """
        Track how predictions evolve across layers.
        
        Args:
            hidden_states_by_layer: List of hidden states for each layer
            true_token_id: Optional true token ID to track its probability
        
        Returns:
            Dictionary with evolution metrics
        """
        layer_predictions = self.analyze_layer_predictions(hidden_states_by_layer)
        
        top1_tokens = [pred['top_1_token'] for pred in layer_predictions]
        confidences = [pred['top_1_confidence'] for pred in layer_predictions]
        
        prediction_changes = sum(
            1 for i in range(1, len(top1_tokens))
            if top1_tokens[i] != top1_tokens[i-1]
        )
        
        convergence_layer = None
        if len(top1_tokens) > 1:
            final_token = top1_tokens[-1]
            for layer_idx in range(len(top1_tokens) - 1, -1, -1):
                if top1_tokens[layer_idx] != final_token:
                    convergence_layer = layer_idx + 1
                    break
            if convergence_layer is None:
                convergence_layer = 0
        
        true_token_trajectory = None
        if true_token_id is not None:
            true_token_trajectory = []
            for layer_idx, hidden_state in enumerate(hidden_states_by_layer):
                logits = self.project_hidden_state(hidden_state, apply_norm=True)
                probs = torch.softmax(logits, dim=-1)
                if logits.dim() > 1:
                    true_token_prob = float(probs[0, true_token_id])
                else:
                    true_token_prob = float(probs[true_token_id])
                true_token_trajectory.append(true_token_prob)
        
        evolution = {
            'layer_predictions': layer_predictions,
            'confidence_trajectory': confidences,
            'prediction_changes': prediction_changes,
            'convergence_layer': convergence_layer,
            'final_prediction': top1_tokens[-1] if top1_tokens else '',
            'final_confidence': confidences[-1] if confidences else 0.0,
            'true_token_trajectory': true_token_trajectory
        }
        
        logger.info(
            f"Prediction evolution: {prediction_changes} changes, "
            f"convergence at layer {convergence_layer}"
        )
        
        return evolution
    
    def identify_hallucination_indicators(
        self,
        evolution: Dict,
        late_convergence_threshold: int = 25,
        low_confidence_threshold: float = 0.5,
        high_change_threshold: int = 5
    ) -> Dict:
        """
        Identify potential hallucination indicators from evolution.
        
        Args:
            evolution: Prediction evolution dictionary
            late_convergence_threshold: Layer threshold for late convergence
            low_confidence_threshold: Confidence threshold
            high_change_threshold: Number of changes threshold
        
        Returns:
            Dictionary with hallucination indicators
        """
        indicators = {
            'late_convergence': False,
            'low_final_confidence': False,
            'high_prediction_changes': False,
            'unstable_predictions': False
        }
        
        if evolution['convergence_layer'] is not None:
            if evolution['convergence_layer'] >= late_convergence_threshold:
                indicators['late_convergence'] = True
        
        if evolution['final_confidence'] < low_confidence_threshold:
            indicators['low_final_confidence'] = True
        
        if evolution['prediction_changes'] >= high_change_threshold:
            indicators['high_prediction_changes'] = True
        
        if len(evolution['confidence_trajectory']) > 5:
            recent_confidences = evolution['confidence_trajectory'][-5:]
            confidence_std = np.std(recent_confidences)
            if confidence_std > 0.1:
                indicators['unstable_predictions'] = True
        
        num_indicators = sum(indicators.values())
        
        indicators['total_indicators'] = num_indicators
        indicators['hallucination_likely'] = num_indicators >= 2
        
        logger.info(
            f"Hallucination indicators: {num_indicators} detected, "
            f"likely={indicators['hallucination_likely']}"
        )
        
        return indicators
    
    def analyze_token_sequence(
        self,
        hidden_states_sequence: List[List[torch.Tensor]],
        true_token_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Analyze logit lens for a sequence of tokens.
        
        Args:
            hidden_states_sequence: List of [layer_hidden_states] for each token
            true_token_ids: Optional list of true token IDs
        
        Returns:
            List of evolution dictionaries for each token
        """
        sequence_analysis = []
        
        for token_idx, token_layer_states in enumerate(hidden_states_sequence):
            true_token_id = true_token_ids[token_idx] if true_token_ids else None
            
            evolution = self.track_prediction_evolution(
                token_layer_states,
                true_token_id=true_token_id
            )
            
            indicators = self.identify_hallucination_indicators(evolution)
            
            sequence_analysis.append({
                'token_idx': token_idx,
                'evolution': evolution,
                'hallucination_indicators': indicators
            })
        
        logger.info(f"Analyzed {len(sequence_analysis)} tokens in sequence")
        
        return sequence_analysis
    
    def compute_confidence_statistics(
        self,
        sequence_analysis: List[Dict]
    ) -> Dict:
        """
        Compute statistics on confidence trajectories.
        
        Args:
            sequence_analysis: List of token analysis dictionaries
        
        Returns:
            Dictionary with confidence statistics
        """
        all_final_confidences = [
            analysis['evolution']['final_confidence']
            for analysis in sequence_analysis
        ]
        
        all_convergence_layers = [
            analysis['evolution']['convergence_layer']
            for analysis in sequence_analysis
            if analysis['evolution']['convergence_layer'] is not None
        ]
        
        all_prediction_changes = [
            analysis['evolution']['prediction_changes']
            for analysis in sequence_analysis
        ]
        
        likely_hallucinations = sum(
            1 for analysis in sequence_analysis
            if analysis['hallucination_indicators']['hallucination_likely']
        )
        
        stats = {
            'mean_final_confidence': float(np.mean(all_final_confidences)) if all_final_confidences else 0.0,
            'std_final_confidence': float(np.std(all_final_confidences)) if all_final_confidences else 0.0,
            'mean_convergence_layer': float(np.mean(all_convergence_layers)) if all_convergence_layers else 0.0,
            'mean_prediction_changes': float(np.mean(all_prediction_changes)) if all_prediction_changes else 0.0,
            'num_likely_hallucinations': likely_hallucinations,
            'hallucination_rate': likely_hallucinations / len(sequence_analysis) if sequence_analysis else 0.0
        }
        
        logger.info(f"Confidence statistics: {stats}")
        
        return stats


def main():
    """
    Example usage of LogitLens.
    
    Note: Requires actual model and tokenizer to run.
    """
    print("LogitLens implementation complete.")
    print("To use LogitLens, initialize with a model and tokenizer:")
    print()
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')")
    print("tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')")
    print("logit_lens = LogitLens(model, tokenizer)")
    print()
    print("Then analyze hidden states:")
    print("evolution = logit_lens.track_prediction_evolution(hidden_states_by_layer)")
    print("indicators = logit_lens.identify_hallucination_indicators(evolution)")


if __name__ == "__main__":
    main()

