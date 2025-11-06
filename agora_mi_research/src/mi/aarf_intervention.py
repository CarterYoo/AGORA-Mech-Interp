"""
AARF (Add Attention Reduce FFN) intervention module.

This module implements real-time hallucination mitigation by dynamically
modifying attention patterns and FFN outputs during generation.

Based on ReDEeP framework: https://github.com/Jeryi-Sun/ReDEeP-ICLR
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from loguru import logger

from .data_structures import InterventionEvent


class HallucinationScoreCalculator:
    """
    Calculate hallucination probability from ECS and PKS scores.
    
    Formula: h_score = alpha * (1 - ECS) + beta * PKS
    where alpha + beta = 1
    
    High ECS (low 1-ECS) and low PKS indicate factual responses.
    Low ECS (high 1-ECS) and high PKS indicate hallucination risk.
    """
    
    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        threshold: float = 0.6
    ):
        """
        Initialize hallucination score calculator.
        
        Args:
            alpha: Weight for (1 - ECS) component
            beta: Weight for PKS component
            threshold: Threshold for triggering intervention
        """
        if not np.isclose(alpha + beta, 1.0):
            logger.warning(
                f"Alpha ({alpha}) + Beta ({beta}) != 1.0, "
                "normalizing to sum to 1.0"
            )
            total = alpha + beta
            alpha /= total
            beta /= total
        
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        
        logger.info(
            f"Initialized HallucinationScoreCalculator: "
            f"alpha={alpha:.2f}, beta={beta:.2f}, threshold={threshold:.2f}"
        )
    
    def compute_hallucination_score(
        self,
        ecs: float,
        pks: float
    ) -> float:
        """
        Compute hallucination score from ECS and PKS.
        
        Args:
            ecs: External Context Score [0, 1]
            pks: Parametric Knowledge Score [0, 1]
        
        Returns:
            Hallucination score [0, 1], higher indicates more hallucination risk
        """
        if not (0 <= ecs <= 1 and 0 <= pks <= 1):
            logger.warning(
                f"ECS or PKS out of range: ECS={ecs}, PKS={pks}, "
                "clamping to [0, 1]"
            )
            ecs = max(0.0, min(1.0, ecs))
            pks = max(0.0, min(1.0, pks))
        
        h_score = self.alpha * (1.0 - ecs) + self.beta * pks
        
        return float(np.clip(h_score, 0.0, 1.0))
    
    def should_intervene(
        self,
        ecs: float,
        pks: float
    ) -> Tuple[bool, float]:
        """
        Determine if intervention should be triggered.
        
        Args:
            ecs: External Context Score
            pks: Parametric Knowledge Score
        
        Returns:
            Tuple of (should_intervene, hallucination_score)
        """
        h_score = self.compute_hallucination_score(ecs, pks)
        should_trigger = h_score >= self.threshold
        
        return should_trigger, h_score


class AARFIntervention:
    """
    AARF (Add Attention Reduce FFN) intervention for real-time hallucination mitigation.
    
    Dynamically modifies model behavior during generation:
    - Amplifies attention to context tokens in copying heads
    - Suppresses FFN output to reduce parametric knowledge injection
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        copying_heads: List[Dict],
        context_start: int,
        context_end: int,
        attention_multiplier: float = 1.5,
        ffn_suppress: float = 0.7
    ):
        """
        Initialize AARF intervention.
        
        Args:
            model: Transformer model to intervene on
            tokenizer: Tokenizer for token position mapping
            copying_heads: List of copying head dictionaries with 'layer' and 'head' keys
            context_start: Token index where context starts
            context_end: Token index where context ends
            attention_multiplier: Multiplier for copying head attention (default: 1.5)
            ffn_suppress: Suppression factor for FFN output (default: 0.7)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.copying_heads = copying_heads
        self.context_start = context_start
        self.context_end = context_end
        self.attention_multiplier = attention_multiplier
        self.ffn_suppress = ffn_suppress
        
        self.hooks = []
        self.intervention_active = False
        self.intervention_stats = {
            'attention_modifications': 0,
            'ffn_suppressions': 0,
            'total_tokens': 0
        }
        self.intervention_events: List[InterventionEvent] = []
        self.generation_step = 0
        self.current_token_position = 0
        
        logger.info(
            f"Initialized AARFIntervention: "
            f"{len(copying_heads)} copying heads, "
            f"context=[{context_start}, {context_end}), "
            f"att_mult={attention_multiplier}, "
            f"ffn_suppress={ffn_suppress}"
        )
    
    def _create_attention_hook(
        self,
        layer_idx: int,
        head_idx: int
    ) -> Callable:
        """
        Create attention hook for specific layer and head.
        
        Modifies attention weights to amplify context tokens in copying heads.
        Note: This hook modifies attention weights before they are used in computation.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
        
        Returns:
            Hook function
        """
        def attention_hook(module, input, output):
            if not self.intervention_active:
                return output
            
            if isinstance(output, tuple):
                attention_output, attention_weights = output
            else:
                attention_output = output
                attention_weights = None
            
            if attention_weights is not None and attention_weights.numel() > 0:
                batch_size, num_heads, seq_len, seq_len = attention_weights.shape
                
                if head_idx < num_heads and self.context_end <= seq_len:
                    for batch_idx in range(batch_size):
                        if self.context_start < self.context_end:
                            before_value = float(
                                attention_weights[
                                    batch_idx, head_idx, :, self.context_start:self.context_end
                                ].mean()
                            )
                            
                            attention_weights[
                                batch_idx, head_idx, :, self.context_start:self.context_end
                            ] *= self.attention_multiplier
                            
                            after_value = float(
                                attention_weights[
                                    batch_idx, head_idx, :, self.context_start:self.context_end
                                ].mean()
                            )
                            
                            self.intervention_stats['attention_modifications'] += 1
                            
                            event = InterventionEvent(
                                generation_step=self.generation_step,
                                token_position=self.current_token_position,
                                intervention_type="attention",
                                layer_idx=layer_idx,
                                head_idx=head_idx,
                                context_start=self.context_start,
                                context_end=self.context_end,
                                attention_multiplier=self.attention_multiplier,
                                before_value=before_value,
                                after_value=after_value
                            )
                            self.intervention_events.append(event)
            
            if isinstance(output, tuple):
                return (attention_output, attention_weights)
            else:
                return attention_output
        
        return attention_hook
    
    def _create_ffn_hook(self, layer_idx: int) -> Callable:
        """
        Create FFN hook for specific layer.
        
        Args:
            layer_idx: Layer index
        
        Returns:
            Hook function
        """
        def ffn_hook(module, input, output):
            if not self.intervention_active:
                return output
            
            before_value = float(output.mean()) if output.numel() > 0 else 0.0
            
            suppressed_output = output * self.ffn_suppress
            self.intervention_stats['ffn_suppressions'] += 1
            
            after_value = float(suppressed_output.mean()) if suppressed_output.numel() > 0 else 0.0
            
            event = InterventionEvent(
                generation_step=self.generation_step,
                token_position=self.current_token_position,
                intervention_type="ffn",
                layer_idx=layer_idx,
                context_start=self.context_start,
                context_end=self.context_end,
                ffn_suppress=self.ffn_suppress,
                before_value=before_value,
                after_value=after_value
            )
            self.intervention_events.append(event)
            
            return suppressed_output
        
        return ffn_hook
    
    def _get_transformer_layers(self) -> List[Tuple[nn.Module, int]]:
        """
        Get list of transformer layer modules and their indices.
        
        Returns:
            List of (layer_module, layer_idx) tuples
        """
        layers = []
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for idx, layer in enumerate(self.model.model.layers):
                layers.append((layer, idx))
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            for idx, layer in enumerate(self.model.transformer.h):
                layers.append((layer, idx))
        elif hasattr(self.model, 'layers'):
            for idx, layer in enumerate(self.model.layers):
                layers.append((layer, idx))
        else:
            logger.warning("Could not find transformer layers in model")
        
        return layers
    
    def apply_intervention(self):
        """
        Apply AARF intervention by registering forward hooks.
        """
        if self.intervention_active:
            logger.warning("Intervention already active")
            return
        
        layers = self._get_transformer_layers()
        
        if not layers:
            logger.error("No transformer layers found, cannot apply intervention")
            return
        
        copying_head_dict = {
            (ch['layer'], ch['head']): ch for ch in self.copying_heads
        }
        
        for layer_module, layer_idx in layers:
            if hasattr(layer_module, 'self_attn'):
                attention_module = layer_module.self_attn
                
                for head_dict in self.copying_heads:
                    if head_dict['layer'] == layer_idx:
                        head_idx = head_dict['head']
                        hook = self._create_attention_hook(layer_idx, head_idx)
                        handle = attention_module.register_forward_hook(hook)
                        self.hooks.append(handle)
            
            if hasattr(layer_module, 'mlp'):
                ffn_module = layer_module.mlp
                hook = self._create_ffn_hook(layer_idx)
                handle = ffn_module.register_forward_hook(hook)
                self.hooks.append(handle)
            elif hasattr(layer_module, 'feed_forward'):
                ffn_module = layer_module.feed_forward
                hook = self._create_ffn_hook(layer_idx)
                handle = ffn_module.register_forward_hook(hook)
                self.hooks.append(handle)
        
        self.intervention_active = True
        logger.info(
            f"AARF intervention applied: {len(self.hooks)} hooks registered "
            f"({len(copying_head_dict)} attention hooks, "
            f"{len(layers)} FFN hooks)"
        )
    
    def remove_intervention(self):
        """
        Remove AARF intervention by unregistering all hooks.
        """
        if not self.intervention_active:
            logger.warning("Intervention not active")
            return
        
        for hook in self.hooks:
            hook.remove()
        
        self.hooks.clear()
        self.intervention_active = False
        
        logger.info(
            f"AARF intervention removed. Stats: "
            f"{self.intervention_stats['attention_modifications']} attention mods, "
            f"{self.intervention_stats['ffn_suppressions']} FFN suppressions, "
            f"{self.intervention_stats['total_tokens']} total tokens"
        )
    
    def reset_stats(self):
        """Reset intervention statistics."""
        self.intervention_stats = {
            'attention_modifications': 0,
            'ffn_suppressions': 0,
            'total_tokens': 0
        }
    
    def get_stats(self) -> Dict:
        """Get intervention statistics."""
        return {
            "attention_modifications": self.intervention_stats['attention_modifications'],
            "ffn_suppressions": self.intervention_stats['ffn_suppressions'],
            "total_tokens": self.intervention_stats['total_tokens'],
            "num_events": len(self.intervention_events),
            "intervention_events": [e.to_dict() for e in self.intervention_events]
        }
    
    def get_intervention_events(self) -> List[InterventionEvent]:
        """Returns list of intervention events."""
        return self.intervention_events
    
    def update_generation_step(self, step: int, token_pos: int):
        """Update current generation step and token position for event tracking."""
        self.generation_step = step
        self.current_token_position = token_pos
    
    def update_context_boundaries(
        self,
        context_start: int,
        context_end: int
    ):
        """
        Update context token boundaries for intervention.
        
        Args:
            context_start: New context start position
            context_end: New context end position
        """
        self.context_start = context_start
        self.context_end = context_end
        logger.debug(
            f"Updated context boundaries: [{context_start}, {context_end})"
        )

