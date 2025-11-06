"""
External Context Score (ECS) calculator module.

This module implements the External Context Score metric from RAGTruth methodology,
which quantifies how much attention the model places on retrieved context versus
parametric knowledge.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger

from .data_structures import TrajectoryData


class ECSCalculator:
    """
    Calculator for External Context Score (ECS).
    
    ECS measures the proportion of attention directed to retrieved context tokens,
    indicating the model's reliance on external information versus parametric knowledge.
    
    Based on RAGTruth methodology: https://arxiv.org/abs/2401.00396
    """
    
    def __init__(
        self,
        high_ecs_threshold: float = 0.3,
        num_layers: int = 32,
        num_heads: int = 32
    ):
        """
        Initialize ECS calculator.
        
        Args:
            high_ecs_threshold: Threshold for identifying copying heads
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
        """
        self.high_ecs_threshold = high_ecs_threshold
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        logger.info(
            f"Initialized ECSCalculator: threshold={high_ecs_threshold}, "
            f"layers={num_layers}, heads={num_heads}"
        )
    
    def compute_token_ecs(
        self,
        attention: torch.Tensor,
        context_start: int,
        context_end: int,
        token_idx: int
    ) -> float:
        """
        Compute ECS for a single generated token.
        
        ECS(token) = mean attention to context tokens across all layers and heads
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            context_start: Start position of context
            context_end: End position of context
            token_idx: Token position to analyze
        
        Returns:
            ECS score [0, 1]
        """
        if token_idx >= attention.shape[2]:
            logger.warning(f"Token index {token_idx} out of range")
            return 0.0
        
        if context_start >= context_end or context_end > attention.shape[3]:
            logger.warning(
                f"Invalid context boundaries: [{context_start}, {context_end})"
            )
            return 0.0
        
        token_attention = attention[:, :, token_idx, :]
        
        context_attention = token_attention[:, :, context_start:context_end]
        
        ecs = float(context_attention.mean())
        
        return ecs
    
    def compute_sequence_ecs(
        self,
        attention: torch.Tensor,
        context_start: int,
        context_end: int,
        generated_start: int,
        generated_end: Optional[int] = None,
        return_trajectory: bool = False
    ) -> Dict:
        """
        Compute ECS for a sequence of generated tokens.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            context_start: Start position of context
            context_end: End position of context
            generated_start: Start position of generated tokens
            generated_end: End position of generated tokens (None = all remaining)
            return_trajectory: Whether to return TrajectoryData object
        
        Returns:
            Dictionary with ECS statistics and optionally TrajectoryData
        """
        if generated_end is None:
            generated_end = attention.shape[2]
        
        token_ecs_list = []
        layer_ecs_matrix = []
        head_ecs_tensor = []
        
        num_layers = attention.shape[0]
        num_heads = attention.shape[1]
        
        for token_idx in range(generated_start, generated_end):
            token_ecs = self.compute_token_ecs(
                attention, context_start, context_end, token_idx
            )
            token_ecs_list.append(token_ecs)
            
            if return_trajectory:
                layer_ecs_for_token = []
                head_ecs_for_token = []
                
                for layer_idx in range(num_layers):
                    layer_attention = attention[layer_idx, :, token_idx, context_start:context_end]
                    layer_ecs_val = float(layer_attention.mean())
                    layer_ecs_for_token.append(layer_ecs_val)
                    
                    head_ecs_for_head = []
                    for head_idx in range(num_heads):
                        head_attention = attention[layer_idx, head_idx, token_idx, context_start:context_end]
                        head_ecs_val = float(head_attention.mean())
                        head_ecs_for_head.append(head_ecs_val)
                    head_ecs_for_token.append(head_ecs_for_head)
                
                layer_ecs_matrix.append(layer_ecs_for_token)
                head_ecs_tensor.append(head_ecs_for_token)
        
        overall_ecs = np.mean(token_ecs_list) if token_ecs_list else 0.0
        
        result = {
            'token_ecs': token_ecs_list,
            'overall_ecs': float(overall_ecs),
            'min_ecs': float(min(token_ecs_list)) if token_ecs_list else 0.0,
            'max_ecs': float(max(token_ecs_list)) if token_ecs_list else 0.0,
            'std_ecs': float(np.std(token_ecs_list)) if token_ecs_list else 0.0,
            'num_tokens': len(token_ecs_list)
        }
        
        if return_trajectory:
            trajectory = TrajectoryData()
            trajectory.token_ecs = np.array(token_ecs_list)
            trajectory.layer_ecs = np.array(layer_ecs_matrix).T  # [num_layers, num_tokens]
            trajectory.head_ecs = np.array(head_ecs_tensor).transpose(0, 2, 1)  # [num_layers, num_heads, num_tokens]
            result['trajectory'] = trajectory
            
            logger.debug(
                f"Computed ECS trajectory: "
                f"token_ecs shape={trajectory.token_ecs.shape}, "
                f"layer_ecs shape={trajectory.layer_ecs.shape}, "
                f"head_ecs shape={trajectory.head_ecs.shape}"
            )
        
        return result
    
    def compute_layer_ecs(
        self,
        attention: torch.Tensor,
        context_start: int,
        context_end: int,
        generated_start: int,
        generated_end: Optional[int] = None
    ) -> List[Dict]:
        """
        Compute ECS for each layer.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            context_start: Start position of context
            context_end: End position of context
            generated_start: Start position of generated tokens
            generated_end: End position of generated tokens
        
        Returns:
            List of dictionaries with per-layer ECS
        """
        if generated_end is None:
            generated_end = attention.shape[2]
        
        layer_ecs_list = []
        
        for layer_idx in range(attention.shape[0]):
            layer_attention = attention[layer_idx:layer_idx+1]
            
            layer_ecs = []
            for token_idx in range(generated_start, generated_end):
                token_attention = layer_attention[0, :, token_idx, context_start:context_end]
                layer_ecs.append(float(token_attention.mean()))
            
            layer_ecs_list.append({
                'layer': layer_idx,
                'ecs': float(np.mean(layer_ecs)) if layer_ecs else 0.0,
                'std': float(np.std(layer_ecs)) if layer_ecs else 0.0
            })
        
        return layer_ecs_list
    
    def identify_copying_heads(
        self,
        attention: torch.Tensor,
        context_start: int,
        context_end: int,
        generated_start: int,
        generated_end: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Identify attention heads that consistently attend to context (copying heads).
        
        A head is considered a copying head if its ECS exceeds the threshold.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            context_start: Start position of context
            context_end: End position of context
            generated_start: Start position of generated tokens
            generated_end: End position of generated tokens
            threshold: ECS threshold (uses default if None)
        
        Returns:
            List of copying head dictionaries sorted by ECS (descending)
        """
        if generated_end is None:
            generated_end = attention.shape[2]
        
        if threshold is None:
            threshold = self.high_ecs_threshold
        
        copying_heads = []
        
        for layer_idx in range(attention.shape[0]):
            for head_idx in range(attention.shape[1]):
                head_ecs_list = []
                
                for token_idx in range(generated_start, generated_end):
                    head_attention = attention[
                        layer_idx, head_idx, token_idx, context_start:context_end
                    ]
                    head_ecs_list.append(float(head_attention.mean()))
                
                mean_ecs = np.mean(head_ecs_list) if head_ecs_list else 0.0
                
                if mean_ecs >= threshold:
                    copying_heads.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'ecs': float(mean_ecs),
                        'std': float(np.std(head_ecs_list)) if head_ecs_list else 0.0
                    })
        
        copying_heads.sort(key=lambda x: x['ecs'], reverse=True)
        
        logger.info(
            f"Identified {len(copying_heads)} copying heads "
            f"(threshold={threshold:.3f})"
        )
        
        return copying_heads
    
    def get_copying_heads_by_layer(
        self,
        copying_heads: List[Dict]
    ) -> Dict[int, List[Dict]]:
        """
        Organize copying heads by layer index.
        
        Args:
            copying_heads: List of copying head dictionaries from identify_copying_heads()
        
        Returns:
            Dictionary mapping layer index to list of copying heads in that layer
        """
        heads_by_layer = {}
        
        for head in copying_heads:
            layer_idx = head['layer']
            if layer_idx not in heads_by_layer:
                heads_by_layer[layer_idx] = []
            heads_by_layer[layer_idx].append(head)
        
        return heads_by_layer
    
    def get_copying_head_indices(
        self,
        copying_heads: List[Dict]
    ) -> List[Tuple[int, int]]:
        """
        Extract (layer, head) index tuples from copying heads list.
        
        Args:
            copying_heads: List of copying head dictionaries
        
        Returns:
            List of (layer_idx, head_idx) tuples
        """
        return [(head['layer'], head['head']) for head in copying_heads]
    
    def compute_context_token_ratio(
        self,
        context_start: int,
        context_end: int,
        total_length: int
    ) -> float:
        """
        Compute ratio of context tokens to total tokens.
        
        Args:
            context_start: Start position of context
            context_end: End position of context
            total_length: Total sequence length
        
        Returns:
            Context ratio [0, 1]
        """
        if total_length <= 0:
            return 0.0
        
        context_length = context_end - context_start
        ratio = context_length / total_length
        
        return float(ratio)
    
    def compute_ecs_analysis(
        self,
        attention: torch.Tensor,
        context_start: int,
        context_end: int,
        generated_start: int,
        generated_end: Optional[int] = None
    ) -> Dict:
        """
        Perform complete ECS analysis.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            context_start: Start position of context
            context_end: End position of context
            generated_start: Start position of generated tokens
            generated_end: End position of generated tokens
        
        Returns:
            Comprehensive ECS analysis dictionary
        """
        logger.info("Computing comprehensive ECS analysis")
        
        sequence_ecs = self.compute_sequence_ecs(
            attention, context_start, context_end,
            generated_start, generated_end
        )
        
        layer_ecs = self.compute_layer_ecs(
            attention, context_start, context_end,
            generated_start, generated_end
        )
        
        copying_heads = self.identify_copying_heads(
            attention, context_start, context_end,
            generated_start, generated_end
        )
        
        context_ratio = self.compute_context_token_ratio(
            context_start, context_end, attention.shape[2]
        )
        
        analysis = {
            'overall_ecs': sequence_ecs['overall_ecs'],
            'token_ecs': sequence_ecs['token_ecs'],
            'ecs_statistics': {
                'min': sequence_ecs['min_ecs'],
                'max': sequence_ecs['max_ecs'],
                'std': sequence_ecs['std_ecs']
            },
            'layer_ecs': layer_ecs,
            'copying_heads': copying_heads,
            'num_copying_heads': len(copying_heads),
            'context_token_ratio': context_ratio,
            'context_boundaries': {
                'start': context_start,
                'end': context_end,
                'length': context_end - context_start
            },
            'generated_boundaries': {
                'start': generated_start,
                'end': generated_end if generated_end else attention.shape[2],
                'length': (generated_end if generated_end else attention.shape[2]) - generated_start
            }
        }
        
        logger.info(
            f"ECS analysis complete: overall_ecs={analysis['overall_ecs']:.3f}, "
            f"copying_heads={len(copying_heads)}"
        )
        
        return analysis
    
    def compute_ecs_trajectory(
        self,
        attention: torch.Tensor,
        context_start: int,
        context_end: int,
        generated_start: int,
        generated_end: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute ECS trajectory across layers for visualization.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            context_start: Start position of context
            context_end: End position of context
            generated_start: Start position of generated tokens
            generated_end: End position of generated tokens
        
        Returns:
            Array of shape [num_layers, num_generated_tokens]
        """
        if generated_end is None:
            generated_end = attention.shape[2]
        
        num_generated = generated_end - generated_start
        trajectory = np.zeros((self.num_layers, num_generated))
        
        for layer_idx in range(self.num_layers):
            for i, token_idx in enumerate(range(generated_start, generated_end)):
                layer_attention = attention[
                    layer_idx, :, token_idx, context_start:context_end
                ]
                trajectory[layer_idx, i] = float(layer_attention.mean())
        
        logger.debug(f"Computed ECS trajectory: shape {trajectory.shape}")
        
        return trajectory
    
    def compare_ecs_distributions(
        self,
        ecs_list_1: List[float],
        ecs_list_2: List[float],
        labels: Tuple[str, str] = ("Group 1", "Group 2")
    ) -> Dict:
        """
        Compare ECS distributions between two groups.
        
        Args:
            ecs_list_1: ECS values for first group
            ecs_list_2: ECS values for second group
            labels: Labels for the two groups
        
        Returns:
            Dictionary with comparison statistics
        """
        comparison = {
            labels[0]: {
                'mean': float(np.mean(ecs_list_1)),
                'std': float(np.std(ecs_list_1)),
                'median': float(np.median(ecs_list_1)),
                'n': len(ecs_list_1)
            },
            labels[1]: {
                'mean': float(np.mean(ecs_list_2)),
                'std': float(np.std(ecs_list_2)),
                'median': float(np.median(ecs_list_2)),
                'n': len(ecs_list_2)
            },
            'difference': {
                'mean_diff': float(np.mean(ecs_list_1) - np.mean(ecs_list_2)),
                'median_diff': float(np.median(ecs_list_1) - np.median(ecs_list_2))
            }
        }
        
        logger.info(
            f"ECS comparison: {labels[0]} mean={comparison[labels[0]]['mean']:.3f}, "
            f"{labels[1]} mean={comparison[labels[1]]['mean']:.3f}, "
            f"diff={comparison['difference']['mean_diff']:.3f}"
        )
        
        return comparison


def main():
    """
    Example usage of ECSCalculator.
    """
    calculator = ECSCalculator(high_ecs_threshold=0.3)
    
    attention = torch.rand(32, 32, 100, 100)
    attention = torch.softmax(attention, dim=-1)
    
    context_start = 10
    context_end = 60
    generated_start = 70
    
    print("Computing ECS analysis...\n")
    
    analysis = calculator.compute_ecs_analysis(
        attention,
        context_start=context_start,
        context_end=context_end,
        generated_start=generated_start
    )
    
    print(f"Overall ECS: {analysis['overall_ecs']:.4f}")
    print(f"Number of copying heads: {analysis['num_copying_heads']}")
    print(f"Context token ratio: {analysis['context_token_ratio']:.4f}")
    
    print(f"\nTop-5 copying heads:")
    for i, head in enumerate(analysis['copying_heads'][:5], 1):
        print(
            f"  {i}. Layer {head['layer']}, Head {head['head']}: "
            f"ECS={head['ecs']:.4f}"
        )
    
    print(f"\nLayer-wise ECS (first 5 layers):")
    for layer_info in analysis['layer_ecs'][:5]:
        print(
            f"  Layer {layer_info['layer']}: "
            f"ECS={layer_info['ecs']:.4f} ± {layer_info['std']:.4f}"
        )
    
    trajectory = calculator.compute_ecs_trajectory(
        attention, context_start, context_end, generated_start
    )
    print(f"\nECS trajectory shape: {trajectory.shape}")


if __name__ == "__main__":
    main()

