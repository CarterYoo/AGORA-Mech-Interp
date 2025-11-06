"""
Attention pattern analysis module.

This module extracts and analyzes attention patterns from transformer layers
to understand model behavior and identify copying heads.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger
from pathlib import Path
import h5py


class AttentionAnalyzer:
    """
    Analyzer for transformer attention patterns.
    
    Extracts attention tensors from model outputs and computes various
    attention-based metrics for mechanistic interpretability.
    """
    
    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32
    ):
        """
        Initialize attention analyzer.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        logger.info(
            f"Initialized AttentionAnalyzer: "
            f"layers={num_layers}, heads={num_heads}"
        )
    
    def extract_attention_from_generate(
        self,
        model_outputs: Dict,
        input_length: int
    ) -> Optional[torch.Tensor]:
        """
        Extract attention tensors from model generation outputs.
        
        Args:
            model_outputs: Dictionary from model.generate()
            input_length: Length of input sequence
        
        Returns:
            Attention tensor [num_generated_tokens, num_layers, num_heads, seq_len, seq_len]
            or None if attention not available
        """
        if not hasattr(model_outputs, 'attentions') or not model_outputs.attentions:
            logger.warning("No attention tensors in model outputs")
            return None
        
        logger.info(
            f"Extracting attention from {len(model_outputs.attentions)} generation steps"
        )
        
        return model_outputs.attentions
    
    def compute_attention_statistics(
        self,
        attention: torch.Tensor,
        token_idx: int
    ) -> Dict:
        """
        Compute statistics for attention at a specific token position.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            token_idx: Token position to analyze
        
        Returns:
            Dictionary with attention statistics
        """
        if token_idx >= attention.shape[2]:
            logger.warning(f"Token index {token_idx} out of range")
            return {}
        
        token_attention = attention[:, :, token_idx, :]
        
        stats = {
            'mean_attention': float(token_attention.mean()),
            'max_attention': float(token_attention.max()),
            'min_attention': float(token_attention.min()),
            'std_attention': float(token_attention.std()),
            'entropy': self._compute_attention_entropy(token_attention)
        }
        
        return stats
    
    def _compute_attention_entropy(
        self,
        attention: torch.Tensor,
        epsilon: float = 1e-10
    ) -> float:
        """
        Compute entropy of attention distribution.
        
        Args:
            attention: Attention tensor
            epsilon: Small constant for numerical stability
        
        Returns:
            Mean entropy across layers and heads
        """
        attention = attention + epsilon
        log_attention = torch.log(attention)
        entropy = -(attention * log_attention).sum(dim=-1)
        
        return float(entropy.mean())
    
    def identify_attention_peaks(
        self,
        attention: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        token_idx: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Identify positions with highest attention.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            layer_idx: Layer index
            head_idx: Head index
            token_idx: Token position to analyze
            top_k: Number of top positions to return
        
        Returns:
            List of (position, attention_weight) tuples
        """
        if layer_idx >= attention.shape[0] or head_idx >= attention.shape[1]:
            logger.warning(f"Invalid layer {layer_idx} or head {head_idx}")
            return []
        
        token_attention = attention[layer_idx, head_idx, token_idx, :]
        
        top_values, top_indices = torch.topk(token_attention, k=min(top_k, len(token_attention)))
        
        peaks = [
            (int(idx), float(val))
            for idx, val in zip(top_indices, top_values)
        ]
        
        return peaks
    
    def compute_layer_attention_distribution(
        self,
        attention: torch.Tensor
    ) -> Dict:
        """
        Compute attention distribution across layers.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
        
        Returns:
            Dictionary mapping layer_idx to attention statistics
        """
        layer_stats = {}
        
        for layer_idx in range(attention.shape[0]):
            layer_attention = attention[layer_idx]
            
            layer_stats[layer_idx] = {
                'mean': float(layer_attention.mean()),
                'std': float(layer_attention.std()),
                'max': float(layer_attention.max()),
                'sparsity': self._compute_sparsity(layer_attention)
            }
        
        return layer_stats
    
    def _compute_sparsity(
        self,
        attention: torch.Tensor,
        threshold: float = 0.1
    ) -> float:
        """
        Compute sparsity of attention (proportion below threshold).
        
        Args:
            attention: Attention tensor
            threshold: Threshold for considering attention as sparse
        
        Returns:
            Sparsity ratio [0, 1]
        """
        below_threshold = (attention < threshold).float()
        sparsity = float(below_threshold.mean())
        
        return sparsity
    
    def compute_head_attention_patterns(
        self,
        attention: torch.Tensor,
        layer_idx: int
    ) -> Dict:
        """
        Compute attention patterns for all heads in a layer.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            layer_idx: Layer index to analyze
        
        Returns:
            Dictionary mapping head_idx to attention patterns
        """
        if layer_idx >= attention.shape[0]:
            logger.warning(f"Invalid layer index {layer_idx}")
            return {}
        
        layer_attention = attention[layer_idx]
        head_patterns = {}
        
        for head_idx in range(layer_attention.shape[0]):
            head_attention = layer_attention[head_idx]
            
            head_patterns[head_idx] = {
                'mean': float(head_attention.mean()),
                'max': float(head_attention.max()),
                'entropy': self._compute_attention_entropy(head_attention.unsqueeze(0)),
                'diagonal_strength': self._compute_diagonal_strength(head_attention)
            }
        
        return head_patterns
    
    def _compute_diagonal_strength(
        self,
        attention: torch.Tensor
    ) -> float:
        """
        Compute strength of attention along diagonal (self-attention).
        
        Args:
            attention: Attention matrix [seq_len, seq_len]
        
        Returns:
            Mean attention on diagonal
        """
        if attention.shape[0] != attention.shape[1]:
            return 0.0
        
        diagonal = torch.diagonal(attention)
        return float(diagonal.mean())
    
    def analyze_attention_flow(
        self,
        attention: torch.Tensor,
        source_positions: List[int],
        target_position: int
    ) -> Dict:
        """
        Analyze attention flow from source positions to target position.
        
        Args:
            attention: Attention tensor [layers, heads, seq_len, seq_len]
            source_positions: List of source token positions
            target_position: Target token position
        
        Returns:
            Dictionary with attention flow analysis
        """
        if target_position >= attention.shape[2]:
            logger.warning(f"Target position {target_position} out of range")
            return {}
        
        flow_by_layer = {}
        
        for layer_idx in range(attention.shape[0]):
            layer_attention = attention[layer_idx, :, target_position, :]
            
            source_attention = layer_attention[:, source_positions]
            
            flow_by_layer[layer_idx] = {
                'mean_flow': float(source_attention.mean()),
                'max_flow': float(source_attention.max()),
                'total_flow': float(source_attention.sum()),
                'proportion_from_sources': float(
                    source_attention.sum() / layer_attention.sum()
                ) if layer_attention.sum() > 0 else 0.0
            }
        
        overall_flow = sum(
            flow_by_layer[l]['proportion_from_sources']
            for l in flow_by_layer
        ) / len(flow_by_layer)
        
        return {
            'by_layer': flow_by_layer,
            'overall_proportion': overall_flow
        }
    
    def save_attention_to_hdf5(
        self,
        attention: torch.Tensor,
        output_path: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save attention tensor to HDF5 file for efficient storage.
        
        Args:
            attention: Attention tensor to save
            output_path: Path to output HDF5 file
            metadata: Optional metadata to store
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(attention, torch.Tensor):
            attention_np = attention.cpu().numpy()
        else:
            attention_np = np.array(attention)
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'attention',
                data=attention_np,
                compression='gzip',
                compression_opts=4
            )
            
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value
        
        logger.info(f"Saved attention tensor to {output_path}")
    
    def load_attention_from_hdf5(
        self,
        input_path: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load attention tensor from HDF5 file.
        
        Args:
            input_path: Path to HDF5 file
        
        Returns:
            Tuple of (attention_array, metadata_dict)
        """
        with h5py.File(input_path, 'r') as f:
            attention = f['attention'][:]
            
            metadata = dict(f.attrs)
        
        logger.info(f"Loaded attention tensor from {input_path}: shape {attention.shape}")
        
        return attention, metadata


def main():
    """
    Example usage of AttentionAnalyzer.
    """
    analyzer = AttentionAnalyzer(num_layers=32, num_heads=32)
    
    attention = torch.rand(32, 32, 50, 50)
    attention = torch.softmax(attention, dim=-1)
    
    print("Analyzing attention patterns...\n")
    
    stats = analyzer.compute_attention_statistics(attention, token_idx=25)
    print(f"Attention statistics for token 25:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    layer_stats = analyzer.compute_layer_attention_distribution(attention)
    print(f"\nLayer-wise attention distribution:")
    for layer_idx in [0, 15, 31]:
        print(f"  Layer {layer_idx}:")
        for key, value in layer_stats[layer_idx].items():
            print(f"    {key}: {value:.4f}")
    
    peaks = analyzer.identify_attention_peaks(
        attention, layer_idx=15, head_idx=10, token_idx=25, top_k=5
    )
    print(f"\nTop-5 attention peaks for layer 15, head 10, token 25:")
    for pos, weight in peaks:
        print(f"  Position {pos}: {weight:.4f}")
    
    flow = analyzer.analyze_attention_flow(
        attention,
        source_positions=[5, 10, 15],
        target_position=25
    )
    print(f"\nAttention flow from positions [5, 10, 15] to 25:")
    print(f"  Overall proportion: {flow['overall_proportion']:.4f}")


if __name__ == "__main__":
    main()

