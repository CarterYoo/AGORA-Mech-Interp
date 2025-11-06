"""
Parametric Knowledge Score (PKS) calculator module.

This module implements the Parametric Knowledge Score metric, which quantifies
the model's reliance on its internal parametric knowledge versus external context.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger
import torch.nn.functional as F


class PKSCalculator:
    """
    Calculator for Parametric Knowledge Score (PKS).
    
    PKS measures the model's reliance on parametric (internal) knowledge through:
    1. Prediction confidence
    2. Hidden state consistency
    3. Distribution entropy (inverted)
    
    High PKS suggests strong parametric knowledge usage.
    """
    
    def __init__(
        self,
        confidence_weight: float = 0.4,
        consistency_weight: float = 0.3,
        entropy_weight: float = 0.3,
        vocab_size: int = 32000
    ):
        """
        Initialize PKS calculator.
        
        Args:
            confidence_weight: Weight for confidence component
            consistency_weight: Weight for consistency component
            entropy_weight: Weight for entropy component
            vocab_size: Size of model vocabulary
        """
        if not np.isclose(confidence_weight + consistency_weight + entropy_weight, 1.0):
            logger.warning(
                f"Component weights sum to "
                f"{confidence_weight + consistency_weight + entropy_weight}, "
                "normalizing to 1.0"
            )
            total = confidence_weight + consistency_weight + entropy_weight
            confidence_weight /= total
            consistency_weight /= total
            entropy_weight /= total
        
        self.confidence_weight = confidence_weight
        self.consistency_weight = consistency_weight
        self.entropy_weight = entropy_weight
        self.vocab_size = vocab_size
        
        logger.info(
            f"Initialized PKSCalculator: "
            f"conf_w={confidence_weight:.2f}, "
            f"cons_w={consistency_weight:.2f}, "
            f"ent_w={entropy_weight:.2f}"
        )
    
    def compute_confidence(
        self,
        logits: torch.Tensor
    ) -> float:
        """
        Compute prediction confidence from logits.
        
        Confidence is the maximum probability in the softmax distribution.
        
        Args:
            logits: Logits tensor [vocab_size] or [batch_size, vocab_size]
        
        Returns:
            Confidence score [0, 1]
        """
        if logits.dim() > 1:
            logits = logits[0]
        
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs.max())
        
        return confidence
    
    def compute_consistency(
        self,
        hidden_state_current: torch.Tensor,
        hidden_state_previous: torch.Tensor
    ) -> float:
        """
        Compute consistency between consecutive hidden states.
        
        Consistency is measured using cosine similarity.
        
        Args:
            hidden_state_current: Current hidden state [hidden_dim]
            hidden_state_previous: Previous hidden state [hidden_dim]
        
        Returns:
            Consistency score [-1, 1] (normalized to [0, 1])
        """
        if hidden_state_current.dim() > 1:
            hidden_state_current = hidden_state_current.squeeze()
        
        if hidden_state_previous.dim() > 1:
            hidden_state_previous = hidden_state_previous.squeeze()
        
        cosine_sim = F.cosine_similarity(
            hidden_state_current.unsqueeze(0),
            hidden_state_previous.unsqueeze(0),
            dim=1
        )
        
        consistency = (float(cosine_sim) + 1) / 2
        
        return consistency
    
    def compute_entropy(
        self,
        logits: torch.Tensor
    ) -> float:
        """
        Compute entropy of prediction distribution.
        
        Low entropy indicates confident, focused predictions (high parametric knowledge).
        
        Args:
            logits: Logits tensor [vocab_size] or [batch_size, vocab_size]
        
        Returns:
            Normalized entropy [0, 1]
        """
        if logits.dim() > 1:
            logits = logits[0]
        
        probs = torch.softmax(logits, dim=-1)
        
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum()
        
        max_entropy = np.log(min(len(probs), self.vocab_size))
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def compute_low_entropy_score(
        self,
        logits: torch.Tensor
    ) -> float:
        """
        Compute inverted entropy as a component of PKS.
        
        Args:
            logits: Logits tensor
        
        Returns:
            Low entropy score [0, 1]
        """
        entropy = self.compute_entropy(logits)
        low_entropy_score = 1.0 - entropy
        
        return low_entropy_score
    
    def compute_token_pks(
        self,
        logits: torch.Tensor,
        hidden_state_current: Optional[torch.Tensor] = None,
        hidden_state_previous: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Compute PKS for a single token.
        
        PKS = w_conf × confidence + w_cons × consistency + w_ent × (1 - entropy)
        
        Args:
            logits: Logits for current token
            hidden_state_current: Current hidden state (optional)
            hidden_state_previous: Previous hidden state (optional)
        
        Returns:
            Dictionary with PKS components and overall score
        """
        confidence = self.compute_confidence(logits)
        
        if hidden_state_current is not None and hidden_state_previous is not None:
            consistency = self.compute_consistency(
                hidden_state_current, hidden_state_previous
            )
        else:
            consistency = 0.5
            logger.debug("No hidden states provided, using default consistency=0.5")
        
        low_entropy = self.compute_low_entropy_score(logits)
        
        pks = (
            self.confidence_weight * confidence +
            self.consistency_weight * consistency +
            self.entropy_weight * low_entropy
        )
        
        return {
            'pks': float(pks),
            'confidence': float(confidence),
            'consistency': float(consistency),
            'low_entropy': float(low_entropy),
            'components': {
                'confidence_contribution': float(self.confidence_weight * confidence),
                'consistency_contribution': float(self.consistency_weight * consistency),
                'entropy_contribution': float(self.entropy_weight * low_entropy)
            }
        }
    
    def compute_sequence_pks(
        self,
        logits_sequence: List[torch.Tensor],
        hidden_states_sequence: Optional[List[torch.Tensor]] = None,
        return_trajectory: bool = False
    ) -> Dict:
        """
        Compute PKS for a sequence of tokens.
        
        Args:
            logits_sequence: List of logits tensors for each token
            hidden_states_sequence: Optional list of hidden states
            return_trajectory: Whether to return TrajectoryData object
        
        Returns:
            Dictionary with sequence PKS statistics and optionally TrajectoryData
        """
        token_pks_list = []
        confidence_list = []
        consistency_list = []
        low_entropy_list = []
        
        for i, logits in enumerate(logits_sequence):
            if hidden_states_sequence and i > 0:
                hidden_current = hidden_states_sequence[i]
                hidden_previous = hidden_states_sequence[i-1]
            else:
                hidden_current = None
                hidden_previous = None
            
            token_pks = self.compute_token_pks(
                logits, hidden_current, hidden_previous
            )
            
            token_pks_list.append(token_pks['pks'])
            confidence_list.append(token_pks['confidence'])
            consistency_list.append(token_pks['consistency'])
            low_entropy_list.append(token_pks['low_entropy'])
        
        result = {
            'token_pks': token_pks_list,
            'overall_pks': float(np.mean(token_pks_list)) if token_pks_list else 0.0,
            'pks_statistics': {
                'min': float(min(token_pks_list)) if token_pks_list else 0.0,
                'max': float(max(token_pks_list)) if token_pks_list else 0.0,
                'std': float(np.std(token_pks_list)) if token_pks_list else 0.0
            },
            'component_averages': {
                'confidence': float(np.mean(confidence_list)) if confidence_list else 0.0,
                'consistency': float(np.mean(consistency_list)) if consistency_list else 0.0,
                'low_entropy': float(np.mean(low_entropy_list)) if low_entropy_list else 0.0
            },
            'num_tokens': len(token_pks_list)
        }
        
        if return_trajectory:
            trajectory = TrajectoryData()
            trajectory.token_pks = np.array(token_pks_list)
            trajectory.pks_confidence = np.array(confidence_list)
            trajectory.pks_consistency = np.array(consistency_list)
            trajectory.pks_entropy = np.array(low_entropy_list)
            result['trajectory'] = trajectory
            
            logger.debug(
                f"Computed PKS trajectory: "
                f"token_pks shape={trajectory.token_pks.shape}, "
                f"confidence shape={trajectory.pks_confidence.shape}, "
                f"consistency shape={trajectory.pks_consistency.shape}, "
                f"entropy shape={trajectory.pks_entropy.shape}"
            )
        
        return result
    
    def compute_layer_pks(
        self,
        hidden_states_by_layer: List[torch.Tensor],
        final_logits: torch.Tensor
    ) -> List[Dict]:
        """
        Compute PKS evolution across layers.
        
        Args:
            hidden_states_by_layer: List of hidden states for each layer
            final_logits: Final layer logits
        
        Returns:
            List of PKS dictionaries for each layer
        """
        layer_pks_list = []
        
        for layer_idx in range(len(hidden_states_by_layer) - 1):
            current_state = hidden_states_by_layer[layer_idx + 1]
            previous_state = hidden_states_by_layer[layer_idx]
            
            consistency = self.compute_consistency(current_state, previous_state)
            
            layer_pks_list.append({
                'layer': layer_idx,
                'consistency': float(consistency)
            })
        
        if len(hidden_states_by_layer) > 0:
            final_confidence = self.compute_confidence(final_logits)
            final_entropy = self.compute_entropy(final_logits)
            
            layer_pks_list.append({
                'layer': len(hidden_states_by_layer) - 1,
                'confidence': float(final_confidence),
                'entropy': float(final_entropy),
                'low_entropy': float(1.0 - final_entropy)
            })
        
        return layer_pks_list
    
    def compute_pks_analysis(
        self,
        logits_sequence: List[torch.Tensor],
        hidden_states_sequence: Optional[List[torch.Tensor]] = None,
        hidden_states_by_layer: Optional[List[List[torch.Tensor]]] = None
    ) -> Dict:
        """
        Perform comprehensive PKS analysis.
        
        Args:
            logits_sequence: List of logits for each generated token
            hidden_states_sequence: Optional list of final hidden states
            hidden_states_by_layer: Optional list of hidden states per layer per token
        
        Returns:
            Comprehensive PKS analysis dictionary
        """
        logger.info("Computing comprehensive PKS analysis")
        
        sequence_pks = self.compute_sequence_pks(
            logits_sequence, hidden_states_sequence
        )
        
        layer_pks = []
        if hidden_states_by_layer:
            for token_idx in range(len(hidden_states_by_layer)):
                token_layer_states = hidden_states_by_layer[token_idx]
                if len(logits_sequence) > token_idx:
                    token_layer_pks = self.compute_layer_pks(
                        token_layer_states, logits_sequence[token_idx]
                    )
                    layer_pks.append(token_layer_pks)
        
        analysis = {
            'overall_pks': sequence_pks['overall_pks'],
            'token_pks': sequence_pks['token_pks'],
            'pks_statistics': sequence_pks['pks_statistics'],
            'component_averages': sequence_pks['component_averages'],
            'layer_pks': layer_pks if layer_pks else None,
            'num_tokens': sequence_pks['num_tokens']
        }
        
        logger.info(
            f"PKS analysis complete: overall_pks={analysis['overall_pks']:.3f}"
        )
        
        return analysis
    
    def compute_p_vs_e_ratio(
        self,
        pks: float,
        ecs: float
    ) -> Dict:
        """
        Compute Parametric vs External knowledge ratio.
        
        Args:
            pks: Parametric Knowledge Score
            ecs: External Context Score
        
        Returns:
            Dictionary with P vs E analysis
        """
        if pks + ecs == 0:
            ratio = 0.0
            dominant = 'none'
        elif pks > ecs:
            ratio = pks / (pks + ecs)
            dominant = 'parametric'
        elif ecs > pks:
            ratio = ecs / (pks + ecs)
            dominant = 'external'
        else:
            ratio = 0.5
            dominant = 'balanced'
        
        return {
            'pks': float(pks),
            'ecs': float(ecs),
            'ratio': float(ratio),
            'dominant_knowledge': dominant,
            'pks_proportion': float(pks / (pks + ecs)) if (pks + ecs) > 0 else 0.0,
            'ecs_proportion': float(ecs / (pks + ecs)) if (pks + ecs) > 0 else 0.0
        }
    
    def compare_pks_distributions(
        self,
        pks_list_1: List[float],
        pks_list_2: List[float],
        labels: Tuple[str, str] = ("Group 1", "Group 2")
    ) -> Dict:
        """
        Compare PKS distributions between two groups.
        
        Args:
            pks_list_1: PKS values for first group
            pks_list_2: PKS values for second group
            labels: Labels for the two groups
        
        Returns:
            Dictionary with comparison statistics
        """
        comparison = {
            labels[0]: {
                'mean': float(np.mean(pks_list_1)),
                'std': float(np.std(pks_list_1)),
                'median': float(np.median(pks_list_1)),
                'n': len(pks_list_1)
            },
            labels[1]: {
                'mean': float(np.mean(pks_list_2)),
                'std': float(np.std(pks_list_2)),
                'median': float(np.median(pks_list_2)),
                'n': len(pks_list_2)
            },
            'difference': {
                'mean_diff': float(np.mean(pks_list_1) - np.mean(pks_list_2)),
                'median_diff': float(np.median(pks_list_1) - np.median(pks_list_2))
            }
        }
        
        logger.info(
            f"PKS comparison: {labels[0]} mean={comparison[labels[0]]['mean']:.3f}, "
            f"{labels[1]} mean={comparison[labels[1]]['mean']:.3f}, "
            f"diff={comparison['difference']['mean_diff']:.3f}"
        )
        
        return comparison


def main():
    """
    Example usage of PKSCalculator.
    """
    calculator = PKSCalculator()
    
    logits = torch.randn(32000)
    logits[100] = 10.0
    
    hidden_current = torch.randn(4096)
    hidden_previous = torch.randn(4096)
    hidden_previous += hidden_current * 0.8
    
    print("Computing PKS for single token...\n")
    
    token_pks = calculator.compute_token_pks(
        logits, hidden_current, hidden_previous
    )
    
    print(f"PKS: {token_pks['pks']:.4f}")
    print(f"Components:")
    print(f"  Confidence: {token_pks['confidence']:.4f}")
    print(f"  Consistency: {token_pks['consistency']:.4f}")
    print(f"  Low Entropy: {token_pks['low_entropy']:.4f}")
    
    logits_sequence = [torch.randn(32000) for _ in range(10)]
    hidden_sequence = [torch.randn(4096) for _ in range(10)]
    
    print("\nComputing PKS for sequence...")
    
    sequence_pks = calculator.compute_sequence_pks(
        logits_sequence, hidden_sequence
    )
    
    print(f"Overall PKS: {sequence_pks['overall_pks']:.4f}")
    print(f"Component averages:")
    for component, value in sequence_pks['component_averages'].items():
        print(f"  {component}: {value:.4f}")
    
    pks_value = sequence_pks['overall_pks']
    ecs_value = 0.45
    
    print(f"\nComputing P vs E ratio...")
    p_vs_e = calculator.compute_p_vs_e_ratio(pks_value, ecs_value)
    
    print(f"PKS: {p_vs_e['pks']:.4f}")
    print(f"ECS: {p_vs_e['ecs']:.4f}")
    print(f"Dominant knowledge: {p_vs_e['dominant_knowledge']}")
    print(f"PKS proportion: {p_vs_e['pks_proportion']:.4f}")
    print(f"ECS proportion: {p_vs_e['ecs_proportion']:.4f}")


if __name__ == "__main__":
    main()

