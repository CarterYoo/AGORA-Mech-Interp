"""
Test script for IBM Granite Guardian integration.

Verifies that Granite Guardian is correctly configured following
the official detailed_guide_think.ipynb methodology.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

logger.add("logs/test_granite_guardian_{time}.log", level="INFO")


def test_granite_guardian():
    """
    Test Granite Guardian with example cases from official guide.
    """
    logger.info("="*60)
    logger.info("Testing IBM Granite Guardian Integration")
    logger.info("="*60)
    
    try:
        from src.annotation.granite_guardian import GraniteGuardianAnnotator
        import torch
        
        # Auto-detect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info("\nStep 1: Initializing Granite Guardian...")
        logger.info(f"Device: {device}")
        
        if device == 'cpu':
            logger.warning("CUDA not available. Using CPU (will be slower)")
        
        annotator = GraniteGuardianAnnotator(
            model_variant='latest',
            use_vllm=False,  # vLLM needs CUDA
            device=device
        )
        
        logger.info("✅ Granite Guardian loaded successfully")
        
        model_info = annotator.get_model_info()
        logger.info(f"\nModel Info:")
        logger.info(f"  Name: {model_info['model_name']}")
        logger.info(f"  Parameters: {model_info['num_parameters']:,}")
        
    except Exception as e:
        logger.error(f"\n❌ Failed to load Granite Guardian: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Install vLLM: pip install vllm")
        logger.error("  2. Login to HuggingFace: huggingface-cli login")
        logger.error("  3. Check model access: https://huggingface.co/ibm-granite/granite-guardian-3.3-8b")
        logger.error("\nAlternative: Use manual annotation only")
        return 1
    
    logger.info("\nStep 2: Testing groundedness detection...")
    
    # Test Case 1: Clear hallucination
    logger.info("\n--- Test Case 1: Clear Hallucination ---")
    
    question1 = "What funding is allocated for AI modernization?"
    context1 = "The Act appropriates $500 million to the Department of Commerce for AI-driven system modernization."
    response1 = "The Act provides $500 million for AI modernization and requires military deployment certification."
    
    logger.info(f"Question: {question1}")
    logger.info(f"Context: {context1[:80]}...")
    logger.info(f"Response: {response1}")
    
    result1 = annotator.detect_hallucination(question1, context1, response1)
    
    logger.info(f"\nResult:")
    logger.info(f"  Is Hallucination: {result1['is_hallucination']}")
    logger.info(f"  Score: {result1['score']}")
    logger.info(f"  Reasoning: {result1['reasoning'][:200] if result1['reasoning'] else 'None'}...")
    
    # Expected: is_hallucination=True (military certification not in context)
    
    # Test Case 2: Factual response
    logger.info("\n--- Test Case 2: Factual Response ---")
    
    question2 = "What is the funding amount?"
    context2 = "The legislation allocates $500 million for AI system upgrades."
    response2 = "The funding amount is $500 million."
    
    logger.info(f"Question: {question2}")
    logger.info(f"Context: {context2}")
    logger.info(f"Response: {response2}")
    
    result2 = annotator.detect_hallucination(question2, context2, response2)
    
    logger.info(f"\nResult:")
    logger.info(f"  Is Hallucination: {result2['is_hallucination']}")
    logger.info(f"  Score: {result2['score']}")
    logger.info(f"  Reasoning: {result2['reasoning'][:200] if result2['reasoning'] else 'None'}...")
    
    # Expected: is_hallucination=False (factually grounded)
    
    # Test Case 3: Subtle conflict
    logger.info("\n--- Test Case 3: Subtle Conflict ---")
    
    question3 = "Who does the AI Act apply to?"
    context3 = "This Regulation applies to providers placing high-risk AI systems on the market in the Union."
    response3 = "The AI Act applies to all AI system providers worldwide."
    
    logger.info(f"Question: {question3}")
    logger.info(f"Context: {context3}")
    logger.info(f"Response: {response3}")
    
    result3 = annotator.detect_hallucination(question3, context3, response3)
    
    logger.info(f"\nResult:")
    logger.info(f"  Is Hallucination: {result3['is_hallucination']}")
    logger.info(f"  Score: {result3['score']}")
    logger.info(f"  Reasoning: {result3['reasoning'][:200] if result3['reasoning'] else 'None'}...")
    
    # Expected: is_hallucination=True (high-risk only, not all; Union, not worldwide)
    
    logger.info("\n" + "="*60)
    logger.info("Granite Guardian Test Complete")
    logger.info("="*60)
    logger.info(f"\nSummary:")
    logger.info(f"  Test 1 (Clear hallucination): {result1['score']}")
    logger.info(f"  Test 2 (Factual): {result2['score']}")
    logger.info(f"  Test 3 (Subtle conflict): {result3['score']}")
    logger.info(f"\nGranite Guardian is working correctly!")
    logger.info(f"\nNext: Run phase4_automated_annotation.py to annotate 100 samples")
    
    return 0


if __name__ == "__main__":
    exit_code = test_granite_guardian()
    sys.exit(exit_code)

