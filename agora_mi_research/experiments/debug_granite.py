"""
Debug script to check Granite Guardian raw outputs.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.annotation.granite_guardian import GraniteGuardianAnnotator

# Auto-detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize
annotator = GraniteGuardianAnnotator(
    model_variant='latest',
    use_vllm=False,
    device=device
)

print("\n" + "="*60)
print("Testing Granite Guardian with Verbose Output")
print("="*60)

# Test Case: Clear hallucination
question = "What funding is allocated for AI modernization?"
context = "The Act appropriates $500 million to the Department of Commerce for AI-driven system modernization."
response = "The Act provides $500 million for AI modernization and requires military deployment certification."

print(f"\nQuestion: {question}")
print(f"\nContext: {context}")
print(f"\nResponse: {response}")
print(f"\n{'='*60}")
print("Expected: Should detect hallucination (military certification not in context)")
print("="*60)

# Detect
result = annotator.detect_hallucination(question, context, response)

print(f"\n{'='*60}")
print("GRANITE GUARDIAN OUTPUT")
print("="*60)

print(f"\nScore: {result['score']}")
print(f"Is Hallucination: {result['is_hallucination']}")

print(f"\nFull Reasoning:")
print("-"*60)
print(result.get('reasoning', 'No reasoning available'))

print(f"\n{'='*60}")
print("RAW MODEL OUTPUT")
print("="*60)
print(result.get('raw_response', 'No raw response'))

print(f"\n{'='*60}")
print("INTERPRETATION")
print("="*60)

if result['score'] == 'yes':
    print("✅ CORRECT: Granite detected hallucination")
elif result['score'] == 'no':
    print("❌ INCORRECT: Granite did NOT detect hallucination (should have)")
    print("   → This is a FALSE NEGATIVE")
else:
    print(f"⚠️  UNKNOWN: Score = {result['score']}")
    print("   → Check if model output format is correct")

print("\n" + "="*60)
print("If all scores are 'no' or None:")
print("  1. Check if model loaded correctly")
print("  2. Verify groundedness template is working")
print("  3. Try with simpler example")
print("  4. Check raw_response for parsing issues")
print("="*60)

