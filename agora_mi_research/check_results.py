"""
Quick check: Print test results from Granite Guardian
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from src.annotation.granite_guardian import GraniteGuardianAnnotator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {device}\n")

annotator = GraniteGuardianAnnotator(
    model_variant='latest',
    use_vllm=False,
    device=device
)

print("Loaded Granite Guardian\n")

# Simple test
question = "What funding is provided?"
context = "The Act provides $500 million for AI."
response = "The Act provides $500 million for AI and defense systems."

print("Test: Military hallucination")
print(f"Question: {question}")
print(f"Context: {context}")
print(f"Response: {response}")
print()

result = annotator.detect_hallucination(question, context, response)

print("RESULT:")
print(f"  score: {result['score']}")
print(f"  is_hallucination: {result['is_hallucination']}")
print()

if result['score'] == 'yes':
    print("✅ Detected hallucination correctly!")
elif result['score'] == 'no':
    print("❌ Failed to detect hallucination")
    print("   (Should be 'yes' - defense systems not in context)")
else:
    print(f"⚠️  Unexpected score: {result['score']}")

print(f"\nReasoning (first 300 chars):")
print(result.get('reasoning', 'None')[:300])
print()

print("Raw response (last 200 chars):")
print(result.get('raw_response', 'None')[-200:])

