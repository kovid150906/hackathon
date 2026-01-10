"""
Quick example of using the optimized Narrative Consistency Checker
"""

# First, set your API key (choose one):
# Option 1: Together AI (recommended)
import os
os.environ['TOGETHER_API_KEY'] = 'your-together-api-key-here'

# Option 2: Cerebras (fastest)
# os.environ['CEREBRAS_API_KEY'] = 'your-cerebras-api-key-here'

# Import the checker
from src.pipeline import NarrativeConsistencyChecker

# Initialize (will use config.yaml settings)
checker = NarrativeConsistencyChecker('config.yaml')

# Example narrative and backstory
narrative = """
Captain Nemo was a mysterious figure who commanded the submarine Nautilus. 
He was known for his brilliance in engineering and his deep hatred of imperialism.
He lived in seclusion beneath the waves, having renounced all ties to land-based nations.
Throughout the journey, he displayed both kindness and ruthlessness, saving the protagonists
while also attacking ships he deemed unjust. His past remained largely mysterious, though
hints suggested a tragic loss that drove him to the sea.
"""

# Consistent backstory
consistent_backstory = """
Captain Nemo was once a prince of a colonized nation. After his family was killed by
imperialist forces, he vowed revenge and built the Nautilus using his engineering genius.
He now lives in self-imposed exile beneath the ocean.
"""

# Inconsistent backstory
inconsistent_backstory = """
Captain Nemo grew up as a peaceful farmer who loved his community and nation. He was
known for his pacifism and worked as a government diplomat promoting international 
cooperation. He recently purchased the Nautilus from a naval shipyard.
"""

# Test consistent backstory
print("\n" + "="*80)
print("Testing CONSISTENT backstory...")
print("="*80)
result1 = checker.process_single_example(
    narrative_text=narrative,
    backstory=consistent_backstory,
    narrative_id="example_1"
)
print(f"\nDecision: {result1['decision']}")
print(f"Confidence: {result1['confidence']:.2f}")
print(f"Reasoning: {result1['reasoning'][:200]}...")

# Test inconsistent backstory
print("\n" + "="*80)
print("Testing INCONSISTENT backstory...")
print("="*80)
result2 = checker.process_single_example(
    narrative_text=narrative,
    backstory=inconsistent_backstory,
    narrative_id="example_2"
)
print(f"\nDecision: {result2['decision']}")
print(f"Confidence: {result2['confidence']:.2f}")
print(f"Reasoning: {result2['reasoning'][:200]}...")

print("\n" + "="*80)
print("Done! Check the logs/ folder for detailed execution logs.")
print("="*80)
