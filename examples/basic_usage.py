#!/usr/bin/env python3
"""
Example: Basic HMM Usage for ECG Scanpath Analysis

This script demonstrates the core functionality of the HMM framework:
1. Creating and training an HMM
2. Computing sequence likelihoods
3. Decoding hidden states with Viterbi
4. Classifying expert vs novice patterns

Run this script from the src/ directory:
    cd src && python ../examples/basic_usage.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hmm import HiddenMarkovModel, ScanpathClassifier
from data_generator import ScanpathDatasetGenerator, generate_training_data

print("=" * 60)
print("ECG Scanpath HMM - Basic Usage Example")
print("=" * 60)

# Step 1: Generate some synthetic data
print("\n1. Generating synthetic scanpath data...")
generator = ScanpathDatasetGenerator(seed=42)
samples = generator.generate_dataset(n_expert=50, n_novice=50)

expert_samples = [s for s in samples if s.expertise_level == 'expert']
novice_samples = [s for s in samples if s.expertise_level == 'novice']

print(f"   Generated {len(expert_samples)} expert samples")
print(f"   Generated {len(novice_samples)} novice samples")

# Step 2: Show an example scanpath
print("\n2. Example expert scanpath:")
example = expert_samples[0]
print(f"   Participant: {example.participant_id}")
print(f"   Length: {len(example.observations)} fixations")
print(f"   First 8 observations (ECG leads): {example.observations[:8]}")
print(f"   First 8 hidden states (phases): {example.hidden_states[:8]}")

# Step 3: Train the classifier
print("\n3. Training HMM classifier...")
expert_data = generate_training_data(samples, expertise='expert')
novice_data = generate_training_data(samples, expertise='novice')

classifier = ScanpathClassifier()
classifier.train(expert_data, novice_data)
print("   Training complete!")

# Step 4: Compute likelihood of a sequence
print("\n4. Computing sequence likelihoods...")
test_sequence = expert_samples[5].observations
expert_ll = classifier.expert_hmm.compute_likelihood(test_sequence)
novice_ll = classifier.novice_hmm.compute_likelihood(test_sequence)
print(f"   Test sequence: {test_sequence[:6]}...")
print(f"   Log-likelihood under expert model:  {expert_ll:.2f}")
print(f"   Log-likelihood under novice model:  {novice_ll:.2f}")
print(f"   Higher likelihood under: {'expert' if expert_ll > novice_ll else 'novice'} model")

# Step 5: Classify a scanpath
print("\n5. Classifying a scanpath...")
prediction, exp_ll, nov_ll = classifier.classify(test_sequence)
print(f"   Prediction: {prediction}")
print(f"   (True label: expert)")

# Step 6: Viterbi decoding
print("\n6. Viterbi decoding (inferring cognitive states)...")
decoded_states, log_prob = classifier.expert_hmm.viterbi(test_sequence)
print(f"   Sequence length: {len(test_sequence)}")
print(f"   Decoded cognitive states (first 8):")
for i in range(min(8, len(test_sequence))):
    print(f"      {i+1}. {test_sequence[i]:5} → {decoded_states[i]}")

# Step 7: Batch classification
print("\n7. Batch classification on test data...")
test_expert = [s.observations for s in expert_samples[40:50]]
test_novice = [s.observations for s in novice_samples[40:50]]

expert_preds = classifier.classify_batch(test_expert)
novice_preds = classifier.classify_batch(test_novice)

expert_correct = sum(1 for p in expert_preds if p == 'EXPERT')
novice_correct = sum(1 for p in novice_preds if p == 'NOVICE')

print(f"   Expert samples: {expert_correct}/10 correct")
print(f"   Novice samples: {novice_correct}/10 correct")
print(f"   Total accuracy: {(expert_correct + novice_correct) / 20 * 100:.0f}%")

# Step 8: Save and load model
print("\n8. Saving and loading model...")
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    classifier.save(tmpdir)
    loaded_classifier = ScanpathClassifier.load(tmpdir)
    print("   Model saved and loaded successfully!")
    
    # Verify loaded model works
    pred, _, _ = loaded_classifier.classify(test_sequence)
    print(f"   Loaded model prediction: {pred}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
