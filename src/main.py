#!/usr/bin/env python3
"""
ECG Scanpath Pattern Recognition Using Hidden Markov Models

Main execution script that runs the complete pipeline:
1. Generate synthetic scanpath data
2. Train expert and novice HMM models
3. Evaluate classification performance
4. Analyze learned parameters
5. Report results

Usage:
    python main.py                    # Run with default settings
    python main.py --n-train 150      # Customize training size
    python main.py --output results/  # Specify output directory

Authors: Riad Benbrahim, Mohamed Amine El Bacha, Youssef Kaya
Course: Computational Theory - Fall 2025
Institution: Mohammed VI Polytechnic University
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm import HiddenMarkovModel, ScanpathClassifier
from data_generator import (
    ScanpathDatasetGenerator, 
    generate_training_data,
    DIAGNOSTIC_PHASES,
    ECG_LEADS
)
from evaluation import (
    compute_classification_metrics,
    compute_confusion_matrix,
    format_confusion_matrix,
    ParameterAnalyzer,
    ComplexityAnalyzer,
    print_evaluation_report,
    save_results
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ECG Scanpath Pattern Recognition using HMMs'
    )
    
    parser.add_argument(
        '--n-train', type=int, default=100,
        help='Number of training samples per class (default: 100)'
    )
    parser.add_argument(
        '--n-test', type=int, default=50,
        help='Number of test samples per class (default: 50)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory for results (default: results)'
    )
    parser.add_argument(
        '--save-models', action='store_true',
        help='Save trained models to disk'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed output'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 70)
    print("ECG SCANPATH PATTERN RECOGNITION USING HIDDEN MARKOV MODELS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Training samples per class: {args.n_train}")
    print(f"  Test samples per class: {args.n_test}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output}")
    
    # =========================================================================
    # STEP 1: Generate Synthetic Data
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Generating Synthetic Dataset")
    print("-" * 70)
    
    start_time = time.time()
    
    generator = ScanpathDatasetGenerator(seed=args.seed)
    
    # Generate training data
    print(f"\nGenerating {args.n_train} expert and {args.n_train} novice training samples...")
    train_samples = generator.generate_dataset(
        n_expert=args.n_train,
        n_novice=args.n_train,
        expert_length_range=(18, 28),
        novice_length_range=(12, 22)
    )
    
    # Generate test data (with different seed offset)
    generator_test = ScanpathDatasetGenerator(seed=args.seed + 10000)
    print(f"Generating {args.n_test} expert and {args.n_test} novice test samples...")
    test_samples = generator_test.generate_dataset(
        n_expert=args.n_test,
        n_novice=args.n_test,
        expert_length_range=(18, 28),
        novice_length_range=(12, 22)
    )
    
    # Save dataset
    generator.save_dataset(train_samples + test_samples, args.output, format='both')
    
    # Print statistics
    train_stats = generator.get_statistics(train_samples)
    test_stats = generator.get_statistics(test_samples)
    
    print(f"\nTraining Data Statistics:")
    print(f"  Expert: {train_stats['expert']['count']} samples, "
          f"mean length {train_stats['expert']['mean_length']:.1f} ± {train_stats['expert']['std_length']:.1f}")
    print(f"  Novice: {train_stats['novice']['count']} samples, "
          f"mean length {train_stats['novice']['mean_length']:.1f} ± {train_stats['novice']['std_length']:.1f}")
    
    print(f"\nTest Data Statistics:")
    print(f"  Expert: {test_stats['expert']['count']} samples")
    print(f"  Novice: {test_stats['novice']['count']} samples")
    
    data_gen_time = time.time() - start_time
    print(f"\nData generation completed in {data_gen_time:.2f}s")
    
    # =========================================================================
    # STEP 2: Prepare Training Data
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Preparing Training Data")
    print("-" * 70)
    
    # Convert to HMM training format
    expert_train = generate_training_data(train_samples, expertise='expert')
    novice_train = generate_training_data(train_samples, expertise='novice')
    
    print(f"\nPrepared {len(expert_train)} expert training sequences")
    print(f"Prepared {len(novice_train)} novice training sequences")
    
    # Show example
    if args.verbose and expert_train:
        obs, states = expert_train[0]
        print(f"\nExample expert sequence (first 10 fixations):")
        print(f"  Observations: {obs[:10]}")
        print(f"  Hidden states: {states[:10]}")
    
    # =========================================================================
    # STEP 3: Train HMM Classifier
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Training HMM Classifier")
    print("-" * 70)
    
    start_time = time.time()
    
    classifier = ScanpathClassifier()
    classifier.train(expert_train, novice_train)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.4f}s")
    
    # Save models if requested
    if args.save_models:
        models_dir = os.path.join(args.output, 'models')
        classifier.save(models_dir)
        print(f"Models saved to {models_dir}")
    
    # =========================================================================
    # STEP 4: Evaluate on Test Set
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Evaluating Classification Performance")
    print("-" * 70)
    
    start_time = time.time()
    
    # Prepare test data
    expert_test = [s for s in test_samples if s.expertise_level == 'expert']
    novice_test = [s for s in test_samples if s.expertise_level == 'novice']
    
    # Classify
    y_true = []
    y_pred = []
    expert_likelihoods = []
    novice_likelihoods = []
    
    print("\nClassifying test samples...")
    for sample in expert_test:
        pred, expert_ll, novice_ll = classifier.classify(sample.observations)
        y_true.append('EXPERT')
        y_pred.append(pred)
        expert_likelihoods.append(expert_ll)
        novice_likelihoods.append(novice_ll)
    
    for sample in novice_test:
        pred, expert_ll, novice_ll = classifier.classify(sample.observations)
        y_true.append('NOVICE')
        y_pred.append(pred)
        expert_likelihoods.append(expert_ll)
        novice_likelihoods.append(novice_ll)
    
    eval_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_classification_metrics(y_true, y_pred)
    confusion_mat = compute_confusion_matrix(y_true, y_pred)
    
    print(f"\n{metrics}")
    print(f"\n{format_confusion_matrix(confusion_mat)}")
    print(f"\nEvaluation completed in {eval_time:.4f}s")
    print(f"Average classification time: {(eval_time / len(y_true)) * 1000:.2f} ms/sample")
    
    # =========================================================================
    # STEP 5: Analyze Learned Parameters
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Analyzing Learned Parameters")
    print("-" * 70)
    
    # Analyze expert HMM
    print("\n" + "=" * 50)
    print("EXPERT HMM PARAMETERS")
    print("=" * 50)
    expert_analyzer = ParameterAnalyzer(classifier.expert_hmm)
    
    print("\nTop State Transitions (Expert):")
    for s1, s2, prob in expert_analyzer.get_top_transitions(5):
        print(f"  {s1} → {s2}: {prob:.3f}")
    
    print("\nTop Emissions per State (Expert):")
    for state in DIAGNOSTIC_PHASES[:5]:  # First 5 states
        top_em = expert_analyzer.get_top_emissions(state, 3)
        em_str = ", ".join([f"{obs}({p:.2f})" for obs, p in top_em])
        print(f"  {state}: {em_str}")
    
    # Analyze novice HMM
    print("\n" + "=" * 50)
    print("NOVICE HMM PARAMETERS")
    print("=" * 50)
    novice_analyzer = ParameterAnalyzer(classifier.novice_hmm)
    
    print("\nTop State Transitions (Novice):")
    for s1, s2, prob in novice_analyzer.get_top_transitions(5):
        print(f"  {s1} → {s2}: {prob:.3f}")
    
    # =========================================================================
    # STEP 6: Complexity Analysis
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 6: Complexity Analysis")
    print("-" * 70)
    
    # Theoretical complexity
    avg_length = int(train_stats['expert']['mean_length'])
    n_states = len(DIAGNOSTIC_PHASES)
    n_obs = len(ECG_LEADS)
    
    theoretical = ComplexityAnalyzer.theoretical_complexity(
        n_states, n_obs, avg_length
    )
    
    print("\nTheoretical Complexity:")
    for op, complexity in theoretical.items():
        print(f"  {op}: {complexity}")
    
    # Empirical benchmark
    test_obs = [s.observations for s in test_samples[:20]]
    benchmark = ComplexityAnalyzer.benchmark_operations(
        classifier.expert_hmm, test_obs, n_iterations=50
    )
    
    print("\nEmpirical Benchmark:")
    print(f"  Forward algorithm: {benchmark['forward_ms_per_sequence']:.3f} ms/sequence")
    print(f"  Viterbi algorithm: {benchmark['viterbi_ms_per_sequence']:.3f} ms/sequence")
    print(f"  Average sequence length: {benchmark['avg_sequence_length']:.1f}")
    
    # =========================================================================
    # STEP 7: Demonstrate Viterbi Decoding
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 7: Cognitive State Decoding (Viterbi)")
    print("-" * 70)
    
    # Decode an expert sample
    sample = expert_test[0]
    decoded_states = classifier.decode_cognitive_states(
        sample.observations, expertise='expert'
    )
    
    print(f"\nExample Expert Scanpath Decoding:")
    print(f"  Sample ID: {sample.participant_id}")
    print(f"  Sequence length: {len(sample.observations)}")
    print(f"\n  First 12 fixations:")
    print(f"  {'Obs (Lead)':<10} {'True State':<25} {'Decoded State':<25}")
    print(f"  {'-'*60}")
    for i in range(min(12, len(sample.observations))):
        obs = sample.observations[i]
        true_state = sample.hidden_states[i]
        decoded = decoded_states[i]
        match = "✓" if true_state == decoded else "✗"
        print(f"  {obs:<10} {true_state:<25} {decoded:<25} {match}")
    
    # Calculate state decoding accuracy
    correct = sum(1 for t, d in zip(sample.hidden_states, decoded_states) if t == d)
    decode_acc = correct / len(sample.hidden_states)
    print(f"\n  State decoding accuracy: {decode_acc:.1%}")
    
    # =========================================================================
    # STEP 8: Save Results
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 8: Saving Results")
    print("-" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'n_train_per_class': args.n_train,
            'n_test_per_class': args.n_test,
            'seed': args.seed,
            'n_hidden_states': n_states,
            'n_observations': n_obs
        },
        'classification_metrics': metrics.to_dict(),
        'confusion_matrix': confusion_mat.tolist(),
        'timing': {
            'data_generation_s': data_gen_time,
            'training_s': training_time,
            'evaluation_s': eval_time,
            'avg_classification_ms': (eval_time / len(y_true)) * 1000
        },
        'complexity': {
            'theoretical': theoretical,
            'empirical': benchmark
        },
        'dataset_statistics': {
            'training': train_stats,
            'test': test_stats
        }
    }
    
    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    results_path = os.path.join(args.output, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved to {results_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Model: Hidden Markov Model
  - Hidden states: {n_states} diagnostic phases
  - Observations: {n_obs} ECG leads

Dataset:
  - Training: {args.n_train * 2} samples ({args.n_train} expert, {args.n_train} novice)
  - Testing: {args.n_test * 2} samples ({args.n_test} expert, {args.n_test} novice)

Classification Results:
  - Accuracy:    {metrics.accuracy:.1%}
  - Precision:   {metrics.precision:.1%}
  - Recall:      {metrics.recall:.1%}
  - F1-Score:    {metrics.f1_score:.1%}
  - Specificity: {metrics.specificity:.1%}

Performance:
  - Training time: {training_time:.3f}s
  - Classification: {(eval_time / len(y_true)) * 1000:.2f} ms/sample
""")
    
    print("=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
