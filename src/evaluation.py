"""
Evaluation Module for ECG Scanpath HMM Classification

This module provides:
- Classification metrics (accuracy, precision, recall, F1, specificity)
- Confusion matrix computation
- Parameter analysis utilities
- Results visualization
- Complexity analysis

Authors: Riad Benbrahim, Mohamed Amine El Bacha, Youssef Kaya
Course: Computational Theory - Fall 2025
Institution: Mohammed VI Polytechnic University
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import time


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    def to_dict(self) -> dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'specificity': self.specificity,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives
        }
    
    def __str__(self) -> str:
        return (
            f"Classification Metrics:\n"
            f"  Accuracy:    {self.accuracy:.3f}\n"
            f"  Precision:   {self.precision:.3f}\n"
            f"  Recall:      {self.recall:.3f}\n"
            f"  F1-Score:    {self.f1_score:.3f}\n"
            f"  Specificity: {self.specificity:.3f}\n"
            f"  TP: {self.true_positives}, FP: {self.false_positives}, "
            f"TN: {self.true_negatives}, FN: {self.false_negatives}"
        )


def compute_classification_metrics(y_true: List[str], 
                                   y_pred: List[str],
                                   positive_class: str = 'EXPERT') -> ClassificationMetrics:
    """
    Compute classification metrics for binary expert/novice classification.
    
    Args:
        y_true: Ground truth labels ('EXPERT' or 'NOVICE')
        y_pred: Predicted labels
        positive_class: Which class is considered positive (default: 'EXPERT')
        
    Returns:
        ClassificationMetrics object with all computed metrics
    """
    assert len(y_true) == len(y_pred), "Label lists must have same length"
    
    # Convert to binary (1 for positive class, 0 for negative)
    y_true_binary = [1 if y == positive_class else 0 for y in y_true]
    y_pred_binary = [1 if y == positive_class else 0 for y in y_pred]
    
    # Compute confusion matrix elements
    tp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 0)
    
    # Compute metrics with safe division
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        specificity=specificity,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn
    )


def compute_confusion_matrix(y_true: List[str], 
                            y_pred: List[str],
                            classes: List[str] = ['EXPERT', 'NOVICE']) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class labels
        
    Returns:
        Confusion matrix as numpy array
    """
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for yt, yp in zip(y_true, y_pred):
        cm[class_to_idx[yt], class_to_idx[yp]] += 1
    
    return cm


def format_confusion_matrix(cm: np.ndarray, 
                           classes: List[str] = ['EXPERT', 'NOVICE']) -> str:
    """Format confusion matrix as a string for display."""
    lines = ["Confusion Matrix:"]
    lines.append("              " + "  ".join(f"{c:>8}" for c in classes) + "  (Predicted)")
    
    for i, c in enumerate(classes):
        row = "  ".join(f"{cm[i, j]:>8}" for j in range(len(classes)))
        lines.append(f"{c:>12}  {row}")
    
    lines.append("(Actual)")
    return "\n".join(lines)


class ParameterAnalyzer:
    """Analyze learned HMM parameters for interpretability."""
    
    def __init__(self, hmm):
        """
        Initialize with a trained HMM.
        
        Args:
            hmm: Trained HiddenMarkovModel instance
        """
        self.hmm = hmm
        self.states = hmm.states
        self.observations = hmm.observations
    
    def get_top_transitions(self, n: int = 5) -> List[Tuple[str, str, float]]:
        """
        Get the top n most probable state transitions.
        
        Returns:
            List of (from_state, to_state, probability) tuples
        """
        transitions = []
        for i, s1 in enumerate(self.states):
            for j, s2 in enumerate(self.states):
                if i != j:  # Exclude self-loops
                    transitions.append((s1, s2, self.hmm.A[i, j]))
        
        transitions.sort(key=lambda x: x[2], reverse=True)
        return transitions[:n]
    
    def get_top_emissions(self, state: str, n: int = 3) -> List[Tuple[str, float]]:
        """
        Get the top n most probable emissions for a given state.
        
        Args:
            state: Name of the hidden state
            n: Number of top emissions to return
            
        Returns:
            List of (observation, probability) tuples
        """
        state_idx = self.hmm.state_to_idx[state]
        emissions = [(obs, self.hmm.B[state_idx, i]) 
                    for i, obs in enumerate(self.observations)]
        emissions.sort(key=lambda x: x[1], reverse=True)
        return emissions[:n]
    
    def get_initial_distribution(self) -> List[Tuple[str, float]]:
        """Get initial state distribution sorted by probability."""
        initial = [(state, self.hmm.pi[i]) 
                  for i, state in enumerate(self.states)]
        initial.sort(key=lambda x: x[1], reverse=True)
        return initial
    
    def summarize(self) -> str:
        """Generate a summary of the learned parameters."""
        lines = []
        lines.append("=" * 60)
        lines.append("HMM Parameter Analysis")
        lines.append("=" * 60)
        
        # Initial distribution
        lines.append("\nInitial State Distribution:")
        for state, prob in self.get_initial_distribution():
            if prob > 0.05:
                lines.append(f"  {state}: {prob:.3f}")
        
        # Top transitions
        lines.append("\nTop State Transitions:")
        for s1, s2, prob in self.get_top_transitions(8):
            lines.append(f"  {s1} → {s2}: {prob:.3f}")
        
        # Top emissions per state
        lines.append("\nTop Emissions per State:")
        for state in self.states:
            top_emissions = self.get_top_emissions(state, 3)
            emissions_str = ", ".join([f"{obs}({prob:.2f})" for obs, prob in top_emissions])
            lines.append(f"  {state}:")
            lines.append(f"    {emissions_str}")
        
        return "\n".join(lines)


class ComplexityAnalyzer:
    """Analyze computational complexity of HMM operations."""
    
    @staticmethod
    def theoretical_complexity(n_states: int, n_observations: int, 
                               sequence_length: int) -> Dict[str, str]:
        """
        Return theoretical complexity bounds.
        
        Args:
            n_states: Number of hidden states (N)
            n_observations: Number of observation symbols (M)
            sequence_length: Length of observation sequence (T)
            
        Returns:
            Dictionary with complexity for each operation
        """
        return {
            'forward_time': f"O(N² · T) = O({n_states}² · {sequence_length}) = O({n_states**2 * sequence_length})",
            'forward_space': f"O(N · T) = O({n_states} · {sequence_length}) = O({n_states * sequence_length})",
            'viterbi_time': f"O(N² · T) = O({n_states}² · {sequence_length}) = O({n_states**2 * sequence_length})",
            'viterbi_space': f"O(N · T) = O({n_states} · {sequence_length}) = O({n_states * sequence_length})",
            'training_time': f"O(N_train · T_avg + N² + N · M) = O(N_train · T_avg + {n_states}² + {n_states} · {n_observations})",
        }
    
    @staticmethod
    def benchmark_operations(hmm, test_sequences: List[List[str]], 
                            n_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark actual runtime of HMM operations.
        
        Args:
            hmm: Trained HMM
            test_sequences: List of observation sequences for testing
            n_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with average time per operation in milliseconds
        """
        results = {}
        
        # Benchmark forward algorithm
        start = time.time()
        for _ in range(n_iterations):
            for seq in test_sequences:
                hmm.forward(seq)
        end = time.time()
        total_ops = n_iterations * len(test_sequences)
        results['forward_ms_per_sequence'] = ((end - start) / total_ops) * 1000
        
        # Benchmark Viterbi algorithm
        start = time.time()
        for _ in range(n_iterations):
            for seq in test_sequences:
                hmm.viterbi(seq)
        end = time.time()
        results['viterbi_ms_per_sequence'] = ((end - start) / total_ops) * 1000
        
        # Compute average sequence length
        avg_length = np.mean([len(seq) for seq in test_sequences])
        results['avg_sequence_length'] = avg_length
        
        return results


def cross_validate(classifier_class, 
                   expert_data: List[Tuple[List[str], List[str]]],
                   novice_data: List[Tuple[List[str], List[str]]],
                   n_folds: int = 5,
                   seed: int = 42) -> Dict:
    """
    Perform k-fold cross-validation.
    
    Args:
        classifier_class: Class to instantiate for each fold
        expert_data: Expert training data (observations, states)
        novice_data: Novice training data
        n_folds: Number of folds
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with per-fold and aggregate metrics
    """
    np.random.seed(seed)
    
    # Shuffle data
    expert_data = list(expert_data)
    novice_data = list(novice_data)
    np.random.shuffle(expert_data)
    np.random.shuffle(novice_data)
    
    # Create folds
    expert_fold_size = len(expert_data) // n_folds
    novice_fold_size = len(novice_data) // n_folds
    
    fold_metrics = []
    
    for fold in range(n_folds):
        # Split data
        expert_start = fold * expert_fold_size
        expert_end = expert_start + expert_fold_size
        novice_start = fold * novice_fold_size
        novice_end = novice_start + novice_fold_size
        
        expert_test = expert_data[expert_start:expert_end]
        expert_train = expert_data[:expert_start] + expert_data[expert_end:]
        novice_test = novice_data[novice_start:novice_end]
        novice_train = novice_data[:novice_start] + novice_data[novice_end:]
        
        # Train classifier
        clf = classifier_class()
        clf.train(expert_train, novice_train)
        
        # Evaluate
        y_true = ['EXPERT'] * len(expert_test) + ['NOVICE'] * len(novice_test)
        test_obs = [obs for obs, _ in expert_test] + [obs for obs, _ in novice_test]
        y_pred = clf.classify_batch(test_obs)
        
        metrics = compute_classification_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
    
    # Aggregate results
    accuracies = [m.accuracy for m in fold_metrics]
    f1_scores = [m.f1_score for m in fold_metrics]
    
    return {
        'fold_metrics': [m.to_dict() for m in fold_metrics],
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'n_folds': n_folds
    }


def save_results(metrics: ClassificationMetrics,
                 parameter_summary: str,
                 complexity_info: Dict,
                 filepath: str) -> None:
    """Save evaluation results to a JSON file."""
    results = {
        'classification_metrics': metrics.to_dict(),
        'parameter_summary': parameter_summary,
        'complexity_analysis': complexity_info
    }
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def print_evaluation_report(metrics: ClassificationMetrics,
                           confusion_matrix: np.ndarray,
                           parameter_analyzer: Optional[ParameterAnalyzer] = None,
                           complexity_info: Optional[Dict] = None) -> None:
    """Print a comprehensive evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    
    # Classification metrics
    print("\n" + str(metrics))
    
    # Confusion matrix
    print("\n" + format_confusion_matrix(confusion_matrix))
    
    # Parameter analysis
    if parameter_analyzer is not None:
        print("\n" + parameter_analyzer.summarize())
    
    # Complexity analysis
    if complexity_info is not None:
        print("\n" + "=" * 60)
        print("Complexity Analysis")
        print("=" * 60)
        for key, value in complexity_info.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Evaluation Module Demo")
    print("=" * 40)
    
    # Create some dummy predictions
    y_true = ['EXPERT'] * 45 + ['NOVICE'] * 55
    y_pred = ['EXPERT'] * 43 + ['NOVICE'] * 2 + ['NOVICE'] * 50 + ['EXPERT'] * 5
    
    metrics = compute_classification_metrics(y_true, y_pred)
    print(metrics)
    
    cm = compute_confusion_matrix(y_true, y_pred)
    print("\n" + format_confusion_matrix(cm))
