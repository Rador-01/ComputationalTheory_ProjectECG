"""
Hidden Markov Model Implementation for ECG Scanpath Analysis

This module implements a complete HMM framework including:
- Forward algorithm for likelihood computation
- Viterbi algorithm for optimal state decoding
- Baum-Welch algorithm for parameter learning
- Maximum likelihood estimation from labeled data

Authors: Riad Benbrahim, Mohamed Amine El Bacha, Youssef Kaya
Course: Computational Theory - Fall 2025
Institution: Mohammed VI Polytechnic University
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class HMMParameters:
    """Container for HMM parameters."""
    states: List[str]           # Hidden states (diagnostic phases)
    observations: List[str]     # Observable symbols (ECG leads)
    A: np.ndarray              # Transition probability matrix
    B: np.ndarray              # Emission probability matrix
    pi: np.ndarray             # Initial state distribution


class HiddenMarkovModel:
    """
    Hidden Markov Model for ECG Scanpath Pattern Recognition.
    
    This implementation models ECG interpretation as a sequence of
    hidden cognitive diagnostic phases that emit observable gaze
    fixations on ECG leads.
    
    Hidden States (Diagnostic Phases):
        - Rhythm-Check
        - Axis-Determination
        - P-wave-Analysis
        - PR-interval-Assessment
        - QRS-Analysis
        - ST-segment-Evaluation
        - T-wave-Examination
        - QT-interval-Measurement
        - Lead-by-Lead-Review
    
    Observations (ECG Leads):
        - I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    """
    
    # Define the hidden states (cognitive diagnostic phases)
    HIDDEN_STATES = [
        'Rhythm-Check',
        'Axis-Determination', 
        'P-wave-Analysis',
        'PR-interval-Assessment',
        'QRS-Analysis',
        'ST-segment-Evaluation',
        'T-wave-Examination',
        'QT-interval-Measurement',
        'Lead-by-Lead-Review'
    ]
    
    # Define the observable states (ECG leads)
    OBSERVATIONS = [
        'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ]
    
    def __init__(self, 
                 states: Optional[List[str]] = None,
                 observations: Optional[List[str]] = None,
                 smoothing_alpha: float = 0.01):
        """
        Initialize the HMM.
        
        Args:
            states: List of hidden state names (default: HIDDEN_STATES)
            observations: List of observation symbols (default: OBSERVATIONS)
            smoothing_alpha: Laplace smoothing parameter for handling unseen transitions
        """
        self.states = states if states is not None else self.HIDDEN_STATES.copy()
        self.observations = observations if observations is not None else self.OBSERVATIONS.copy()
        
        self.N = len(self.states)          # Number of hidden states
        self.M = len(self.observations)    # Number of observation symbols
        self.smoothing_alpha = smoothing_alpha
        
        # Create mappings for efficient indexing
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        self.obs_to_idx = {o: i for i, o in enumerate(self.observations)}
        self.idx_to_obs = {i: o for i, o in enumerate(self.observations)}
        
        # Initialize parameters (will be set during training)
        self.A = None  # Transition matrix
        self.B = None  # Emission matrix
        self.pi = None # Initial distribution
        
        self._is_trained = False
    
    def _observation_to_indices(self, obs_sequence: List[str]) -> np.ndarray:
        """Convert observation sequence to indices."""
        return np.array([self.obs_to_idx[o] for o in obs_sequence])
    
    def _state_to_indices(self, state_sequence: List[str]) -> np.ndarray:
        """Convert state sequence to indices."""
        return np.array([self.state_to_idx[s] for s in state_sequence])
    
    # =========================================================================
    # FORWARD ALGORITHM - Evaluation Problem
    # =========================================================================
    
    def forward(self, observations: List[str], use_log: bool = True) -> Tuple[np.ndarray, float]:
        """
        Forward Algorithm: Compute P(O|λ) - the probability of observations given the model.
        
        This solves the Evaluation Problem: How likely is an observation sequence
        given the model parameters?
        
        Args:
            observations: Sequence of observation symbols (ECG leads)
            use_log: If True, use log probabilities to avoid underflow
            
        Returns:
            alpha: Forward variables α_t(i) matrix of shape (T, N)
            likelihood: P(O|λ) or log P(O|λ) if use_log=True
            
        Complexity: O(N² · T) time, O(N · T) space
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before computing forward probabilities")
        
        T = len(observations)
        obs_indices = self._observation_to_indices(observations)
        
        if use_log:
            return self._forward_log(obs_indices, T)
        else:
            return self._forward_standard(obs_indices, T)
    
    def _forward_standard(self, obs_indices: np.ndarray, T: int) -> Tuple[np.ndarray, float]:
        """Standard forward algorithm (may underflow for long sequences)."""
        alpha = np.zeros((T, self.N))
        
        # Initialization: α_1(i) = π_i · b_i(o_1)
        alpha[0] = self.pi * self.B[:, obs_indices[0]]
        
        # Induction: α_t(j) = [Σ_i α_{t-1}(i) · a_ij] · b_j(o_t)
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, obs_indices[t]]
        
        # Termination: P(O|λ) = Σ_i α_T(i)
        likelihood = np.sum(alpha[T-1])
        
        return alpha, likelihood
    
    def _forward_log(self, obs_indices: np.ndarray, T: int) -> Tuple[np.ndarray, float]:
        """Log-space forward algorithm (numerically stable)."""
        log_alpha = np.zeros((T, self.N))
        
        # Use small constant to avoid log(0)
        eps = 1e-300
        
        log_pi = np.log(self.pi + eps)
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)
        
        # Initialization
        log_alpha[0] = log_pi + log_B[:, obs_indices[0]]
        
        # Induction using log-sum-exp trick
        for t in range(1, T):
            for j in range(self.N):
                # log(Σ_i exp(log_alpha[t-1, i] + log_A[i, j]))
                log_alpha[t, j] = self._log_sum_exp(log_alpha[t-1] + log_A[:, j]) + log_B[j, obs_indices[t]]
        
        # Termination
        log_likelihood = self._log_sum_exp(log_alpha[T-1])
        
        return log_alpha, log_likelihood
    
    def _log_sum_exp(self, log_probs: np.ndarray) -> float:
        """Compute log(Σ exp(log_probs)) in a numerically stable way."""
        max_log = np.max(log_probs)
        if max_log == -np.inf:
            return -np.inf
        return max_log + np.log(np.sum(np.exp(log_probs - max_log)))
    
    # =========================================================================
    # VITERBI ALGORITHM - Decoding Problem
    # =========================================================================
    
    def viterbi(self, observations: List[str]) -> Tuple[List[str], float]:
        """
        Viterbi Algorithm: Find the most likely hidden state sequence.
        
        This solves the Decoding Problem: What is the most probable sequence
        of hidden states that generated the observations?
        
        Q* = argmax_Q P(Q|O, λ)
        
        Args:
            observations: Sequence of observation symbols (ECG leads)
            
        Returns:
            best_path: Most likely sequence of hidden states
            best_prob: Probability of the best path (log probability)
            
        Complexity: O(N² · T) time, O(N · T) space
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before running Viterbi")
        
        T = len(observations)
        obs_indices = self._observation_to_indices(observations)
        
        # Use log probabilities for numerical stability
        eps = 1e-300
        log_pi = np.log(self.pi + eps)
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)
        
        # δ_t(i): max probability of any path ending in state i at time t
        delta = np.zeros((T, self.N))
        # ψ_t(i): argmax - backpointer to previous state
        psi = np.zeros((T, self.N), dtype=int)
        
        # Initialization: δ_1(i) = π_i · b_i(o_1)
        delta[0] = log_pi + log_B[:, obs_indices[0]]
        psi[0] = 0
        
        # Recursion: δ_t(j) = max_i[δ_{t-1}(i) · a_ij] · b_j(o_t)
        for t in range(1, T):
            for j in range(self.N):
                candidates = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[j, obs_indices[t]]
        
        # Termination: find best final state
        best_last_state = np.argmax(delta[T-1])
        best_log_prob = delta[T-1, best_last_state]
        
        # Backtracking: reconstruct the best path
        best_path_indices = np.zeros(T, dtype=int)
        best_path_indices[T-1] = best_last_state
        
        for t in range(T-2, -1, -1):
            best_path_indices[t] = psi[t+1, best_path_indices[t+1]]
        
        # Convert indices back to state names
        best_path = [self.idx_to_state[i] for i in best_path_indices]
        
        return best_path, best_log_prob
    
    # =========================================================================
    # PARAMETER LEARNING - Training
    # =========================================================================
    
    def train_supervised(self, 
                        training_data: List[Tuple[List[str], List[str]]]) -> None:
        """
        Train HMM using Maximum Likelihood Estimation from labeled data.
        
        Given annotated data with both observations and hidden state labels,
        compute MLE estimates for A, B, and π.
        
        Args:
            training_data: List of (observations, states) tuples
                          Each tuple contains parallel sequences of the same length
        """
        # Initialize counts with Laplace smoothing
        alpha = self.smoothing_alpha
        
        transition_counts = np.full((self.N, self.N), alpha)
        emission_counts = np.full((self.N, self.M), alpha)
        initial_counts = np.full(self.N, alpha)
        
        # Count occurrences
        for observations, states in training_data:
            obs_indices = self._observation_to_indices(observations)
            state_indices = self._state_to_indices(states)
            
            # Count initial state
            initial_counts[state_indices[0]] += 1
            
            # Count transitions and emissions
            for t in range(len(states)):
                # Emission count
                emission_counts[state_indices[t], obs_indices[t]] += 1
                
                # Transition count (for t < T-1)
                if t < len(states) - 1:
                    transition_counts[state_indices[t], state_indices[t+1]] += 1
        
        # Normalize to get probabilities
        self.pi = initial_counts / np.sum(initial_counts)
        self.A = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
        self.B = emission_counts / np.sum(emission_counts, axis=1, keepdims=True)
        
        self._is_trained = True
    
    def train_baum_welch(self,
                         observations_list: List[List[str]],
                         max_iterations: int = 100,
                         convergence_threshold: float = 1e-6,
                         verbose: bool = True) -> List[float]:
        """
        Train HMM using Baum-Welch algorithm (EM for HMMs).
        
        This is used when we only have observations without state labels.
        Iteratively updates parameters to maximize P(O|λ).
        
        Args:
            observations_list: List of observation sequences
            max_iterations: Maximum number of EM iterations
            convergence_threshold: Stop when log-likelihood improvement < threshold
            verbose: Print progress information
            
        Returns:
            log_likelihoods: List of log-likelihoods at each iteration
        """
        # Initialize parameters randomly if not already set
        if not self._is_trained:
            self._initialize_random()
        
        log_likelihoods = []
        
        for iteration in range(max_iterations):
            # E-step: Compute expected counts using current parameters
            total_log_likelihood = 0
            
            # Accumulators for expected counts
            expected_initial = np.zeros(self.N)
            expected_transition = np.zeros((self.N, self.N))
            expected_emission = np.zeros((self.N, self.M))
            
            for observations in observations_list:
                T = len(observations)
                obs_indices = self._observation_to_indices(observations)
                
                # Compute forward and backward variables
                alpha, log_likelihood = self._forward_log(obs_indices, T)
                beta = self._backward_log(obs_indices, T)
                
                total_log_likelihood += log_likelihood
                
                # Compute γ_t(i) = P(q_t = s_i | O, λ)
                gamma = self._compute_gamma(alpha, beta, log_likelihood)
                
                # Compute ξ_t(i,j) = P(q_t = s_i, q_{t+1} = s_j | O, λ)
                xi = self._compute_xi(alpha, beta, obs_indices, log_likelihood)
                
                # Accumulate expected counts
                expected_initial += np.exp(gamma[0])
                expected_transition += np.sum(np.exp(xi), axis=0)
                
                for t in range(T):
                    expected_emission[:, obs_indices[t]] += np.exp(gamma[t])
            
            log_likelihoods.append(total_log_likelihood)
            
            if verbose:
                print(f"Iteration {iteration + 1}: log-likelihood = {total_log_likelihood:.4f}")
            
            # Check convergence
            if iteration > 0:
                improvement = log_likelihoods[-1] - log_likelihoods[-2]
                if improvement < convergence_threshold:
                    if verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    break
            
            # M-step: Update parameters
            # Add smoothing to avoid zeros
            expected_initial += self.smoothing_alpha
            expected_transition += self.smoothing_alpha
            expected_emission += self.smoothing_alpha
            
            self.pi = expected_initial / np.sum(expected_initial)
            self.A = expected_transition / np.sum(expected_transition, axis=1, keepdims=True)
            self.B = expected_emission / np.sum(expected_emission, axis=1, keepdims=True)
        
        return log_likelihoods
    
    def _backward_log(self, obs_indices: np.ndarray, T: int) -> np.ndarray:
        """Backward algorithm in log space."""
        log_beta = np.zeros((T, self.N))
        
        eps = 1e-300
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)
        
        # Initialization: β_T(i) = 1, so log(β_T(i)) = 0
        log_beta[T-1] = 0
        
        # Induction (backwards)
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                log_beta[t, i] = self._log_sum_exp(
                    log_A[i, :] + log_B[:, obs_indices[t+1]] + log_beta[t+1]
                )
        
        return log_beta
    
    def _compute_gamma(self, log_alpha: np.ndarray, log_beta: np.ndarray, 
                       log_likelihood: float) -> np.ndarray:
        """Compute γ_t(i) = P(q_t = s_i | O, λ) in log space."""
        return log_alpha + log_beta - log_likelihood
    
    def _compute_xi(self, log_alpha: np.ndarray, log_beta: np.ndarray,
                    obs_indices: np.ndarray, log_likelihood: float) -> np.ndarray:
        """Compute ξ_t(i,j) = P(q_t = s_i, q_{t+1} = s_j | O, λ) in log space."""
        T = len(obs_indices)
        eps = 1e-300
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)
        
        log_xi = np.zeros((T-1, self.N, self.N))
        
        for t in range(T-1):
            for i in range(self.N):
                for j in range(self.N):
                    log_xi[t, i, j] = (
                        log_alpha[t, i] + 
                        log_A[i, j] + 
                        log_B[j, obs_indices[t+1]] + 
                        log_beta[t+1, j] -
                        log_likelihood
                    )
        
        return log_xi
    
    def _initialize_random(self) -> None:
        """Initialize parameters with random values."""
        # Random initial distribution (normalized)
        self.pi = np.random.dirichlet(np.ones(self.N))
        
        # Random transition matrix (row-normalized)
        self.A = np.random.dirichlet(np.ones(self.N), size=self.N)
        
        # Random emission matrix (row-normalized)
        self.B = np.random.dirichlet(np.ones(self.M), size=self.N)
        
        self._is_trained = True
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def compute_likelihood(self, observations: List[str]) -> float:
        """Compute log P(O|λ) for an observation sequence."""
        _, log_likelihood = self.forward(observations, use_log=True)
        return log_likelihood
    
    def get_parameters(self) -> HMMParameters:
        """Return current HMM parameters."""
        return HMMParameters(
            states=self.states.copy(),
            observations=self.observations.copy(),
            A=self.A.copy() if self.A is not None else None,
            B=self.B.copy() if self.B is not None else None,
            pi=self.pi.copy() if self.pi is not None else None
        )
    
    def save(self, filepath: str) -> None:
        """Save model parameters to JSON file."""
        if not self._is_trained:
            raise ValueError("Cannot save untrained model")
        
        data = {
            'states': self.states,
            'observations': self.observations,
            'A': self.A.tolist(),
            'B': self.B.tolist(),
            'pi': self.pi.tolist(),
            'smoothing_alpha': self.smoothing_alpha
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'HiddenMarkovModel':
        """Load model from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        model = cls(
            states=data['states'],
            observations=data['observations'],
            smoothing_alpha=data.get('smoothing_alpha', 0.01)
        )
        
        model.A = np.array(data['A'])
        model.B = np.array(data['B'])
        model.pi = np.array(data['pi'])
        model._is_trained = True
        
        return model
    
    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"HiddenMarkovModel(N={self.N} states, M={self.M} observations, {status})"


# =========================================================================
# CLASSIFIER
# =========================================================================

class ScanpathClassifier:
    """
    Binary classifier for expert vs novice scanpath classification.
    
    Uses two HMMs: one trained on expert patterns, one on novice patterns.
    Classification is based on likelihood ratio.
    """
    
    def __init__(self):
        self.expert_hmm = HiddenMarkovModel()
        self.novice_hmm = HiddenMarkovModel()
        self._is_trained = False
    
    def train(self, 
              expert_data: List[Tuple[List[str], List[str]]],
              novice_data: List[Tuple[List[str], List[str]]]) -> None:
        """
        Train both expert and novice HMMs.
        
        Args:
            expert_data: List of (observations, states) tuples from experts
            novice_data: List of (observations, states) tuples from novices
        """
        print("Training expert HMM...")
        self.expert_hmm.train_supervised(expert_data)
        
        print("Training novice HMM...")
        self.novice_hmm.train_supervised(novice_data)
        
        self._is_trained = True
        print("Training complete!")
    
    def classify(self, observations: List[str]) -> Tuple[str, float, float]:
        """
        Classify a scanpath as expert or novice.
        
        Args:
            observations: Sequence of ECG lead fixations
            
        Returns:
            prediction: 'EXPERT' or 'NOVICE'
            expert_likelihood: Log-likelihood under expert model
            novice_likelihood: Log-likelihood under novice model
        """
        if not self._is_trained:
            raise ValueError("Classifier must be trained first")
        
        expert_ll = self.expert_hmm.compute_likelihood(observations)
        novice_ll = self.novice_hmm.compute_likelihood(observations)
        
        prediction = 'EXPERT' if expert_ll > novice_ll else 'NOVICE'
        
        return prediction, expert_ll, novice_ll
    
    def classify_batch(self, observations_list: List[List[str]]) -> List[str]:
        """Classify multiple scanpaths."""
        predictions = []
        for obs in observations_list:
            pred, _, _ = self.classify(obs)
            predictions.append(pred)
        return predictions
    
    def decode_cognitive_states(self, observations: List[str], 
                                 expertise: str = 'expert') -> List[str]:
        """
        Decode the most likely sequence of cognitive states.
        
        Args:
            observations: Sequence of ECG lead fixations
            expertise: Which model to use ('expert' or 'novice')
            
        Returns:
            List of cognitive diagnostic phases
        """
        hmm = self.expert_hmm if expertise == 'expert' else self.novice_hmm
        states, _ = hmm.viterbi(observations)
        return states
    
    def save(self, directory: str) -> None:
        """Save both models to a directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        self.expert_hmm.save(os.path.join(directory, 'expert_hmm.json'))
        self.novice_hmm.save(os.path.join(directory, 'novice_hmm.json'))
    
    @classmethod
    def load(cls, directory: str) -> 'ScanpathClassifier':
        """Load classifier from directory."""
        import os
        classifier = cls()
        classifier.expert_hmm = HiddenMarkovModel.load(
            os.path.join(directory, 'expert_hmm.json')
        )
        classifier.novice_hmm = HiddenMarkovModel.load(
            os.path.join(directory, 'novice_hmm.json')
        )
        classifier._is_trained = True
        return classifier


if __name__ == "__main__":
    # Quick test
    print("HMM Module loaded successfully!")
    print(f"Hidden States: {HiddenMarkovModel.HIDDEN_STATES}")
    print(f"Observations: {HiddenMarkovModel.OBSERVATIONS}")
