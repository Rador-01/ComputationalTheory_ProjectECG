"""
Synthetic Scanpath Data Generator for ECG Interpretation

This module generates clinically-grounded synthetic scanpath data
based on AHA/ACCF guidelines for systematic ECG interpretation.

Expert patterns follow the recommended diagnostic workflow:
1. Rate/Rhythm assessment (Lead II, V1)
2. Axis determination (Lead I, aVF)
3. P-wave analysis
4. PR interval assessment
5. QRS complex analysis
6. ST segment evaluation
7. T-wave examination
8. QT interval measurement
9. Lead-by-lead systematic review

Novice patterns exhibit:
- Erratic, non-systematic viewing
- Incomplete coverage of ECG regions
- High backtracking and redundant fixations
- Random-like transitions

Authors: Riad Benbrahim, Mohamed Amine El Bacha, Youssef Kaya
Course: Computational Theory - Fall 2025
Institution: Mohammed VI Polytechnic University

Reference: Surawicz, B. et al. (2009). AHA/ACCF/HRS recommendations for 
the standardization and interpretation of the electrocardiogram.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from dataclasses import dataclass, asdict
import csv
import os


# Define the cognitive diagnostic phases (hidden states)
DIAGNOSTIC_PHASES = [
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

# Define ECG leads (observations)
ECG_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Lead groupings based on cardiac regions
LEAD_GROUPS = {
    'limb': ['I', 'II', 'III'],
    'augmented': ['aVR', 'aVL', 'aVF'],
    'septal': ['V1', 'V2'],
    'anterior': ['V3', 'V4'],
    'lateral': ['V5', 'V6'],
    'inferior': ['II', 'III', 'aVF'],
    'lateral_extended': ['I', 'aVL', 'V5', 'V6']
}


@dataclass
class ScanpathSample:
    """Container for a single scanpath sample."""
    participant_id: str
    expertise_level: str  # 'expert' or 'novice'
    trial_id: str
    observations: List[str]  # Sequence of ECG leads
    hidden_states: List[str]  # Sequence of diagnostic phases
    fixation_durations: List[int]  # Duration in ms
    
    def to_dict(self) -> dict:
        return asdict(self)


class ExpertScanpathGenerator:
    """
    Generates expert-like scanpaths following clinical guidelines.
    
    Expert characteristics:
    - Systematic workflow following AHA/ACCF guidelines
    - Efficient coverage of all ECG regions
    - Focused fixations on diagnostically relevant leads
    - Smooth transitions between related diagnostic phases
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        # Expert transition probabilities between diagnostic phases
        # Based on clinical workflow: Rhythm → Axis → P-wave → PR → QRS → ST → T → QT → Review
        self.transition_probs = self._build_expert_transitions()
        
        # Emission probabilities: which leads are examined in each phase
        self.emission_probs = self._build_expert_emissions()
        
        # Initial state distribution (experts usually start with rhythm check)
        self.initial_probs = np.array([
            0.70,  # Rhythm-Check (most common start)
            0.15,  # Axis-Determination
            0.05,  # P-wave-Analysis
            0.02,  # PR-interval-Assessment
            0.03,  # QRS-Analysis
            0.02,  # ST-segment-Evaluation
            0.01,  # T-wave-Examination
            0.01,  # QT-interval-Measurement
            0.01   # Lead-by-Lead-Review
        ])
    
    def _build_expert_transitions(self) -> np.ndarray:
        """
        Build expert transition matrix following clinical workflow.
        
        The systematic approach follows:
        Rhythm → Axis → P-wave → PR → QRS → ST → T → QT → Review
        
        With some flexibility for returning to earlier phases.
        """
        n_states = len(DIAGNOSTIC_PHASES)
        A = np.zeros((n_states, n_states))
        
        # Primary workflow transitions (forward progress)
        # Rhythm-Check → Axis-Determination (high probability)
        A[0, 1] = 0.50  # Move to axis
        A[0, 0] = 0.30  # Stay in rhythm check
        A[0, 2] = 0.15  # Skip to P-wave
        A[0, 4] = 0.05  # Jump to QRS (quick overview)
        
        # Axis-Determination
        A[1, 2] = 0.55  # Move to P-wave
        A[1, 1] = 0.20  # Stay in axis
        A[1, 0] = 0.10  # Return to rhythm
        A[1, 4] = 0.15  # Jump to QRS
        
        # P-wave-Analysis
        A[2, 3] = 0.60  # Move to PR interval
        A[2, 2] = 0.20  # Stay in P-wave
        A[2, 0] = 0.10  # Return to rhythm
        A[2, 4] = 0.10  # Jump to QRS
        
        # PR-interval-Assessment
        A[3, 4] = 0.65  # Move to QRS
        A[3, 3] = 0.20  # Stay in PR
        A[3, 2] = 0.10  # Return to P-wave
        A[3, 0] = 0.05  # Return to rhythm
        
        # QRS-Analysis
        A[4, 5] = 0.55  # Move to ST segment
        A[4, 4] = 0.25  # Stay in QRS
        A[4, 6] = 0.10  # Skip to T-wave
        A[4, 8] = 0.10  # Jump to review
        
        # ST-segment-Evaluation
        A[5, 6] = 0.60  # Move to T-wave
        A[5, 5] = 0.25  # Stay in ST
        A[5, 4] = 0.10  # Return to QRS
        A[5, 8] = 0.05  # Jump to review
        
        # T-wave-Examination
        A[6, 7] = 0.55  # Move to QT interval
        A[6, 6] = 0.25  # Stay in T-wave
        A[6, 5] = 0.10  # Return to ST
        A[6, 8] = 0.10  # Jump to review
        
        # QT-interval-Measurement
        A[7, 8] = 0.60  # Move to review
        A[7, 7] = 0.25  # Stay in QT
        A[7, 6] = 0.10  # Return to T-wave
        A[7, 0] = 0.05  # Return to start (complete cycle)
        
        # Lead-by-Lead-Review
        A[8, 8] = 0.70  # Stay in review (systematic coverage)
        A[8, 4] = 0.15  # Return to QRS for clarification
        A[8, 5] = 0.10  # Return to ST
        A[8, 0] = 0.05  # Return to rhythm (final check)
        
        # Normalize rows
        A = A / A.sum(axis=1, keepdims=True)
        
        return A
    
    def _build_expert_emissions(self) -> np.ndarray:
        """
        Build expert emission matrix.
        
        Maps diagnostic phases to likely ECG lead fixations based on
        clinical relevance of each lead for each diagnostic task.
        """
        n_states = len(DIAGNOSTIC_PHASES)
        n_obs = len(ECG_LEADS)
        B = np.zeros((n_states, n_obs))
        
        # Index mapping
        lead_idx = {lead: i for i, lead in enumerate(ECG_LEADS)}
        
        # Rhythm-Check: Focus on Lead II (rhythm strip) and V1
        B[0, lead_idx['II']] = 0.40
        B[0, lead_idx['V1']] = 0.25
        B[0, lead_idx['I']] = 0.15
        B[0, lead_idx['aVF']] = 0.10
        B[0, lead_idx['V5']] = 0.10
        
        # Axis-Determination: Lead I and aVF (quadrant method)
        B[1, lead_idx['I']] = 0.35
        B[1, lead_idx['aVF']] = 0.35
        B[1, lead_idx['II']] = 0.15
        B[1, lead_idx['aVL']] = 0.10
        B[1, lead_idx['III']] = 0.05
        
        # P-wave-Analysis: Multiple leads, focus on II and V1
        B[2, lead_idx['II']] = 0.30
        B[2, lead_idx['V1']] = 0.25
        B[2, lead_idx['I']] = 0.15
        B[2, lead_idx['aVL']] = 0.10
        B[2, lead_idx['aVF']] = 0.10
        B[2, lead_idx['III']] = 0.10
        
        # PR-interval-Assessment: Lead II primarily
        B[3, lead_idx['II']] = 0.45
        B[3, lead_idx['V1']] = 0.25
        B[3, lead_idx['I']] = 0.15
        B[3, lead_idx['V5']] = 0.15
        
        # QRS-Analysis: Precordial leads for morphology
        B[4, lead_idx['V1']] = 0.18
        B[4, lead_idx['V2']] = 0.17
        B[4, lead_idx['V3']] = 0.15
        B[4, lead_idx['V4']] = 0.15
        B[4, lead_idx['V5']] = 0.15
        B[4, lead_idx['V6']] = 0.10
        B[4, lead_idx['I']] = 0.05
        B[4, lead_idx['II']] = 0.05
        
        # ST-segment-Evaluation: Precordial and inferior leads
        B[5, lead_idx['V2']] = 0.18
        B[5, lead_idx['V3']] = 0.17
        B[5, lead_idx['V4']] = 0.15
        B[5, lead_idx['II']] = 0.15
        B[5, lead_idx['III']] = 0.12
        B[5, lead_idx['aVF']] = 0.12
        B[5, lead_idx['V5']] = 0.06
        B[5, lead_idx['V6']] = 0.05
        
        # T-wave-Examination: Multiple leads
        B[6, lead_idx['V2']] = 0.15
        B[6, lead_idx['V3']] = 0.15
        B[6, lead_idx['V4']] = 0.15
        B[6, lead_idx['V5']] = 0.15
        B[6, lead_idx['II']] = 0.15
        B[6, lead_idx['I']] = 0.10
        B[6, lead_idx['aVL']] = 0.08
        B[6, lead_idx['aVF']] = 0.07
        
        # QT-interval-Measurement: Lead II or V5 (where T is most visible)
        B[7, lead_idx['II']] = 0.40
        B[7, lead_idx['V5']] = 0.35
        B[7, lead_idx['V4']] = 0.15
        B[7, lead_idx['I']] = 0.10
        
        # Lead-by-Lead-Review: More uniform across all leads
        uniform = 1.0 / n_obs
        B[8, :] = uniform * 0.7  # Base uniform
        # Slight bias toward precordial leads
        B[8, lead_idx['V1']] += 0.05
        B[8, lead_idx['V2']] += 0.05
        B[8, lead_idx['V3']] += 0.05
        B[8, lead_idx['V4']] += 0.05
        B[8, lead_idx['V5']] += 0.05
        B[8, lead_idx['V6']] += 0.05
        
        # Normalize rows
        B = B / B.sum(axis=1, keepdims=True)
        
        return B
    
    def generate_scanpath(self, 
                         min_length: int = 15,
                         max_length: int = 30) -> Tuple[List[str], List[str], List[int]]:
        """
        Generate a single expert scanpath.
        
        Returns:
            observations: Sequence of ECG leads
            hidden_states: Sequence of diagnostic phases
            durations: Fixation durations in ms
        """
        length = np.random.randint(min_length, max_length + 1)
        
        observations = []
        hidden_states = []
        durations = []
        
        # Sample initial state
        current_state = np.random.choice(len(DIAGNOSTIC_PHASES), p=self.initial_probs)
        
        for _ in range(length):
            # Record current state
            hidden_states.append(DIAGNOSTIC_PHASES[current_state])
            
            # Emit observation based on current state
            obs_idx = np.random.choice(len(ECG_LEADS), p=self.emission_probs[current_state])
            observations.append(ECG_LEADS[obs_idx])
            
            # Generate fixation duration (experts: 150-400ms, focused)
            duration = int(np.random.normal(250, 50))
            duration = max(100, min(500, duration))
            durations.append(duration)
            
            # Transition to next state
            current_state = np.random.choice(len(DIAGNOSTIC_PHASES), 
                                            p=self.transition_probs[current_state])
        
        return observations, hidden_states, durations


class NoviceScanpathGenerator:
    """
    Generates novice-like scanpaths with erratic, non-systematic patterns.
    
    Novice characteristics:
    - Non-systematic viewing (random-like transitions)
    - Incomplete coverage
    - High backtracking
    - More uniform/random fixation distribution
    - Shorter overall scanpaths (premature termination)
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.transition_probs = self._build_novice_transitions()
        self.emission_probs = self._build_novice_emissions()
        
        # Novice initial distribution (more random)
        self.initial_probs = np.array([
            0.25,  # Rhythm-Check
            0.15,  # Axis-Determination
            0.10,  # P-wave-Analysis
            0.05,  # PR-interval-Assessment
            0.20,  # QRS-Analysis (often jump straight to QRS)
            0.10,  # ST-segment-Evaluation
            0.05,  # T-wave-Examination
            0.05,  # QT-interval-Measurement
            0.05   # Lead-by-Lead-Review
        ])
    
    def _build_novice_transitions(self) -> np.ndarray:
        """
        Build novice transition matrix with erratic patterns.
        
        Characteristics:
        - High self-loops (getting stuck)
        - Random jumps between phases
        - Frequent backtracking
        """
        n_states = len(DIAGNOSTIC_PHASES)
        
        # Start with more uniform transitions
        A = np.ones((n_states, n_states)) * 0.08
        
        # Higher self-loops (getting stuck on one thing)
        np.fill_diagonal(A, 0.35)
        
        # Add some structure but with noise
        # Novices might follow some logical order but inconsistently
        for i in range(n_states - 1):
            A[i, i+1] += 0.15  # Some forward movement
            if i > 0:
                A[i, i-1] += 0.10  # Backtracking
        
        # Random jumps
        for i in range(n_states):
            jump_target = np.random.randint(n_states)
            A[i, jump_target] += 0.08
        
        # Normalize rows
        A = A / A.sum(axis=1, keepdims=True)
        
        return A
    
    def _build_novice_emissions(self) -> np.ndarray:
        """
        Build novice emission matrix with scattered fixations.
        
        Novices don't know which leads are relevant for each diagnostic task,
        so emissions are more uniform with some random biases.
        """
        n_states = len(DIAGNOSTIC_PHASES)
        n_obs = len(ECG_LEADS)
        
        # More uniform distribution
        B = np.ones((n_states, n_obs)) / n_obs
        
        # Add some random noise/biases
        noise = np.random.dirichlet(np.ones(n_obs) * 2, size=n_states) * 0.3
        B = B * 0.7 + noise
        
        # Normalize rows
        B = B / B.sum(axis=1, keepdims=True)
        
        return B
    
    def generate_scanpath(self,
                         min_length: int = 12,
                         max_length: int = 25) -> Tuple[List[str], List[str], List[int]]:
        """
        Generate a single novice scanpath.
        
        Novice scanpaths tend to be shorter and less complete.
        """
        length = np.random.randint(min_length, max_length + 1)
        
        observations = []
        hidden_states = []
        durations = []
        
        # Sample initial state
        current_state = np.random.choice(len(DIAGNOSTIC_PHASES), p=self.initial_probs)
        
        for _ in range(length):
            # Record current state
            hidden_states.append(DIAGNOSTIC_PHASES[current_state])
            
            # Emit observation
            obs_idx = np.random.choice(len(ECG_LEADS), p=self.emission_probs[current_state])
            observations.append(ECG_LEADS[obs_idx])
            
            # Generate fixation duration (novices: more variable, 100-600ms)
            duration = int(np.random.normal(280, 100))
            duration = max(80, min(700, duration))
            durations.append(duration)
            
            # Transition to next state
            current_state = np.random.choice(len(DIAGNOSTIC_PHASES),
                                            p=self.transition_probs[current_state])
        
        return observations, hidden_states, durations


class ScanpathDatasetGenerator:
    """
    Generate complete datasets of expert and novice scanpaths.
    """
    
    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self.expert_gen = ExpertScanpathGenerator(seed=seed)
        self.novice_gen = NoviceScanpathGenerator(seed=seed + 1000 if seed else None)
    
    def generate_dataset(self,
                        n_expert: int = 100,
                        n_novice: int = 100,
                        expert_length_range: Tuple[int, int] = (18, 28),
                        novice_length_range: Tuple[int, int] = (12, 22)) -> List[ScanpathSample]:
        """
        Generate a complete dataset with both expert and novice samples.
        
        Args:
            n_expert: Number of expert samples
            n_novice: Number of novice samples
            expert_length_range: (min, max) length for expert scanpaths
            novice_length_range: (min, max) length for novice scanpaths
            
        Returns:
            List of ScanpathSample objects
        """
        samples = []
        
        # Generate expert samples
        for i in range(n_expert):
            obs, states, durations = self.expert_gen.generate_scanpath(
                min_length=expert_length_range[0],
                max_length=expert_length_range[1]
            )
            
            sample = ScanpathSample(
                participant_id=f"E{i+1:03d}",
                expertise_level='expert',
                trial_id=f"T{i+1:03d}",
                observations=obs,
                hidden_states=states,
                fixation_durations=durations
            )
            samples.append(sample)
        
        # Generate novice samples
        for i in range(n_novice):
            obs, states, durations = self.novice_gen.generate_scanpath(
                min_length=novice_length_range[0],
                max_length=novice_length_range[1]
            )
            
            sample = ScanpathSample(
                participant_id=f"N{i+1:03d}",
                expertise_level='novice',
                trial_id=f"T{i+1:03d}",
                observations=obs,
                hidden_states=states,
                fixation_durations=durations
            )
            samples.append(sample)
        
        return samples
    
    def save_dataset(self, 
                    samples: List[ScanpathSample],
                    output_dir: str,
                    format: str = 'both') -> None:
        """
        Save dataset to files.
        
        Args:
            samples: List of ScanpathSample objects
            output_dir: Directory to save files
            format: 'json', 'csv', or 'both'
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if format in ['json', 'both']:
            # Save as JSON
            json_path = os.path.join(output_dir, 'scanpath_dataset.json')
            with open(json_path, 'w') as f:
                json.dump([s.to_dict() for s in samples], f, indent=2)
            print(f"Saved JSON dataset to {json_path}")
        
        if format in ['csv', 'both']:
            # Save as CSV (flattened format for each fixation)
            csv_path = os.path.join(output_dir, 'scanpath_dataset.csv')
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'participant_id', 'expertise_level', 'trial_id',
                    'fixation_id', 'ecg_lead', 'diagnostic_phase', 'duration_ms'
                ])
                
                for sample in samples:
                    for i, (obs, state, dur) in enumerate(zip(
                        sample.observations, 
                        sample.hidden_states,
                        sample.fixation_durations
                    )):
                        writer.writerow([
                            sample.participant_id,
                            sample.expertise_level,
                            sample.trial_id,
                            i + 1,
                            obs,
                            state,
                            dur
                        ])
            
            print(f"Saved CSV dataset to {csv_path}")
    
    def split_dataset(self,
                     samples: List[ScanpathSample],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     shuffle: bool = True) -> Tuple[List[ScanpathSample], 
                                                    List[ScanpathSample],
                                                    List[ScanpathSample]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            samples: Full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation (test = 1 - train - val)
            shuffle: Whether to shuffle before splitting
            
        Returns:
            (train_samples, val_samples, test_samples)
        """
        if shuffle:
            np.random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return (
            samples[:train_end],
            samples[train_end:val_end],
            samples[val_end:]
        )
    
    def get_statistics(self, samples: List[ScanpathSample]) -> Dict:
        """Compute dataset statistics."""
        expert_samples = [s for s in samples if s.expertise_level == 'expert']
        novice_samples = [s for s in samples if s.expertise_level == 'novice']
        
        def compute_stats(sample_list):
            lengths = [len(s.observations) for s in sample_list]
            durations = [d for s in sample_list for d in s.fixation_durations]
            return {
                'count': len(sample_list),
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'min_length': np.min(lengths),
                'max_length': np.max(lengths),
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations)
            }
        
        return {
            'total_samples': len(samples),
            'expert': compute_stats(expert_samples),
            'novice': compute_stats(novice_samples)
        }


def generate_training_data(samples: List[ScanpathSample], 
                          expertise: str = 'expert') -> List[Tuple[List[str], List[str]]]:
    """
    Convert ScanpathSamples to training format for HMM.
    
    Args:
        samples: List of ScanpathSample objects
        expertise: Filter by 'expert', 'novice', or 'all'
        
    Returns:
        List of (observations, hidden_states) tuples
    """
    if expertise == 'all':
        filtered = samples
    else:
        filtered = [s for s in samples if s.expertise_level == expertise]
    
    return [(s.observations, s.hidden_states) for s in filtered]


if __name__ == "__main__":
    # Demo: Generate and save a dataset
    print("=" * 60)
    print("ECG Scanpath Dataset Generator")
    print("=" * 60)
    
    generator = ScanpathDatasetGenerator(seed=42)
    
    # Generate dataset
    print("\nGenerating dataset...")
    samples = generator.generate_dataset(
        n_expert=100,
        n_novice=100
    )
    
    # Print statistics
    stats = generator.get_statistics(samples)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"\n  Expert samples:")
    print(f"    Count: {stats['expert']['count']}")
    print(f"    Mean length: {stats['expert']['mean_length']:.1f} ± {stats['expert']['std_length']:.1f}")
    print(f"    Mean fixation duration: {stats['expert']['mean_duration']:.1f} ms")
    print(f"\n  Novice samples:")
    print(f"    Count: {stats['novice']['count']}")
    print(f"    Mean length: {stats['novice']['mean_length']:.1f} ± {stats['novice']['std_length']:.1f}")
    print(f"    Mean fixation duration: {stats['novice']['mean_duration']:.1f} ms")
    
    # Show example
    print("\n" + "=" * 60)
    print("Example Expert Scanpath:")
    print("=" * 60)
    expert_sample = [s for s in samples if s.expertise_level == 'expert'][0]
    print(f"  Participant: {expert_sample.participant_id}")
    print(f"  Length: {len(expert_sample.observations)}")
    print(f"  Observations (first 10): {expert_sample.observations[:10]}")
    print(f"  Hidden states (first 10): {expert_sample.hidden_states[:10]}")
    
    print("\n" + "=" * 60)
    print("Example Novice Scanpath:")
    print("=" * 60)
    novice_sample = [s for s in samples if s.expertise_level == 'novice'][0]
    print(f"  Participant: {novice_sample.participant_id}")
    print(f"  Length: {len(novice_sample.observations)}")
    print(f"  Observations (first 10): {novice_sample.observations[:10]}")
    print(f"  Hidden states (first 10): {novice_sample.hidden_states[:10]}")
