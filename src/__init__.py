"""
ECG Scanpath Pattern Recognition using Hidden Markov Models

A computational framework for analyzing medical expertise through
eye-tracking scanpath patterns during ECG interpretation.

Authors: Riad Benbrahim, Mohamed Amine El Bacha, Youssef Kaya
Course: Computational Theory - Fall 2025
Institution: Mohammed VI Polytechnic University
"""

from .hmm import HiddenMarkovModel, ScanpathClassifier, HMMParameters
from .data_generator import (
    ScanpathDatasetGenerator,
    ExpertScanpathGenerator,
    NoviceScanpathGenerator,
    ScanpathSample,
    generate_training_data,
    DIAGNOSTIC_PHASES,
    ECG_LEADS
)
from .evaluation import (
    compute_classification_metrics,
    compute_confusion_matrix,
    ClassificationMetrics,
    ParameterAnalyzer,
    ComplexityAnalyzer
)

__version__ = '1.0.0'
__author__ = 'Riad Benbrahim, Mohamed Amine El Bacha, Youssef Kaya'

__all__ = [
    # HMM
    'HiddenMarkovModel',
    'ScanpathClassifier',
    'HMMParameters',
    
    # Data Generation
    'ScanpathDatasetGenerator',
    'ExpertScanpathGenerator', 
    'NoviceScanpathGenerator',
    'ScanpathSample',
    'generate_training_data',
    'DIAGNOSTIC_PHASES',
    'ECG_LEADS',
    
    # Evaluation
    'compute_classification_metrics',
    'compute_confusion_matrix',
    'ClassificationMetrics',
    'ParameterAnalyzer',
    'ComplexityAnalyzer'
]
