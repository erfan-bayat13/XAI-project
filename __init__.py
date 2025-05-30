# xai_adversarial_detection/__init__.py
"""
XAI Adversarial Detection for Vision Transformers
A comprehensive framework for detecting adversarial examples using explainable AI techniques.
"""

from .config import Config

# datasets/__init__.py
"""
Dataset utilities for XAI adversarial detection
"""
from .datasets.mnist_evenodd import MNISTEvenOdd, create_data_loaders

# models/__init__.py
"""
Model implementations for XAI adversarial detection
"""
from models.vit_wrapper import MNISTViTWrapper, train_even_odd_classifier, evaluate_model
from models.mlp_concept import AttentionToConceptMLP, train_attention_to_concept_mlp, detect_adversarial_with_mlp

# analyzers/__init__.py
"""
Analysis tools for attention and concept detection
"""
from analyzers.attention_analyzer import MNISTViTAttentionAnalyzer
from analyzers.concept_detector import ConceptBasedDetector

# attacks/__init__.py
"""
Adversarial attack implementations
"""
from attacks.fgsm_generator import FGSMAttackGenerator

# utils/__init__.py
"""
Utility functions for visualization and metrics
"""
from utils.visualization import (visualize_adversarial_comparison, plot_detection_scores_distribution,
                          plot_detection_performance_metrics, plot_epsilon_vs_performance)

# experiments/__init__.py
"""
Experiment runners and evaluation tools
"""
from experiments.runner import ExperimentRunner