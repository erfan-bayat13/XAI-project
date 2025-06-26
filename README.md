# XAI Adversarial Detection for Vision Transformers

A concept-based adversarial detection framework that leverages Vision Transformer attention patterns to identify adversarial examples through semantic consistency checking.

## Overview

This project implements a novel approach to adversarial detection by monitoring the consistency between ViT attention patterns and expected concept activations. Instead of relying on statistical anomalies, our method detects semantic inconsistencies that adversarial attacks create in Vision Transformer middle layers.

## Key Features

- **Concept-based Detection**: Uses attention-to-concept mapping via MLP for semantic consistency checking
- **ViT Integration**: Leverages hierarchical concept development in Vision Transformer layers 6-8
- **ART Compatibility**: Integrated with Adversarial Robustness Toolbox for robust attack generation
- **Interpretable Defense**: Provides explainable detection decisions through concept analysis

## Architecture

```
Input Image → ViT → Attention Extraction → MLP Concept Predictor → Detection Decision
                    (Middle Layers 6-8)     (603D features)      (Confidence-based)
```

## Quick Start

### Installation

```bash
pip install torch torchvision transformers adversarial-robustness-toolbox
pip install matplotlib numpy scikit-learn
```

### Basic Usage

```bash
# Run quick demo
python main.py --mode demo --digit 5

# Full pipeline
python main.py --mode full

# Training only
python main.py --mode train --epochs 3

# Paper experiments
python test.py
```

## Experiments

The `test.py` file contains the complete experimental setup from our paper, including:
- MLP attention-to-concept mapping training
- FGSM/PGD adversarial attack generation
- Comprehensive detection evaluation
- Visualization of results

## Dataset

Uses MNIST adapted for even-odd classification:
- **Concept**: Digit identity (0-9)
- **Task**: Even/Odd classification
- Perfect for testing concept-attention consistency

## Results

- **MLP Concept Accuracy**: ~85%
- **False Positive Rate**: <5%
- **Attack Detection Rate**: >90% for ε ≥ 0.15
- **Concept Change Rate**: 47.1% under adversarial attack

## Citation

```bibtex
@article{bayat2024concept,
  title={Concept-based Adversarial Detection for Vision Transformers},
  author={Bayat, Erfan and Contarin, Diletta},
  year={2024},
  institution={Politecnico di Torino}
}
```

## Project Structure

```
├── main.py              # Main entry point
├── test.py              # Paper experimental setup
├── config.py            # Configuration settings
├── datasets/
│   └── mnist_evenodd.py # MNIST Even-Odd dataset
├── models/
│   ├── vit_wrapper.py   # ViT model wrapper
│   └── mlp_concept.py   # MLP concept mapper
├── analyzers/
│   ├── attention_analyzer.py # Attention analysis
│   └── concept_detector.py   # Concept-based detection
├── attacks/
│   └── fgsm_generator.py     # FGSM/PGD attacks with ART
└── utils/
    └── visualization.py      # Plotting utilities
```

## Acknowledgements
This project is part of the Explainable and Trustworthy AI course at Politecnico di Torino.