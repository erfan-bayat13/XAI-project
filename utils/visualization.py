"""
Visualization utilities for XAI Adversarial Detection
"""
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Tuple
from config import Config


def visualize_adversarial_comparison(vit_analyzer,
                                   fgsm_attacker,
                                   dataset,
                                   digit: int = 5, 
                                   epsilon: float = Config.DEFAULT_EPSILON) -> None:
    """
    Visualize side-by-side comparison of clean vs adversarial attention patterns
    
    Args:
        vit_analyzer: ViT attention analyzer
        fgsm_attacker: FGSM attack generator
        dataset: Dataset to sample from
        digit: Digit to analyze
        epsilon: Attack strength
    """
    print(f"🔍 Comparing Clean vs Adversarial Attention for Digit {digit}")

    # Find a sample with the specified digit
    image, concept, task = None, None, None
    for i, (img, conc, tsk) in enumerate(dataset):
        if conc == digit:
            image, concept, task = img, conc, tsk
            break
    else:
        print(f"❌ No sample found for digit {digit}")
        return

    # Generate adversarial example
    adv_image = fgsm_attacker.generate_fgsm_attack(image, task, epsilon=epsilon)

    # Test attack success
    attack_result = fgsm_attacker.test_attack_success(image, adv_image, task)

    print(f"📊 Attack Results:")
    print(f"   Original: Digit {concept} → {'Even' if task == 0 else 'Odd'}")
    print(f"   Predicted Original: {'Even' if attack_result['original_prediction'] == 0 else 'Odd'}")
    print(f"   Predicted Adversarial: {'Even' if attack_result['adversarial_prediction'] == 0 else 'Odd'}")
    print(f"   Attack Success: {'✅' if attack_result['attack_success'] else '❌'}")

    # Get attention patterns for both
    clean_attention, _ = vit_analyzer.extract_attention_maps(image)
    adv_attention, _ = vit_analyzer.extract_attention_maps(adv_image.squeeze(0))

    # Visualize comparison
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))

    # Images
    clean_img = image[0].to(Config.DEVICE) * Config.MNIST_STD + Config.MNIST_MEAN
    adv_img = adv_image.squeeze(0)[0] * Config.MNIST_STD + Config.MNIST_MEAN
    perturbation = torch.abs(adv_img - clean_img)

    axes[0, 0].imshow(clean_img.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Clean Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(adv_img.cpu().numpy(), cmap='gray')
    axes[0, 1].set_title('Adversarial Image')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(perturbation.cpu().numpy(), cmap='hot')
    axes[0, 2].set_title(f'Perturbation (ε={epsilon})')
    axes[0, 2].axis('off')

    # Hide unused image plots
    for j in range(3, 6):
        axes[0, j].axis('off')

    # Attention patterns for key layers
    key_layers = Config.KEY_LAYERS

    for i, layer_idx in enumerate(key_layers):
        if i < 5:  # Only plot first 5 layers
            # Clean attention
            clean_att = clean_attention[layer_idx].reshape(14, 14)
            im1 = axes[1, i].imshow(clean_att.cpu().numpy(), cmap=Config.ATTENTION_COLORMAP)
            axes[1, i].set_title(f'Clean L{layer_idx}')
            axes[1, i].axis('off')

            # Adversarial attention
            adv_att = adv_attention[layer_idx].reshape(14, 14)
            im2 = axes[2, i].imshow(adv_att.cpu().numpy(), cmap=Config.ATTENTION_COLORMAP)
            axes[2, i].set_title(f'Adversarial L{layer_idx}')
            axes[2, i].axis('off')

    # Hide last column in attention rows
    axes[1, 5].axis('off')
    axes[2, 5].axis('off')

    plt.suptitle(f'Clean vs Adversarial Comparison - Digit {digit}\n'
                f'True: {concept} ({"Even" if task == 0 else "Odd"}) | '
                f'Adv Pred: {"Even" if attack_result["adversarial_prediction"] == 0 else "Odd"}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_detection_scores_distribution(clean_scores: List[float], 
                                     adv_scores: List[float],
                                     threshold: float = Config.DETECTION_THRESHOLD) -> None:
    """
    Plot distribution of detection scores for clean vs adversarial images
    
    Args:
        clean_scores: Detection scores for clean images
        adv_scores: Detection scores for adversarial images
        threshold: Detection threshold
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(clean_scores, alpha=0.7, label='Clean Images', bins=20, color='green', density=True)
    plt.hist(adv_scores, alpha=0.7, label='Adversarial Images', bins=20, color='red', density=True)
    
    # Add threshold line
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
    
    plt.xlabel('Detection Score')
    plt.ylabel('Density')
    plt.title('Distribution of Detection Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    clean_mean, clean_std = np.mean(clean_scores), np.std(clean_scores)
    adv_mean, adv_std = np.mean(adv_scores), np.std(adv_scores)
    
    plt.text(0.02, 0.98, f'Clean: μ={clean_mean:.3f}, σ={clean_std:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.text(0.02, 0.88, f'Adversarial: μ={adv_mean:.3f}, σ={adv_std:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.show()


def plot_attention_evolution(attention_maps: List[torch.Tensor], 
                           image: torch.Tensor,
                           title: str = "Attention Evolution") -> None:
    """
    Plot attention evolution across ViT layers
    
    Args:
        attention_maps: List of attention maps from different layers
        image: Original image tensor
        title: Plot title
    """
    # Convert image for display
    img_display = image.clone()
    img_display = img_display * Config.MNIST_STD + Config.MNIST_MEAN
    img_display = torch.clamp(img_display, 0, 1)
    img_display = img_display[0]  # Take first channel

    # Plot attention evolution
    fig, axes = plt.subplots(2, len(Config.KEY_LAYERS), figsize=(20, 8))

    for i, layer_idx in enumerate(Config.KEY_LAYERS):
        if layer_idx < len(attention_maps):
            attention_map = attention_maps[layer_idx]
            spatial_attention = attention_map.reshape(14, 14)

            # Raw attention
            im1 = axes[0, i].imshow(spatial_attention.cpu().numpy(), 
                                  cmap=Config.ATTENTION_COLORMAP, interpolation='bicubic')
            axes[0, i].set_title(f"Layer {layer_idx}")
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

            # Overlay on original image
            attention_resized = torch.nn.functional.interpolate(
                spatial_attention.unsqueeze(0).unsqueeze(0),
                size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                mode='bicubic'
            )[0, 0]

            axes[1, i].imshow(img_display.numpy(), cmap='gray')
            axes[1, i].imshow(attention_resized.cpu().numpy(), 
                            cmap=Config.ATTENTION_COLORMAP, alpha=0.6)
            axes[1, i].set_title(f"Layer {layer_idx} Overlay")
            axes[1, i].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_detection_performance_metrics(metrics: Dict[str, float]) -> None:
    """
    Plot detection performance metrics as a bar chart
    
    Args:
        metrics: Dictionary with performance metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Main metrics
    main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    main_values = [metrics.get(m, 0) for m in main_metrics]
    main_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    bars1 = ax1.bar(main_labels, main_values, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.set_title('Detection Performance Metrics')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, main_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Error rates
    error_metrics = ['false_positive_rate', 'true_positive_rate']
    error_values = [metrics.get(m, 0) for m in error_metrics]
    error_labels = ['False Positive Rate', 'True Positive Rate']
    
    bars2 = ax2.bar(error_labels, error_values, color=['red', 'green'])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Rate')
    ax2.set_title('Detection Rates')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, error_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_epsilon_vs_performance(epsilon_results: Dict[float, Dict[str, float]]) -> None:
    """
    Plot attack success rate and detection rate vs epsilon
    
    Args:
        epsilon_results: Dictionary mapping epsilon to performance metrics
    """
    epsilons = sorted(epsilon_results.keys())
    attack_rates = [epsilon_results[eps]['attack_success_rate'] for eps in epsilons]
    detection_rates = [epsilon_results[eps]['detection_rate'] for eps in epsilons]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(epsilons, attack_rates, 'ro-', label='Attack Success Rate', linewidth=2, markersize=8)
    plt.plot(epsilons, detection_rates, 'bo-', label='Detection Rate', linewidth=2, markersize=8)
    
    plt.xlabel('Epsilon (Attack Strength)')
    plt.ylabel('Rate')
    plt.title('Attack Success vs Detection Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, eps in enumerate(epsilons):
        plt.annotate(f'{attack_rates[i]:.2f}', (eps, attack_rates[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', color='red')
        plt.annotate(f'{detection_rates[i]:.2f}', (eps, detection_rates[i]), 
                    textcoords="offset points", xytext=(0,-15), ha='center', color='blue')
    
    plt.ylim(0, 1.1)
    plt.show()


def create_confusion_matrix(true_positives: int, false_positives: int,
                          true_negatives: int, false_negatives: int) -> None:
    """
    Create and display confusion matrix
    
    Args:
        true_positives: Number of true positives
        false_positives: Number of false positives
        true_negatives: Number of true negatives
        false_negatives: Number of false negatives
    """
    # Create confusion matrix
    cm = np.array([[true_negatives, false_positives],
                   [false_negatives, true_positives]])
    
    # Create labels
    labels = np.array([['TN', 'FP'],
                      ['FN', 'TP']])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Clean (Predicted)', 'Adversarial (Predicted)'],
                yticklabels=['Clean (Actual)', 'Adversarial (Actual)'])
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.title('Confusion Matrix for Adversarial Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def visualize_concept_profiles(concept_profiles: Dict[int, Dict[str, Any]]) -> None:
    """
    Visualize concept profiles for digits
    
    Args:
        concept_profiles: Dictionary mapping digits to their profiles
    """
    digits = list(range(10))
    even_counts = [concept_profiles[d]['samples_count'] for d in digits if d % 2 == 0]
    odd_counts = [concept_profiles[d]['samples_count'] for d in digits if d % 2 == 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Samples per digit
    ax1.bar(digits, [concept_profiles[d]['samples_count'] for d in digits], 
           color=['lightblue' if d % 2 == 0 else 'lightcoral' for d in digits])
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Distribution by Digit')
    ax1.set_xticks(digits)
    
    # Even vs Odd distribution
    ax2.pie([sum(even_counts), sum(odd_counts)], 
           labels=['Even', 'Odd'], 
           colors=['lightblue', 'lightcoral'],
           autopct='%1.1f%%',
           startangle=90)
    ax2.set_title('Even vs Odd Distribution')
    
    plt.tight_layout()
    plt.show()


def save_visualization(figure, filename: str, dpi: int = 150) -> None:
    """
    Save a matplotlib figure
    
    Args:
        figure: Matplotlib figure object
        filename: Output filename
        dpi: Resolution in DPI
    """
    figure.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"💾 Visualization saved as {filename}")


def plot_attention_statistics_comparison(clean_stats: Dict[str, float],
                                       adv_stats: Dict[str, float]) -> None:
    """
    Compare attention statistics between clean and adversarial images
    
    Args:
        clean_stats: Statistics from clean images
        adv_stats: Statistics from adversarial images
    """
    metrics = ['entropy', 'concentration', 'mean', 'std']
    clean_values = [clean_stats.get(m, 0) for m in metrics]
    adv_values = [adv_stats.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, clean_values, width, label='Clean Images', color='lightgreen')
    bars2 = ax.bar(x + width/2, adv_values, width, label='Adversarial Images', color='lightcoral')
    
    ax.set_xlabel('Attention Statistics')
    ax.set_ylabel('Values')
    ax.set_title('Attention Statistics: Clean vs Adversarial')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()