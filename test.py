#!/usr/bin/env python3
"""
Updated MLP Detector Test Script
Compatible with the current XAI Adversarial Detection repository structure
"""
import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Move into the project directory
os.chdir('/content/XAI_project')

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

sys.path.append(os.getcwd())  # add current dir to path

from datasets.mnist_evenodd import create_data_loaders
from models.vit_wrapper import MNISTViTWrapper, train_even_odd_classifier, create_art_compatible_model
from models.mlp_concept import (
    train_attention_to_concept_mlp,
    detect_adversarial_with_mlp,
    evaluate_mlp_concept_prediction
)
from attacks.fgsm_generator import FGSMAttackGenerator
from config import Config


def test_mlp_detector_comprehensive():
    """
    Comprehensive test of the MLP adversarial detector with ART integration
    """
    print("🧠 COMPREHENSIVE MLP ADVERSARIAL DETECTOR TEST")
    print("=" * 70)
    print(f"Device: {Config.DEVICE}")
    print(f"ART Integration: ✅ Enabled")

    # Step 1: Setup datasets
    print("\n📁 Step 1: Loading MNIST Even-Odd datasets...")
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
        train_size=10000, 
        test_size=5000,
        batch_size=32
    )
    
    train_dataset.analyze_distribution("Training Set")
    test_dataset.analyze_distribution("Test Set")
    print(f"✅ Datasets loaded successfully")

    # Step 2: Setup and train ViT model
    print("\n🧠 Step 2: Setting up ViT model...")
    vit_model = MNISTViTWrapper().to(Config.DEVICE)
    
    print("🏋️ Training ViT for Even/Odd classification...")
    training_results = train_even_odd_classifier(vit_model, train_loader, epochs=1)
    
    # Create ART-compatible wrapper
    art_compatible_model = create_art_compatible_model(vit_model)
    print(f"✅ ViT training completed - Accuracy: {training_results['final_accuracy']:.2f}%")

    # Step 3: Train MLP concept mapper
    print("\n🧠 Step 3: Training MLP Attention-to-Concept mapper...")
    try:
        mlp_model = train_attention_to_concept_mlp(
            vit_model,
            train_dataset,
            epochs=20,
            max_samples=8000
        )
        print("✅ MLP training completed")
    except Exception as e:
        print(f"❌ MLP training failed: {e}")
        return None

    # Step 4: Evaluate MLP concept prediction
    print("\n📊 Step 4: Evaluating MLP concept prediction accuracy...")
    mlp_accuracy = evaluate_mlp_concept_prediction(
        mlp_model, vit_model, test_dataset, max_samples=1000
    )

    # Step 5: Initialize FGSM attacker with ART
    print("\n⚡ Step 5: Setting up FGSM attacker with ART...")
    fgsm_attacker = FGSMAttackGenerator(art_compatible_model)
    print("✅ ART FGSM attacker initialized")

    # Step 6: Test detection on clean images
    print("\n🧪 Step 6: Testing detection on clean images...")
    clean_results = test_clean_detection(vit_model, mlp_model, test_dataset, num_samples=100)

    # Step 7: Test detection on adversarial images
    print("\n⚡ Step 7: Testing detection on adversarial images...")
    adv_results = test_adversarial_detection(
        vit_model, mlp_model, fgsm_attacker, test_dataset, num_samples=100
    )

    # Step 8: Comprehensive attack analysis
    print("\n🔍 Step 8: Comprehensive attack analysis...")
    attack_analysis = analyze_attack_effectiveness(fgsm_attacker, test_dataset, num_samples=50)

    # Step 9: Results summary
    print("\n📊 FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"✅ MLP Concept Accuracy: {mlp_accuracy['concept_accuracy']:.2f}%")
    print(f"✅ Clean Detection (FPR): {clean_results['false_positive_rate']:.3f}")
    print(f"✅ Adversarial Detection (TPR): {adv_results['true_positive_rate']:.3f}")
    print(f"✅ Attack Success Rate: {adv_results['attack_success_rate']:.3f}")
    print(f"✅ MLP Confidence Drop: {adv_results.get('mlp_confidence_drop', 0):.3f}")
    print(f"✅ Concept Change Rate: {adv_results.get('concept_change_rate', 0):.3f}")

    # Step 10: Visualizations
    print("\n📊 Step 10: Creating visualizations...")
    create_comprehensive_visualizations(clean_results, adv_results, attack_analysis)

    return {
        'mlp_accuracy': mlp_accuracy,
        'clean_results': clean_results,
        'adversarial_results': adv_results,
        'attack_analysis': attack_analysis
    }


def test_clean_detection(vit_model, mlp_model, test_dataset, num_samples=100):
    """Test MLP detector on clean images"""
    print(f"   Testing MLP detector on {num_samples} clean images...")
    
    false_positives = 0
    detection_scores = []
    detailed_results = []

    vit_model.eval()
    
    for i in range(min(num_samples, len(test_dataset))):
        try:
            image, concept, task = test_dataset[i]

            # Get ViT prediction using return_dict=True
            with torch.no_grad():
                vit_output = vit_model(
                    image.unsqueeze(0).to(Config.DEVICE), 
                    output_attentions=False, 
                    return_dict=True
                )
                predicted_task = torch.argmax(vit_output['logits'], dim=1).item()
                confidence = torch.softmax(vit_output['logits'], dim=1).max().item()

            # Test MLP detection
            detection_result = detect_adversarial_with_mlp(
                vit_model, mlp_model, image, concept, predicted_task
            )

            detection_scores.append(detection_result['detection_score'])
            
            detailed_results.append({
                'digit': concept,
                'true_task': task,
                'predicted_task': predicted_task,
                'vit_confidence': confidence,
                'mlp_prediction': detection_result['predicted_concept'],
                'mlp_confidence': detection_result['concept_confidence'],
                'detection_score': detection_result['detection_score'],
                'is_adversarial': detection_result['is_adversarial']
            })

            if detection_result['is_adversarial']:
                false_positives += 1
                if i < 5:  # Show first few false positives
                    print(f"      ❌ False positive {false_positives}: Digit {concept} "
                          f"(MLP confidence: {detection_result['concept_confidence']:.3f})")

        except Exception as e:
            print(f"      ⚠️ Error processing clean sample {i}: {e}")
            continue

    fpr = false_positives / num_samples
    avg_score = np.mean(detection_scores) if detection_scores else 0
    
    print(f"   📊 Clean Images Results:")
    print(f"      False Positives: {false_positives}/{num_samples}")
    print(f"      False Positive Rate: {fpr:.3f}")
    print(f"      Average Detection Score: {avg_score:.3f}")

    return {
        'false_positive_rate': fpr,
        'false_positives': false_positives,
        'total_samples': num_samples,
        'detection_scores': detection_scores,
        'average_score': avg_score,
        'detailed_results': detailed_results
    }


def test_adversarial_detection(vit_model, mlp_model, fgsm_attacker, test_dataset, num_samples=100):
    """Test MLP detector on adversarial images with detailed analysis"""
    print(f"   Testing MLP detector on {num_samples} adversarial images...")
    
    true_positives = 0
    total_adversarial = 0
    successful_attacks = 0
    detection_scores = []
    
    # Confidence tracking
    clean_mlp_confidences = []
    adv_mlp_confidences = []
    concept_changes = []
    detailed_results = []

    vit_model.eval()
    epsilon = 0.25  # Use moderate attack strength

    for i in range(min(num_samples, len(test_dataset))):
        try:
            image, concept, task = test_dataset[i]

            # Get clean prediction
            with torch.no_grad():
                clean_vit_output = vit_model(
                    image.unsqueeze(0).to(Config.DEVICE), 
                    output_attentions=False, 
                    return_dict=True
                )
                clean_task_pred = torch.argmax(clean_vit_output['logits'], dim=1).item()
                clean_vit_conf = torch.softmax(clean_vit_output['logits'], dim=1).max().item()

            # Get clean MLP prediction
            clean_mlp_result = detect_adversarial_with_mlp(
                vit_model, mlp_model, image, concept, clean_task_pred
            )
            clean_mlp_confidences.append(clean_mlp_result['concept_confidence'])

            # Generate adversarial example using ART
            adv_image = fgsm_attacker.generate_fgsm_attack(image, task, epsilon=epsilon)
            
            # Test attack success
            attack_result = fgsm_attacker.test_attack_success(image, adv_image, task)
            
            if attack_result['attack_success']:
                successful_attacks += 1
                
                # Test MLP detection on adversarial
                adv_mlp_result = detect_adversarial_with_mlp(
                    vit_model, mlp_model, 
                    adv_image.squeeze(0) if len(adv_image.shape) == 4 else adv_image,
                    concept,
                    attack_result['adversarial_prediction']
                )
                
                adv_mlp_confidences.append(adv_mlp_result['concept_confidence'])
                concept_changes.append(
                    clean_mlp_result['predicted_concept'] != adv_mlp_result['predicted_concept']
                )
                detection_scores.append(adv_mlp_result['detection_score'])
                
                if adv_mlp_result['is_adversarial']:
                    true_positives += 1
                
                detailed_results.append({
                    'digit': concept,
                    'true_task': task,
                    'clean_task_pred': clean_task_pred,
                    'adv_task_pred': attack_result['adversarial_prediction'],
                    'clean_mlp_concept': clean_mlp_result['predicted_concept'],
                    'adv_mlp_concept': adv_mlp_result['predicted_concept'],
                    'clean_mlp_conf': clean_mlp_result['concept_confidence'],
                    'adv_mlp_conf': adv_mlp_result['concept_confidence'],
                    'detection_score': adv_mlp_result['detection_score'],
                    'detected': adv_mlp_result['is_adversarial'],
                    'concept_changed': clean_mlp_result['predicted_concept'] != adv_mlp_result['predicted_concept']
                })
                
                # Show detailed results for first few samples
                if i < 5:
                    print(f"\n      🔍 Sample {i+1}: Digit {concept}")
                    print(f"         Task: True={task} → Clean_pred={clean_task_pred} → Adv_pred={attack_result['adversarial_prediction']}")
                    print(f"         MLP: Clean_concept={clean_mlp_result['predicted_concept']}({clean_mlp_result['concept_confidence']:.3f}) → "
                          f"Adv_concept={adv_mlp_result['predicted_concept']}({adv_mlp_result['concept_confidence']:.3f})")
                    print(f"         Detection: Score={adv_mlp_result['detection_score']:.3f}, "
                          f"Detected={'✅' if adv_mlp_result['is_adversarial'] else '❌'}")
                
                total_adversarial += 1

        except Exception as e:
            print(f"      ⚠️ Error processing adversarial sample {i}: {e}")
            continue

    # Calculate metrics
    tpr = true_positives / total_adversarial if total_adversarial > 0 else 0
    attack_success_rate = successful_attacks / num_samples
    avg_score = np.mean(detection_scores) if detection_scores else 0
    
    # Confidence analysis
    mlp_conf_drop = (np.mean(clean_mlp_confidences) - np.mean(adv_mlp_confidences)) if adv_mlp_confidences else 0
    concept_change_rate = np.mean(concept_changes) if concept_changes else 0

    print(f"\n   📊 Adversarial Detection Results:")
    print(f"      Attack Success Rate: {attack_success_rate:.3f}")
    print(f"      True Positives: {true_positives}/{total_adversarial}")
    print(f"      True Positive Rate: {tpr:.3f}")
    print(f"      Average Detection Score: {avg_score:.3f}")
    print(f"      MLP Confidence Drop: {mlp_conf_drop:.3f}")
    print(f"      Concept Change Rate: {concept_change_rate:.3f}")

    return {
        'true_positive_rate': tpr,
        'true_positives': true_positives,
        'total_adversarial': total_adversarial,
        'attack_success_rate': attack_success_rate,
        'detection_scores': detection_scores,
        'average_score': avg_score,
        'mlp_confidence_drop': mlp_conf_drop,
        'concept_change_rate': concept_change_rate,
        'detailed_results': detailed_results
    }


def analyze_attack_effectiveness(fgsm_attacker, test_dataset, num_samples=50):
    """Analyze attack effectiveness across different epsilon values"""
    print(f"   Analyzing attack effectiveness on {num_samples} samples...")
    
    epsilon_values = [0.05, 0.1, 0.15, 0.2, 0.3]
    attack_analysis = []

    for i in range(min(num_samples, len(test_dataset))):
        try:
            image, concept, task = test_dataset[i]
            
            sample_analysis = {
                'digit': concept,
                'true_task': task,
                'epsilon_results': {}
            }

            for epsilon in epsilon_values:
                try:
                    # Generate adversarial example
                    adv_image = fgsm_attacker.generate_pgd_attack(image, task, epsilon=epsilon)
                    
                    # Test attack success
                    attack_result = fgsm_attacker.test_attack_success(image, adv_image, task)
                    
                    sample_analysis['epsilon_results'][epsilon] = {
                        'attack_success': attack_result['attack_success'],
                        'confidence_drop': attack_result['confidence_drop'],
                        'adv_prediction': attack_result['adversarial_prediction']
                    }
                except Exception as e:
                    print(f"         ⚠️ Error with epsilon {epsilon}: {e}")
                    sample_analysis['epsilon_results'][epsilon] = {
                        'attack_success': False,
                        'confidence_drop': 0,
                        'adv_prediction': task
                    }

            attack_analysis.append(sample_analysis)
            
        except Exception as e:
            print(f"      ⚠️ Error analyzing sample {i}: {e}")
            continue

    # Print summary
    print(f"\n   📈 Attack Effectiveness Summary:")
    for epsilon in epsilon_values:
        successes = sum(1 for sample in attack_analysis 
                       if sample['epsilon_results'][epsilon]['attack_success'])
        rate = successes / len(attack_analysis) if attack_analysis else 0
        
        avg_conf_drop = np.mean([sample['epsilon_results'][epsilon]['confidence_drop'] 
                                for sample in attack_analysis]) if attack_analysis else 0
        
        print(f"      ε={epsilon}: Success Rate={rate:.3f}, Avg Confidence Drop={avg_conf_drop:.3f}")

    return attack_analysis


def create_comprehensive_visualizations(clean_results, adv_results, attack_analysis):
    """Create comprehensive visualizations of detection results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MLP-Based Adversarial Detection Results', fontsize=16, fontweight='bold')

    # 1. Detection scores distribution
    clean_scores = clean_results['detection_scores']
    adv_scores = adv_results['detection_scores']

    axes[0,0].hist(clean_scores, alpha=0.7, label='Clean Images', bins=20, color='green', density=True)
    axes[0,0].hist(adv_scores, alpha=0.7, label='Adversarial Images', bins=20, color='red', density=True)
    axes[0,0].axvline(x=Config.DETECTION_THRESHOLD, color='black', linestyle='--', label='Threshold')
    axes[0,0].set_xlabel('Detection Score')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Detection Score Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Performance metrics
    metrics = ['FPR', 'TPR', 'Attack Success']
    values = [
        clean_results['false_positive_rate'],
        adv_results['true_positive_rate'],
        adv_results['attack_success_rate']
    ]
    colors = ['red', 'green', 'orange']

    bars = axes[0,1].bar(metrics, values, color=colors, alpha=0.7)
    axes[0,1].set_ylabel('Rate')
    axes[0,1].set_title('Detection Performance Metrics')
    axes[0,1].set_ylim(0, 1)

    for bar, value in zip(bars, values):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 3. Confidence analysis
    confidence_metrics = ['MLP Conf Drop', 'Concept Change Rate']
    confidence_values = [
        adv_results.get('mlp_confidence_drop', 0),
        adv_results.get('concept_change_rate', 0)
    ]

    bars2 = axes[0,2].bar(confidence_metrics, confidence_values, color=['blue', 'purple'], alpha=0.7)
    axes[0,2].set_ylabel('Value')
    axes[0,2].set_title('Confidence Analysis')
    axes[0,2].set_ylim(0, max(confidence_values) + 0.1 if confidence_values else 1)

    for bar, value in zip(bars2, confidence_values):
        axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 4. Attack success vs epsilon
    if attack_analysis:
        epsilon_values = [0.05, 0.1, 0.15, 0.2, 0.3]
        success_rates = []
        
        for epsilon in epsilon_values:
            successes = sum(1 for sample in attack_analysis 
                           if sample['epsilon_results'][epsilon]['attack_success'])
            rate = successes / len(attack_analysis) if attack_analysis else 0
            success_rates.append(rate)

        axes[1,0].plot(epsilon_values, success_rates, 'ro-', linewidth=2, markersize=8)
        axes[1,0].set_xlabel('Epsilon (Attack Strength)')
        axes[1,0].set_ylabel('Attack Success Rate')
        axes[1,0].set_title('Attack Success vs Epsilon')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim(0, 1)

        for eps, rate in zip(epsilon_values, success_rates):
            axes[1,0].annotate(f'{rate:.2f}', (eps, rate), 
                              textcoords="offset points", xytext=(0,10), ha='center')

    # 5. Per-digit detection accuracy
    if 'detailed_results' in adv_results:
        digit_detection = {}
        for result in adv_results['detailed_results']:
            digit = result['digit']
            if digit not in digit_detection:
                digit_detection[digit] = {'detected': 0, 'total': 0}
            digit_detection[digit]['total'] += 1
            if result['detected']:
                digit_detection[digit]['detected'] += 1

        digits = sorted(digit_detection.keys())
        detection_rates = [digit_detection[d]['detected'] / digit_detection[d]['total'] 
                          if digit_detection[d]['total'] > 0 else 0 for d in digits]
        colors = ['lightblue' if d % 2 == 0 else 'lightcoral' for d in digits]

        axes[1,1].bar(digits, detection_rates, color=colors, alpha=0.7)
        axes[1,1].set_xlabel('Digit')
        axes[1,1].set_ylabel('Detection Rate')
        axes[1,1].set_title('Detection Rate by Digit')
        axes[1,1].set_ylim(0, 1)
        axes[1,1].grid(True, alpha=0.3)

    # 6. Average scores comparison
    score_types = ['Clean Avg', 'Adversarial Avg']
    avg_scores = [
        clean_results['average_score'],
        adv_results['average_score']
    ]

    bars3 = axes[1,2].bar(score_types, avg_scores, color=['lightgreen', 'lightcoral'], alpha=0.7)
    axes[1,2].set_ylabel('Average Detection Score')
    axes[1,2].set_title('Average Detection Scores')
    axes[1,2].set_ylim(0, 1)

    for bar, value in zip(bars3, avg_scores):
        axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the comprehensive test"""
    try:
        print("🚀 Starting Comprehensive MLP Detector Test")
        print("Compatible with updated XAI Adversarial Detection repository")
        print("="*70)
        
        results = test_mlp_detector_comprehensive()
        
        if results:
            print("\n🎉 Test completed successfully!")
            print("\n📋 Key Takeaways:")
            print(f"   • MLP can predict digit concepts with {results['mlp_accuracy']['concept_accuracy']:.1f}% accuracy")
            print(f"   • False positive rate: {results['clean_results']['false_positive_rate']:.3f}")
            print(f"   • True positive rate: {results['adversarial_results']['true_positive_rate']:.3f}")
            print(f"   • FGSM attacks succeed {results['adversarial_results']['attack_success_rate']:.3f} of the time")
            print(f"   • MLP confidence drops by {results['adversarial_results'].get('mlp_confidence_drop', 0):.3f} under attack")
        else:
            print("\n❌ Test failed")
            
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()