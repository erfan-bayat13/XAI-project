"""
Concept-Based Adversarial Detector using attention-concept inconsistencies
"""
import torch
import numpy as np
from typing import Dict, Any, List
from config import Config


class ConceptBasedDetector:
    """
    Concept-based adversarial detector using attention-concept inconsistencies
    """

    def __init__(self, vit_analyzer, concept_profiles: Dict[int, Dict[str, Any]]):
        """
        Initialize the concept-based detector
        
        Args:
            vit_analyzer: The ViT attention analyzer
            concept_profiles: Dictionary mapping concepts to their profiles
        """
        self.vit_analyzer = vit_analyzer
        self.concept_profiles = concept_profiles
        self.device = Config.DEVICE

        # Store baseline attention statistics for clean images
        self.baseline_stats = None
        
        print("🛡️ Concept-Based Adversarial Detector initialized")

    def compute_attention_statistics(self, attention_maps: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical features from attention maps
        
        Args:
            attention_maps: List of attention map tensors
            
        Returns:
            Dictionary of statistics per layer
        """
        stats = {}

        # Focus on middle layers (6-8) where concepts emerge
        for layer_idx in Config.MIDDLE_LAYERS:
            if layer_idx < len(attention_maps):
                attention_map = attention_maps[layer_idx]
                attention_2d = attention_map.reshape(14, 14)

                stats[f'layer_{layer_idx}'] = {
                    'mean': float(attention_2d.mean()),
                    'std': float(attention_2d.std()),
                    'max': float(attention_2d.max()),
                    'min': float(attention_2d.min()),
                    'entropy': float(-torch.sum(attention_2d * torch.log(attention_2d + 1e-8))),
                    'concentration': float(torch.sum(attention_2d ** 2)),  # How concentrated is attention
                    'uniformity': float(1.0 / (attention_2d.std() + 1e-8))  # How uniform is attention
                }

        return stats

    def compute_cross_layer_consistency(self, middle_layer_maps: List[torch.Tensor]) -> float:
        """
        Compute consistency of attention patterns across middle layers
        Clean images should have consistent attention, adversarial images should be chaotic
        
        Args:
            middle_layer_maps: List of attention maps from middle layers
            
        Returns:
            Average consistency score across layers
        """
        if len(middle_layer_maps) < 2:
            return 1.0

        consistencies = []

        for i in range(len(middle_layer_maps) - 1):
            map1 = middle_layer_maps[i].reshape(14, 14)
            map2 = middle_layer_maps[i + 1].reshape(14, 14)

            # Compute correlation between consecutive layers
            map1_flat = map1.flatten()
            map2_flat = map2.flatten()

            # Pearson correlation
            correlation = torch.corrcoef(torch.stack([map1_flat, map2_flat]))[0, 1]
            
            # Handle NaN correlations
            if not torch.isnan(correlation):
                consistencies.append(float(correlation))

        return np.mean(consistencies) if consistencies else 0.0

    def compute_detection_score(self, 
                              entropy: float, 
                              concentration: float, 
                              confidence: float, 
                              consistency: float) -> float:
        """
        Combine different signals into a single detection score
        Higher score = more likely to be adversarial
        
        Args:
            entropy: Attention entropy score
            concentration: Attention concentration score
            confidence: Model confidence score
            consistency: Cross-layer consistency score
            
        Returns:
            Detection score (0-1, higher means more likely adversarial)
        """
        # Normalize scores (these would be learned from training data)
        # For now, using reasonable heuristics

        # High entropy suggests adversarial (chaotic attention)
        entropy_signal = min(entropy / 10.0, 1.0)

        # Low concentration suggests adversarial (scattered attention)
        concentration_signal = max(0, 1.0 - concentration / 0.1)

        # Low confidence might suggest adversarial
        confidence_signal = max(0, 1.0 - confidence)

        # Low consistency suggests adversarial (inconsistent across layers)
        consistency_signal = max(0, 1.0 - consistency) if consistency is not None else 0.5

        # Weighted combination
        detection_score = (
            0.3 * entropy_signal +
            0.3 * concentration_signal +
            0.2 * confidence_signal +
            0.2 * consistency_signal
        )

        return min(detection_score, 1.0)

    def detect_concept_inconsistency(self, 
                                   image: torch.Tensor, 
                                   predicted_task: int, 
                                   predicted_concept: int = None) -> Dict[str, Any]:
        """
        Detect adversarial examples based on concept inconsistencies

        This is our core detection method!
        
        Args:
            image: Input image tensor
            predicted_task: Predicted task (0=Even, 1=Odd)
            predicted_concept: Predicted concept (digit), optional
            
        Returns:
            Dictionary with detection results
        """
        # Extract attention maps
        attention_maps, outputs = self.vit_analyzer.extract_attention_maps(image)

        # Compute attention statistics
        attention_stats = self.compute_attention_statistics(attention_maps)

        # Get prediction details
        prediction = self.vit_analyzer.predict_even_odd(image)

        # Detection Method 1: Attention Entropy Analysis
        # FGSM attacks create chaotic attention patterns with high entropy
        entropy_scores = []
        for layer_idx in Config.MIDDLE_LAYERS:
            layer_key = f'layer_{layer_idx}'
            if layer_key in attention_stats:
                entropy_scores.append(attention_stats[layer_key]['entropy'])

        avg_entropy = np.mean(entropy_scores) if entropy_scores else 0

        # Detection Method 2: Attention Concentration Analysis
        # Clean images have focused attention, adversarial images have scattered attention
        concentration_scores = []
        for layer_idx in Config.MIDDLE_LAYERS:
            layer_key = f'layer_{layer_idx}'
            if layer_key in attention_stats:
                concentration_scores.append(attention_stats[layer_key]['concentration'])

        avg_concentration = np.mean(concentration_scores) if concentration_scores else 0

        # Detection Method 3: Prediction Confidence Analysis
        # Adversarial examples often have lower confidence or inconsistent confidence
        confidence = prediction['confidence']

        # Detection Method 4: Cross-Layer Consistency
        # Check if attention patterns are consistent across layers
        middle_maps = [attention_maps[i] for i in Config.MIDDLE_LAYERS if i < len(attention_maps)]
        layer_consistency = self.compute_cross_layer_consistency(middle_maps)

        # Combine detection signals
        detection_score = self.compute_detection_score(
            avg_entropy, avg_concentration, confidence, layer_consistency
        )

        # Threshold-based detection
        is_adversarial = detection_score > Config.DETECTION_THRESHOLD

        return {
            'is_adversarial': is_adversarial,
            'detection_score': detection_score,
            'entropy': avg_entropy,
            'concentration': avg_concentration,
            'confidence': confidence,
            'layer_consistency': layer_consistency,
            'attention_stats': attention_stats,
            'threshold': Config.DETECTION_THRESHOLD,
            'prediction': prediction
        }

    def batch_detect(self, 
                    images: List[torch.Tensor], 
                    predicted_tasks: List[int]) -> List[Dict[str, Any]]:
        """
        Detect adversarial examples in a batch
        
        Args:
            images: List of image tensors
            predicted_tasks: List of predicted tasks
            
        Returns:
            List of detection results
        """
        results = []
        
        for image, task in zip(images, predicted_tasks):
            result = self.detect_concept_inconsistency(image, task)
            results.append(result)
            
        return results

    def evaluate_detection_performance(self, 
                                     clean_samples: List[torch.Tensor],
                                     adversarial_samples: List[torch.Tensor],
                                     clean_tasks: List[int],
                                     adv_tasks: List[int]) -> Dict[str, float]:
        """
        Evaluate detection performance on clean and adversarial samples
        
        Args:
            clean_samples: List of clean image tensors
            adversarial_samples: List of adversarial image tensors
            clean_tasks: List of clean image tasks
            adv_tasks: List of adversarial image tasks
            
        Returns:
            Dictionary with performance metrics
        """
        print("📊 Evaluating detection performance...")
        
        # Test on clean samples (should detect as clean)
        clean_results = []
        for image, task in zip(clean_samples, clean_tasks):
            result = self.detect_concept_inconsistency(image, task)
            clean_results.append(result)
        
        # Test on adversarial samples (should detect as adversarial)
        adv_results = []
        for image, task in zip(adversarial_samples, adv_tasks):
            result = self.detect_concept_inconsistency(image, task)
            adv_results.append(result)
        
        # Compute metrics
        false_positives = sum(1 for r in clean_results if r['is_adversarial'])
        true_negatives = len(clean_results) - false_positives
        
        true_positives = sum(1 for r in adv_results if r['is_adversarial'])
        false_negatives = len(adv_results) - true_positives
        
        # Calculate rates
        fpr = false_positives / len(clean_results) if clean_results else 0
        tpr = true_positives / len(adv_results) if adv_results else 0
        
        accuracy = (true_positives + true_negatives) / (len(clean_results) + len(adv_results))
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = tpr
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   False Positive Rate: {fpr:.3f}")
        print(f"   True Positive Rate: {tpr:.3f}")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1_score:.3f}")
        
        return {
            'false_positive_rate': fpr,
            'true_positive_rate': tpr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }

    def set_detection_threshold(self, threshold: float):
        """Set detection threshold"""
        Config.DETECTION_THRESHOLD = threshold
        print(f"🎯 Detection threshold set to {threshold}")

    def optimize_threshold(self, 
                          clean_samples: List[torch.Tensor],
                          adversarial_samples: List[torch.Tensor],
                          clean_tasks: List[int],
                          adv_tasks: List[int],
                          target_fpr: float = 0.05) -> float:
        """
        Optimize detection threshold for target false positive rate
        
        Args:
            clean_samples: Clean samples for evaluation
            adversarial_samples: Adversarial samples for evaluation
            clean_tasks: Clean sample tasks
            adv_tasks: Adversarial sample tasks
            target_fpr: Target false positive rate
            
        Returns:
            Optimized threshold
        """
        print(f"🎯 Optimizing detection threshold for FPR={target_fpr}")
        
        # Get detection scores for clean samples
        clean_scores = []
        for image, task in zip(clean_samples, clean_tasks):
            result = self.detect_concept_inconsistency(image, task)
            clean_scores.append(result['detection_score'])
        
        # Find threshold that gives target FPR
        clean_scores.sort()
        threshold_idx = int((1 - target_fpr) * len(clean_scores))
        optimal_threshold = clean_scores[min(threshold_idx, len(clean_scores) - 1)]
        
        # Set the new threshold
        self.set_detection_threshold(optimal_threshold)
        
        # Evaluate with new threshold
        performance = self.evaluate_detection_performance(
            clean_samples, adversarial_samples, clean_tasks, adv_tasks
        )
        
        print(f"✅ Optimized threshold: {optimal_threshold:.3f}")
        return optimal_threshold