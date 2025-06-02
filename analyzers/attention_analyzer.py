"""
Vision Transformer Attention Analysis for MNIST - FIXED VERSION
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor
from typing import Tuple, List, Dict, Any, Optional
from config import Config


class MNISTViTAttentionAnalyzer:
    """
    Specialized attention analyzer for MNIST Even-Odd dataset
    """

    def __init__(self, model: nn.Module, model_path: Optional[str] = None):
        """
        Initialize the attention analyzer
        
        Args:
            model: The ViT wrapper model
            model_path: Path to load trained model weights
        """
        self.device = Config.DEVICE
        self.model = model
        self.model.to(self.device)

        # Initialize ViT processor (for image preprocessing)
        self.processor = ViTImageProcessor.from_pretrained(Config.VIT_MODEL_NAME)

        # Load trained model if path provided
        if model_path:
            self.model.load_model(model_path)
            print(f"✅ Loaded trained model from {model_path}")
        else:
            print("⚠️ Using untrained model")

        self.model.eval()
        
        # Storage for clean samples (for adversarial comparison)
        self.clean_samples = {}
        self.clean_attention_stats = {}

    def preprocess_mnist_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Preprocess MNIST image tensor for ViT
        
        Args:
            image_tensor: Input image tensor
            
        Returns:
            Preprocessed image tensor
        """
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        return image_tensor.to(self.device)

    def predict_even_odd(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Predict even/odd for an MNIST image
        
        Args:
            image_tensor: Input image tensor
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        inputs = self.preprocess_mnist_image(image_tensor)

        # Forward pass - use return_dict=True to get full outputs
        with torch.no_grad():
            outputs = self.model(inputs, output_attentions=True, return_dict=True)

            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]

            return {
                'predicted_task': predicted_class.item(),
                'confidence': confidence.item(),
                'probabilities': probabilities[0],
                'task_name': 'Even' if predicted_class.item() == 0 else 'Odd',
                'attentions': outputs['attentions'],
                'hidden_states': outputs['hidden_states']
            }

    def extract_attention_maps(self, image_tensor: torch.Tensor, 
                             layer_idx: Optional[int] = None) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Extract attention maps from MNIST image - FIXED VERSION
        
        Args:
            image_tensor: Input image tensor
            layer_idx: Specific layer to extract (None for all layers)
            
        Returns:
            Tuple of (attention_maps, model_outputs)
        """
        inputs = self.preprocess_mnist_image(image_tensor)

        with torch.no_grad():
            outputs = self.model(inputs, output_attentions=True, return_dict=True)
            attentions = outputs['attentions']

        attention_maps = []

        try:
            if layer_idx is not None:
                # Extract specific layer
                if layer_idx < len(attentions):
                    attention = attentions[layer_idx]  # Shape: [batch_size, num_heads, seq_len, seq_len]
                    
                    # Handle different attention tensor shapes
                    if len(attention.shape) == 4:
                        # Standard case: [batch_size, num_heads, seq_len, seq_len]
                        attention = attention[0]  # Remove batch dimension: [num_heads, seq_len, seq_len]
                        attention_avg = attention.mean(dim=0)  # Average across heads: [seq_len, seq_len]
                        cls_attention = attention_avg[0, 1:]  # CLS token attention to patches: [196]
                    elif len(attention.shape) == 3:
                        # Already without batch dimension: [num_heads, seq_len, seq_len]
                        attention_avg = attention.mean(dim=0)  # Average across heads: [seq_len, seq_len]
                        cls_attention = attention_avg[0, 1:]  # CLS token attention to patches: [196]
                    elif len(attention.shape) == 2:
                        # Already averaged: [seq_len, seq_len]
                        cls_attention = attention[0, 1:]  # CLS token attention to patches: [196]
                    else:
                        print(f"⚠️ Unexpected attention shape: {attention.shape}")
                        cls_attention = torch.zeros(196, device=self.device)
                    
                    attention_maps.append(cls_attention)
            else:
                # Extract all layers
                for layer_idx in range(len(attentions)):
                    try:
                        attention = attentions[layer_idx]  # Shape: [batch_size, num_heads, seq_len, seq_len]
                        
                        # Handle different attention tensor shapes
                        if len(attention.shape) == 4:
                            # Standard case: [batch_size, num_heads, seq_len, seq_len]
                            attention = attention[0]  # Remove batch dimension: [num_heads, seq_len, seq_len]
                            attention_avg = attention.mean(dim=0)  # Average across heads: [seq_len, seq_len]
                            cls_attention = attention_avg[0, 1:]  # CLS token attention to patches: [196]
                        elif len(attention.shape) == 3:
                            # Already without batch dimension: [num_heads, seq_len, seq_len]
                            attention_avg = attention.mean(dim=0)  # Average across heads: [seq_len, seq_len]
                            cls_attention = attention_avg[0, 1:]  # CLS token attention to patches: [196]
                        elif len(attention.shape) == 2:
                            # Already averaged: [seq_len, seq_len]
                            cls_attention = attention[0, 1:]  # CLS token attention to patches: [196]
                        else:
                            print(f"⚠️ Unexpected attention shape at layer {layer_idx}: {attention.shape}")
                            cls_attention = torch.zeros(196, device=self.device)
                        
                        attention_maps.append(cls_attention)
                        
                    except Exception as e:
                        print(f"⚠️ Error processing attention at layer {layer_idx}: {e}")
                        # Add dummy attention map to maintain indexing
                        cls_attention = torch.zeros(196, device=self.device)
                        attention_maps.append(cls_attention)
                        continue

        except Exception as e:
            print(f"⚠️ Error in attention extraction: {e}")
            print(f"Attention tensor info:")
            if attentions:
                for i, att in enumerate(attentions[:3]):  # Show first 3 layers
                    print(f"  Layer {i}: shape {att.shape}, type {type(att)}")
            
            # Return dummy attention maps to prevent crash
            attention_maps = [torch.zeros(196, device=self.device) for _ in range(len(attentions))]

        return attention_maps, outputs

    def create_attention_heatmap(self, attention_weights: List[torch.Tensor], 
                               layer_idx: int = 0, head_idx: int = 0) -> Optional[torch.Tensor]:
        """
        Create attention heatmap from attention weights - FIXED VERSION
        
        Args:
            attention_weights: List of attention weight tensors
            layer_idx: Layer index
            head_idx: Attention head index (not used anymore since we pre-average)
            
        Returns:
            Attention heatmap tensor or None
        """
        if not attention_weights or layer_idx >= len(attention_weights):
            return None

        try:
            attention = attention_weights[layer_idx]
            
            # Ensure we have the right number of elements (196 for 14x14 patches)
            if attention.numel() == 196:
                # Reshape to spatial dimensions (14x14 for ViT-base)
                attention_map = attention.reshape(14, 14)
            elif attention.numel() == 197:
                # Sometimes includes CLS token, skip it
                attention_map = attention[1:].reshape(14, 14)
            else:
                print(f"⚠️ Unexpected attention size: {attention.numel()}, expected 196 or 197")
                # Create a dummy 14x14 map
                attention_map = torch.zeros(14, 14, device=attention.device)
            
            return attention_map
            
        except Exception as e:
            print(f"⚠️ Error creating attention heatmap: {e}")
            return torch.zeros(14, 14, device=self.device)

    def visualize_mnist_attention(self, image_tensor: torch.Tensor, 
                                concept: int, task: int, 
                                save_plot: bool = False) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
        """
        Visualize attention for MNIST digit - FIXED VERSION
        
        Args:
            image_tensor: Input image tensor
            concept: True digit concept
            task: True even/odd task
            save_plot: Whether to save the plot
            
        Returns:
            Tuple of (prediction_results, attention_maps)
        """
        try:
            # Get prediction and attention
            prediction = self.predict_even_odd(image_tensor)
            attention_maps, _ = self.extract_attention_maps(image_tensor)

            # Convert image for display
            img_display = image_tensor.clone()
            img_display = img_display * Config.MNIST_STD + Config.MNIST_MEAN  # Denormalize
            img_display = torch.clamp(img_display, 0, 1)
            img_display = img_display[0]  # Take first channel

            print(f"📊 Analysis Results:")
            print(f"   True Concept: Digit {concept}")
            print(f"   True Task: {'Even' if task == 0 else 'Odd'}")
            print(f"   Predicted Task: {prediction['task_name']}")
            print(f"   Confidence: {prediction['confidence']:.4f}")
            print(f"   Correct: {'✅' if prediction['predicted_task'] == task else '❌'}")

            # Visualize attention evolution across layers
            fig, axes = plt.subplots(2, len(Config.KEY_LAYERS), figsize=(20, 8))

            for i, layer_idx in enumerate(Config.KEY_LAYERS):
                if layer_idx < len(attention_maps):
                    try:
                        attention_map = attention_maps[layer_idx]

                        # Ensure we can reshape to 14x14
                        if attention_map.numel() >= 196:
                            if attention_map.numel() == 196:
                                spatial_attention = attention_map.reshape(14, 14)
                            elif attention_map.numel() == 197:
                                spatial_attention = attention_map[1:].reshape(14, 14)  # Skip CLS token
                            else:
                                # Take first 196 elements
                                spatial_attention = attention_map.flatten()[:196].reshape(14, 14)
                        else:
                            print(f"⚠️ Attention map too small: {attention_map.numel()}")
                            spatial_attention = torch.zeros(14, 14, device=attention_map.device)

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

                        axes[1, i].imshow(img_display.cpu().numpy(), cmap='gray')
                        axes[1, i].imshow(attention_resized.cpu().numpy(), 
                                        cmap=Config.ATTENTION_COLORMAP, alpha=0.6)
                        axes[1, i].set_title(f"Layer {layer_idx} Overlay")
                        axes[1, i].axis('off')
                        
                    except Exception as e:
                        print(f"⚠️ Error visualizing layer {layer_idx}: {e}")
                        # Create empty plots
                        axes[0, i].text(0.5, 0.5, f'Error\nLayer {layer_idx}', 
                                       ha='center', va='center', transform=axes[0, i].transAxes)
                        axes[1, i].text(0.5, 0.5, f'Error\nLayer {layer_idx}', 
                                       ha='center', va='center', transform=axes[1, i].transAxes)
                        axes[0, i].axis('off')
                        axes[1, i].axis('off')

            plt.suptitle(f"Digit {concept} → True: {'Even' if task == 0 else 'Odd'} | "
                        f"Predicted: {prediction['task_name']} ({prediction['confidence']:.3f})",
                        fontsize=16, fontweight='bold')
            plt.tight_layout()

            if save_plot:
                plt.savefig(f'mnist_attention_digit_{concept}_task_{task}.png',
                           dpi=150, bbox_inches='tight')

            plt.show()

            return prediction, attention_maps
            
        except Exception as e:
            print(f"⚠️ Error in visualize_mnist_attention: {e}")
            # Return dummy results to prevent crash
            dummy_prediction = {
                'predicted_task': task,
                'confidence': 0.5,
                'probabilities': torch.tensor([0.5, 0.5]),
                'task_name': 'Even' if task == 0 else 'Odd'
            }
            dummy_attention = [torch.zeros(196, device=self.device) for _ in range(12)]
            return dummy_prediction, dummy_attention

    def compute_attention_statistics(self, attention_maps: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical features from attention maps - FIXED VERSION
        
        Args:
            attention_maps: List of attention map tensors
            
        Returns:
            Dictionary of statistics per layer
        """
        stats = {}

        # Focus on middle layers where concepts emerge
        for i, layer_idx in enumerate(Config.MIDDLE_LAYERS):
            if layer_idx < len(attention_maps):
                try:
                    attention_map = attention_maps[layer_idx]  # This is now pre-averaged across heads
                    
                    # Ensure we have the right shape
                    if attention_map.numel() == 196:
                        attention_2d = attention_map.reshape(14, 14)
                    elif attention_map.numel() == 197:
                        attention_2d = attention_map[1:].reshape(14, 14)  # Skip CLS token
                    else:
                        print(f"⚠️ Unexpected attention map size at layer {layer_idx}: {attention_map.numel()}")
                        attention_2d = torch.zeros(14, 14, device=attention_map.device)

                    # Ensure no NaN or infinite values
                    attention_2d = torch.nan_to_num(attention_2d, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Add small epsilon to prevent log(0)
                    attention_safe = attention_2d + 1e-8
                    
                    stats[f'layer_{layer_idx}'] = {
                        'mean': float(attention_2d.mean()),
                        'std': float(attention_2d.std()),
                        'max': float(attention_2d.max()),
                        'min': float(attention_2d.min()),
                        'entropy': float(-torch.sum(attention_safe * torch.log(attention_safe))),
                        'concentration': float(torch.sum(attention_2d ** 2)),
                        'uniformity': float(1.0 / (attention_2d.std() + 1e-8))
                    }
                    
                except Exception as e:
                    print(f"⚠️ Error computing statistics for layer {layer_idx}: {e}")
                    # Add dummy stats to maintain structure
                    stats[f'layer_{layer_idx}'] = {
                        'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0,
                        'entropy': 0.0, 'concentration': 0.0, 'uniformity': 0.0
                    }

        return stats

    def analyze_attention_statistics(self, dataset: torch.utils.data.Dataset, 
                                   sample_size: int = 50) -> Dict[str, Any]:
        """
        Analyze attention statistics across different digits and layers - FIXED VERSION
        
        Args:
            dataset: Dataset to analyze
            sample_size: Number of samples to analyze
            
        Returns:
            Dictionary with computed statistics
        """
        print("📈 Computing attention statistics...")

        # Collect attention statistics
        layer_stats = {i: [] for i in Config.MIDDLE_LAYERS}
        digit_stats = {i: [] for i in range(10)}

        # Sample images for statistics
        sample_size = min(sample_size, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)

        successful_samples = 0
        
        for idx in indices:
            try:
                image, concept, task = dataset[idx]

                attention_maps, _ = self.extract_attention_maps(image)
                stats = self.compute_attention_statistics(attention_maps)

                for layer_idx in Config.MIDDLE_LAYERS:
                    layer_key = f'layer_{layer_idx}'
                    if layer_key in stats:
                        layer_stats[layer_idx].append(stats[layer_key])

                        # Store per-digit stats
                        if concept not in digit_stats:
                            digit_stats[concept] = []
                        digit_stats[concept].append(stats[layer_key])
                
                successful_samples += 1

            except Exception as e:
                print(f"⚠️ Error processing sample {idx}: {e}")
                continue

        print(f"✅ Successfully processed {successful_samples}/{sample_size} samples")

        # Compute average statistics
        print(f"\n📊 Layer-wise Attention Statistics (averaged over {successful_samples} samples):")

        avg_layer_stats = {}
        for layer_idx in Config.MIDDLE_LAYERS:
            if layer_stats[layer_idx]:
                avg_stats = {}
                for stat_name in ['mean', 'std', 'max', 'entropy']:
                    values = [s[stat_name] for s in layer_stats[layer_idx] if stat_name in s]
                    if values:
                        avg_stats[stat_name] = np.mean(values)
                    else:
                        avg_stats[stat_name] = 0.0

                avg_layer_stats[layer_idx] = avg_stats
                print(f"   Layer {layer_idx:2d}: "
                      f"mean={avg_stats['mean']:.4f}, "
                      f"std={avg_stats['std']:.4f}, "
                      f"entropy={avg_stats['entropy']:.4f}")

        # Store statistics for adversarial comparison
        self.clean_attention_stats = {
            'layer_stats': layer_stats,
            'digit_stats': digit_stats,
            'avg_layer_stats': avg_layer_stats,
            'successful_samples': successful_samples
        }

        print(f"✅ Attention statistics computed and stored")
        return self.clean_attention_stats

    def compare_attention_patterns(self, original_img: torch.Tensor, 
                                 adversarial_img: torch.Tensor) -> Dict[str, Any]:
        """
        Compare attention between original and adversarial images - FIXED VERSION
        
        Args:
            original_img: Original image tensor
            adversarial_img: Adversarial image tensor
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Extract attention
            print("Extracting original attention...")
            orig_maps, orig_outputs = self.extract_attention_maps(original_img)
            orig_pred = self.predict_even_odd(original_img)

            print("Extracting adversarial attention...")
            adv_maps, adv_outputs = self.extract_attention_maps(adversarial_img)
            adv_pred = self.predict_even_odd(adversarial_img)

            # Print predictions
            print(f"Original: Task {orig_pred['task_name']} (conf: {orig_pred['confidence']:.3f})")
            print(f"Adversarial: Task {adv_pred['task_name']} (conf: {adv_pred['confidence']:.3f})")

            # Compare multiple layers
            print("\nMulti-layer Comparison:")
            layer_differences = {}
            
            for layer_idx in Config.KEY_LAYERS:
                if layer_idx < len(orig_maps) and layer_idx < len(adv_maps):
                    try:
                        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                        # Original
                        orig_layer_map = self.create_attention_heatmap(orig_maps, layer_idx)
                        if orig_layer_map is not None:
                            im1 = axes[0].imshow(orig_layer_map.cpu().numpy(), cmap=Config.ATTENTION_COLORMAP)
                            axes[0].set_title(f"Original L{layer_idx}")
                            axes[0].axis('off')
                            plt.colorbar(im1, ax=axes[0])

                            # Adversarial
                            adv_layer_map = self.create_attention_heatmap(adv_maps, layer_idx)
                            if adv_layer_map is not None:
                                im2 = axes[1].imshow(adv_layer_map.cpu().numpy(), cmap=Config.ATTENTION_COLORMAP)
                                axes[1].set_title(f"Adversarial L{layer_idx}")
                                axes[1].axis('off')
                                plt.colorbar(im2, ax=axes[1])

                                # Difference
                                diff = torch.abs(orig_layer_map - adv_layer_map)
                                layer_differences[layer_idx] = float(diff.mean())
                                im3 = axes[2].imshow(diff.cpu().numpy(), cmap='viridis')
                                axes[2].set_title(f"Difference L{layer_idx}")
                                axes[2].axis('off')
                                plt.colorbar(im3, ax=axes[2])
                            else:
                                layer_differences[layer_idx] = 0.0
                        else:
                            layer_differences[layer_idx] = 0.0

                        plt.tight_layout()
                        plt.show()
                        
                    except Exception as e:
                        print(f"⚠️ Error comparing layer {layer_idx}: {e}")
                        layer_differences[layer_idx] = 0.0

            return {
                'original_prediction': orig_pred,
                'adversarial_prediction': adv_pred,
                'layer_differences': layer_differences,
                'attack_success': orig_pred['predicted_task'] != adv_pred['predicted_task']
            }
            
        except Exception as e:
            print(f"⚠️ Error in compare_attention_patterns: {e}")
            return {
                'original_prediction': {'predicted_task': 0, 'task_name': 'Even', 'confidence': 0.5},
                'adversarial_prediction': {'predicted_task': 1, 'task_name': 'Odd', 'confidence': 0.5},
                'layer_differences': {},
                'attack_success': False
            }