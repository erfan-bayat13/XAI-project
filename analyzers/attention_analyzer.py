"""
Vision Transformer Attention Analysis for MNIST
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

        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs)

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
        Extract attention maps from MNIST image
        
        Args:
            image_tensor: Input image tensor
            layer_idx: Specific layer to extract (None for all layers)
            
        Returns:
            Tuple of (attention_maps, model_outputs)
        """
        inputs = self.preprocess_mnist_image(image_tensor)

        with torch.no_grad():
            outputs = self.model(inputs)
            attentions = outputs['attentions']

        attention_maps = []

        if layer_idx is not None:
            # Extract specific layer
            if layer_idx < len(attentions):
                attention = attentions[layer_idx][0, 0]  # First batch, first head
                cls_attention = attention[0, 1:]  # CLS token attention to patches
                attention_maps.append(cls_attention)
        else:
            # Extract all layers
            for layer_idx in range(len(attentions)):
                attention = attentions[layer_idx][0, 0]  # First batch, first head
                cls_attention = attention[0, 1:]  # CLS token attention to patches
                attention_maps.append(cls_attention)

        return attention_maps, outputs

    def create_attention_heatmap(self, attention_weights: List[torch.Tensor], 
                               layer_idx: int = 0, head_idx: int = 0) -> Optional[torch.Tensor]:
        """
        Create attention heatmap from attention weights
        
        Args:
            attention_weights: List of attention weight tensors
            layer_idx: Layer index
            head_idx: Attention head index
            
        Returns:
            Attention heatmap tensor or None
        """
        if not attention_weights or layer_idx >= len(attention_weights):
            return None

        attention = attention_weights[layer_idx]
        
        # Reshape to spatial dimensions (14x14 for ViT-base)
        patch_dim = int(np.sqrt(len(attention)))
        attention_map = attention.reshape(patch_dim, patch_dim)

        return attention_map

    def visualize_mnist_attention(self, image_tensor: torch.Tensor, 
                                concept: int, task: int, 
                                save_plot: bool = False) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
        """
        Visualize attention for MNIST digit
        
        Args:
            image_tensor: Input image tensor
            concept: True digit concept
            task: True even/odd task
            save_plot: Whether to save the plot
            
        Returns:
            Tuple of (prediction_results, attention_maps)
        """
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
                attention_map = attention_maps[layer_idx]

                # Reshape to spatial dimensions (14x14)
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

        plt.suptitle(f"Digit {concept} → True: {'Even' if task == 0 else 'Odd'} | "
                    f"Predicted: {prediction['task_name']} ({prediction['confidence']:.3f})",
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_plot:
            plt.savefig(f'mnist_attention_digit_{concept}_task_{task}.png',
                       dpi=150, bbox_inches='tight')

        plt.show()

        return prediction, attention_maps

    def compute_attention_statistics(self, attention_maps: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical features from attention maps
        
        Args:
            attention_maps: List of attention map tensors
            
        Returns:
            Dictionary of statistics per layer
        """
        stats = {}

        # Focus on middle layers where concepts emerge
        for i, layer_idx in enumerate(Config.MIDDLE_LAYERS):
            if layer_idx < len(attention_maps):
                attention_map = attention_maps[layer_idx]
                attention_2d = attention_map.reshape(14, 14)

                stats[f'layer_{layer_idx}'] = {
                    'mean': float(attention_2d.mean()),
                    'std': float(attention_2d.std()),
                    'max': float(attention_2d.max()),
                    'min': float(attention_2d.min()),
                    'entropy': float(-torch.sum(attention_2d * torch.log(attention_2d + 1e-8))),
                    'concentration': float(torch.sum(attention_2d ** 2)),
                    'uniformity': float(1.0 / (attention_2d.std() + 1e-8))
                }

        return stats

    def analyze_attention_statistics(self, dataset: torch.utils.data.Dataset, 
                                   sample_size: int = 50) -> Dict[str, Any]:
        """
        Analyze attention statistics across different digits and layers
        
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

        for idx in indices:
            image, concept, task = dataset[idx]

            try:
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

            except Exception as e:
                print(f"⚠️ Error processing sample {idx}: {e}")
                continue

        # Compute average statistics
        print(f"\n📊 Layer-wise Attention Statistics (averaged over {sample_size} samples):")

        avg_layer_stats = {}
        for layer_idx in Config.MIDDLE_LAYERS:
            if layer_stats[layer_idx]:
                avg_stats = {}
                for stat_name in ['mean', 'std', 'max', 'entropy']:
                    values = [s[stat_name] for s in layer_stats[layer_idx]]
                    avg_stats[stat_name] = np.mean(values)

                avg_layer_stats[layer_idx] = avg_stats
                print(f"   Layer {layer_idx:2d}: "
                      f"mean={avg_stats['mean']:.4f}, "
                      f"std={avg_stats['std']:.4f}, "
                      f"entropy={avg_stats['entropy']:.4f}")

        # Store statistics for adversarial comparison
        self.clean_attention_stats = {
            'layer_stats': layer_stats,
            'digit_stats': digit_stats,
            'avg_layer_stats': avg_layer_stats
        }

        print(f"✅ Attention statistics computed and stored")
        return self.clean_attention_stats

    def compare_attention_patterns(self, original_img: torch.Tensor, 
                                 adversarial_img: torch.Tensor) -> Dict[str, Any]:
        """
        Compare attention between original and adversarial images
        
        Args:
            original_img: Original image tensor
            adversarial_img: Adversarial image tensor
            
        Returns:
            Dictionary with comparison results
        """
        # Convert to PIL-like format for consistent processing
        to_pil = transforms.ToPILImage()
        
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
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                # Original
                orig_layer_map = orig_maps[layer_idx].reshape(14, 14)
                im1 = axes[0].imshow(orig_layer_map.cpu().numpy(), cmap=Config.ATTENTION_COLORMAP)
                axes[0].set_title(f"Original L{layer_idx}")
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0])

                # Adversarial
                adv_layer_map = adv_maps[layer_idx].reshape(14, 14)
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

                plt.tight_layout()
                plt.show()

        return {
            'original_prediction': orig_pred,
            'adversarial_prediction': adv_pred,
            'layer_differences': layer_differences,
            'attack_success': orig_pred['predicted_task'] != adv_pred['predicted_task']
        }