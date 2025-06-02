"""
Vision Transformer Wrapper for MNIST Even-Odd Classification
Updated for ART (Adversarial Robustness Toolbox) compatibility
"""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from typing import Dict, Any, Optional, Union
from config import Config


class MNISTViTWrapper(nn.Module):
    """
    Wrapper around pre-trained ViT for MNIST Even-Odd classification
    Compatible with ART (Adversarial Robustness Toolbox)
    """

    def __init__(self, model_name: str = Config.VIT_MODEL_NAME, num_classes: int = 2):
        """
        Initialize the ViT wrapper
        
        Args:
            model_name: Name of the pre-trained ViT model
            num_classes: Number of output classes (2 for Even/Odd)
        """
        super(MNISTViTWrapper, self).__init__()

        # Load pre-trained ViT (frozen for feature extraction)
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True
        )

        # Freeze ViT parameters (we only want attention patterns, not retraining)
        for param in self.vit.parameters():
            param.requires_grad = False

        # Get the hidden size
        hidden_size = self.vit.config.hidden_size  # 768 for ViT-base

        # New classification head for Even/Odd
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self._print_info(model_name, num_classes, hidden_size)

    def _print_info(self, model_name: str, num_classes: int, hidden_size: int):
        """Print model information"""
        print(f"✅ MNIST-ViT Wrapper created:")
        print(f"   🧠 Backbone: {model_name}")
        print(f"   🔒 ViT parameters: Frozen")
        print(f"   🎯 Task: {num_classes}-class classification (Even/Odd)")
        print(f"   📐 Hidden size: {hidden_size}")
        print(f"   🛡️ ART Compatible: True")

    def forward(self, pixel_values: torch.Tensor, 
               output_attentions: bool = True,
               return_dict: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the model
        Compatible with both ART (expects logits) and analysis (expects dict)
        
        Args:
            pixel_values: Input image tensor
            output_attentions: Whether to output attention weights
            return_dict: Whether to return dictionary (for analysis) or tensor (for ART)
            
        Returns:
            Either logits tensor (for ART) or dictionary with detailed outputs (for analysis)
        """
        # Get ViT features and attention
        vit_outputs = self.vit.vit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True
        )

        # Get [CLS] token representation
        last_hidden_state = vit_outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0]  # [CLS] token is at position 0

        # Classification
        logits = self.classifier(cls_token)

        if return_dict:
            # Return dictionary for analysis code
            return {
                'logits': logits,
                'attentions': vit_outputs.attentions if output_attentions else None,
                'hidden_states': vit_outputs.hidden_states,
                'cls_token': cls_token
            }
        else:
            # Return logits tensor for ART compatibility
            return logits

    def freeze_vit(self):
        """Freeze ViT parameters"""
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_vit(self):
        """Unfreeze ViT parameters"""
        for param in self.vit.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """Get trainable parameters (only classifier head)"""
        return [param for param in self.classifier.parameters() if param.requires_grad]

    def save_model(self, path: str = Config.MODEL_SAVE_PATH):
        """Save the model state dict"""
        torch.save(self.state_dict(), path)
        print(f"💾 Model saved as '{path}'")

    def load_model(self, path: str = Config.MODEL_SAVE_PATH):
        """Load the model state dict"""
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"✅ Loaded model from '{path}'")
        
    def train_mode(self):
        """Set model to training mode and ensure gradients for classifier"""
        self.train()
        # Ensure ViT stays frozen
        self.freeze_vit()
        # Enable gradients for classifier
        for param in self.classifier.parameters():
            param.requires_grad = True

    def eval_mode(self):
        """Set model to evaluation mode"""
        self.eval()
        # Note: Don't disable gradients here as ART needs them for attacks


class ARTCompatibleWrapper(nn.Module):
    """
    Additional wrapper specifically for ART that ensures clean forward pass
    """
    
    def __init__(self, vit_wrapper: MNISTViTWrapper):
        """
        Initialize ART-compatible wrapper
        
        Args:
            vit_wrapper: The MNISTViTWrapper instance
        """
        super(ARTCompatibleWrapper, self).__init__()
        self.vit_wrapper = vit_wrapper
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Clean forward pass that returns only logits (required by ART)
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments (ignored for ART compatibility)
            
        Returns:
            Logits tensor
        """
        # Ensure we get only logits, not a dictionary
        # Always use output_attentions=False and return_dict=False for ART
        output = self.vit_wrapper(x, output_attentions=False, return_dict=False)
        return output

    def train(self, mode: bool = True):
        """Override train method to properly handle training mode"""
        super().train(mode)
        self.vit_wrapper.train(mode)
        return self

    def eval(self):
        """Override eval method to properly handle evaluation mode"""
        super().eval()
        self.vit_wrapper.eval()
        return self


def train_even_odd_classifier(model: MNISTViTWrapper, 
                             train_loader: torch.utils.data.DataLoader, 
                             epochs: int = Config.EVEN_ODD_EPOCHS,
                             device: torch.device = Config.DEVICE) -> Dict[str, float]:
    """
    Train the Even/Odd classification head
    
    Args:
        model: The ViT wrapper model
        train_loader: Training data loader
        epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        Training results dictionary
    """
    print("🏋️ Training Even/Odd classifier...")

    # Set model to training mode
    model.train_mode()
    model.to(device)

    # Only train the classifier head (ViT backbone is frozen)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, concepts, tasks) in enumerate(train_loader):
            images, tasks = images.to(device), tasks.to(device)

            # Forward pass (get dictionary output for training)
            outputs = model(images, output_attentions=False, return_dict=True)
            loss = criterion(outputs['logits'], tasks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs['logits'].data, 1)
            total += tasks.size(0)
            correct += (predicted == tasks).sum().item()

            if batch_idx % 50 == 0:
                print(f'   Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%')

        epoch_acc = 100 * correct / total
        epoch_loss = total_loss / len(train_loader)
        print(f'✅ Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    # Set model back to eval mode
    model.eval_mode()
    print(f"🎉 Training complete!")

    # Save the trained model
    model.save_model()
    
    return {
        'final_loss': epoch_loss,
        'final_accuracy': epoch_acc,
        'epochs_completed': epochs
    }


def evaluate_model(model: MNISTViTWrapper,
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device = Config.DEVICE) -> Dict[str, float]:
    """
    Evaluate the model on test data
    
    Args:
        model: The ViT wrapper model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("📊 Evaluating model on test set...")
    
    model.eval_mode()
    model.to(device)
    
    correct = 0
    total = 0
    concept_correct = {}  # Track accuracy per digit
    
    for digit in range(10):
        concept_correct[digit] = {'correct': 0, 'total': 0}
    
    with torch.no_grad():
        for images, concepts, tasks in test_loader:
            images, tasks = images.to(device), tasks.to(device)
            
            # Get predictions (use return_dict=True for consistency)
            outputs = model(images, output_attentions=False, return_dict=True)
            _, predicted = torch.max(outputs['logits'].data, 1)
            
            total += tasks.size(0)
            correct += (predicted == tasks).sum().item()
            
            # Per-concept accuracy
            for i in range(len(concepts)):
                concept = concepts[i].item()
                concept_correct[concept]['total'] += 1
                if predicted[i] == tasks[i]:
                    concept_correct[concept]['correct'] += 1
    
    overall_accuracy = 100 * correct / total
    print(f"📊 Overall Test Accuracy: {overall_accuracy:.2f}%")
    
    print(f"\n📈 Per-Digit Accuracy:")
    per_digit_accuracy = {}
    for digit in range(10):
        if concept_correct[digit]['total'] > 0:
            acc = 100 * concept_correct[digit]['correct'] / concept_correct[digit]['total']
            per_digit_accuracy[digit] = acc
            expected_task = "Even" if digit % 2 == 0 else "Odd"
            print(f"   Digit {digit} ({expected_task}): {acc:.1f}% "
                  f"({concept_correct[digit]['correct']}/{concept_correct[digit]['total']})")
    
    return {
        'overall_accuracy': overall_accuracy,
        'per_digit_accuracy': per_digit_accuracy,
        'total_samples': total
    }


def create_art_compatible_model(vit_wrapper: MNISTViTWrapper) -> ARTCompatibleWrapper:
    """
    Create an ART-compatible version of the model
    
    Args:
        vit_wrapper: The trained ViT wrapper
        
    Returns:
        ART-compatible wrapper
    """
    art_model = ARTCompatibleWrapper(vit_wrapper)
    art_model.eval()
    return art_model