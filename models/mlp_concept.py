"""
MLP for Attention-to-Concept Mapping
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from ..config import Config


class AttentionToConceptMLP(nn.Module):
    """
    MLP that maps ViT attention patterns to digit concepts
    Input: Attention maps from middle layers (6-8)
    Output: Digit logits (0-9)
    """
    
    def __init__(self, 
                 input_size: int = Config.MLP_INPUT_SIZE, 
                 num_concepts: int = Config.MLP_NUM_CONCEPTS):
        """
        Initialize the MLP
        
        Args:
            input_size: Size of input attention features (588 = 3 layers * 196 patches)
            num_concepts: Number of concepts to predict (10 digits)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, Config.MLP_HIDDEN_SIZE_1),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT),
            nn.Linear(Config.MLP_HIDDEN_SIZE_1, Config.MLP_HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT),
            nn.Linear(Config.MLP_HIDDEN_SIZE_2, num_concepts)
        )
        
        print(f"✅ MLP Attention-to-Concept created:")
        print(f"   📐 Input size: {input_size}")
        print(f"   🧠 Hidden sizes: {Config.MLP_HIDDEN_SIZE_1}, {Config.MLP_HIDDEN_SIZE_2}")
        print(f"   🎯 Output concepts: {num_concepts}")

    def forward(self, attention_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP"""
        return self.mlp(attention_features)


def extract_attention_features(vit_model: nn.Module, 
                             image: torch.Tensor,
                             middle_layers: List[int] = Config.MIDDLE_LAYERS,
                             device: torch.device = Config.DEVICE) -> torch.Tensor:
    """
    Extract attention from middle layers and flatten
    
    Args:
        vit_model: The ViT model
        image: Input image tensor
        middle_layers: Which layers to extract attention from
        device: Device to use
        
    Returns:
        Flattened attention features tensor
    """
    with torch.no_grad():
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # The model returns a dictionary, access attentions via key
        outputs = vit_model(image.to(device), output_attentions=True)
        attentions = outputs['attentions']

        # Extract middle layers where concepts emerge
        middle_attentions = []
        for layer_idx in middle_layers:
            if layer_idx < len(attentions):
                attention = attentions[layer_idx][0, 0]  # [seq_len, seq_len]
                cls_attention = attention[0, 1:]  # CLS token attention to patches [196]
                middle_attentions.append(cls_attention)

        # Concatenate all middle layer attentions
        attention_features = torch.cat(middle_attentions, dim=0)  # [588]
        return attention_features.unsqueeze(0)  # [1, 588]


def train_attention_to_concept_mlp(vit_model: nn.Module,
                                  train_dataset: torch.utils.data.Dataset,
                                  epochs: int = Config.MLP_EPOCHS,
                                  learning_rate: float = Config.MLP_LEARNING_RATE,
                                  device: torch.device = Config.DEVICE,
                                  max_samples: int = 2000) -> AttentionToConceptMLP:
    """
    Train MLP to map attention patterns to digit concepts
    
    Args:
        vit_model: The trained ViT model
        train_dataset: Training dataset
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        device: Device to train on
        max_samples: Maximum number of samples to use for training
        
    Returns:
        Trained MLP model
    """
    print("🧠 Training Attention-to-Concept MLP...")

    # Initialize MLP
    mlp = AttentionToConceptMLP().to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Prepare training data
    print("📊 Extracting attention features from training data...")
    attention_features = []
    concept_labels = []

    # Limiting training data size for demo/speed
    num_samples = min(max_samples, len(train_dataset))
    for i in range(num_samples):
        image, concept, task = train_dataset[i]

        # Extract attention features
        features = extract_attention_features(vit_model, image, device=device)
        attention_features.append(features.squeeze(0))
        concept_labels.append(concept)

        if (i + 1) % 200 == 0 or (i + 1) == num_samples:
            print(f"   Processed {i+1}/{num_samples} samples")

    # Convert to tensors
    X = torch.stack(attention_features).to(device)
    y = torch.tensor(concept_labels).to(device)

    print(f"✅ Training data prepared: {X.shape[0]} samples")

    # Training loop
    mlp.train()
    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(X.shape[0])
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        total_loss = 0
        correct = 0

        # Mini-batch training
        batch_size = 64
        num_batches = (X.shape[0] + batch_size - 1) // batch_size

        for i in range(0, X.shape[0], batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]

            # Forward pass
            outputs = mlp(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / X.shape[0]
        # Use num_batches for average loss calculation
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

    print("✅ MLP training complete!")
    mlp.eval()
    return mlp


def detect_adversarial_with_mlp(vit_model: nn.Module,
                               mlp: AttentionToConceptMLP,
                               image: torch.Tensor,
                               true_concept: int,
                               predicted_task: int,
                               device: torch.device = Config.DEVICE) -> Dict[str, Any]:
    """
    Detect adversarial examples using MLP concept predictions
    Core detection logic: If attention predicts different concept than expected, it's adversarial
    
    Args:
        vit_model: The ViT model
        mlp: The trained MLP
        image: Input image tensor
        true_concept: True digit concept
        predicted_task: Predicted even/odd task
        device: Device to use
        
    Returns:
        Dictionary with detection results
    """
    # Extract attention features
    attention_features = extract_attention_features(vit_model, image, device=device)

    # Get MLP concept prediction
    with torch.no_grad():
        logits = mlp(attention_features)
        concept_probs = F.softmax(logits, dim=1)
        predicted_concept = torch.argmax(concept_probs, dim=1).item()
        concept_confidence = torch.max(concept_probs, dim=1)[0].item()

    # Detection logic: Check concept-task consistency
    expected_concept_task = true_concept % 2  # True even/odd for the digit
    predicted_concept_task = predicted_concept % 2  # MLP predicted digit's even/odd

    # If MLP predicts different concept that leads to different task, it's likely adversarial
    # Also consider low confidence as a potential indicator
    is_adversarial = (predicted_concept_task != predicted_task) or (concept_confidence < Config.CONFIDENCE_THRESHOLD)

    return {
        'is_adversarial': is_adversarial,
        'predicted_concept': predicted_concept,
        'concept_confidence': concept_confidence,
        'true_concept': true_concept,
        'concept_task_mismatch': predicted_concept_task != predicted_task,
        'low_confidence': concept_confidence < Config.CONFIDENCE_THRESHOLD,
        'detection_score': 1.0 - concept_confidence if concept_confidence < Config.CONFIDENCE_THRESHOLD else 
                          (1.0 if predicted_concept_task != predicted_task else 0.0)
    }


def evaluate_mlp_concept_prediction(mlp: AttentionToConceptMLP,
                                   vit_model: nn.Module,
                                   test_dataset: torch.utils.data.Dataset,
                                   device: torch.device = Config.DEVICE,
                                   max_samples: int = 500) -> Dict[str, float]:
    """
    Evaluate MLP concept prediction accuracy
    
    Args:
        mlp: Trained MLP model
        vit_model: ViT model for feature extraction
        test_dataset: Test dataset
        device: Device to use
        max_samples: Maximum samples to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("📊 Evaluating MLP concept prediction accuracy...")
    
    mlp.eval()
    correct = 0
    total = 0
    
    num_samples = min(max_samples, len(test_dataset))
    
    with torch.no_grad():
        for i in range(num_samples):
            image, concept, task = test_dataset[i]
            
            # Extract attention features and predict
            attention_features = extract_attention_features(vit_model, image, device=device)
            logits = mlp(attention_features)
            predicted_concept = torch.argmax(logits, dim=1).item()
            
            if predicted_concept == concept:
                correct += 1
            total += 1
            
            if (i + 1) % 100 == 0:
                print(f"   Evaluated {i+1}/{num_samples} samples")
    
    accuracy = 100 * correct / total
    print(f"✅ MLP Concept Prediction Accuracy: {accuracy:.2f}%")
    
    return {
        'concept_accuracy': accuracy,
        'correct_predictions': correct,
        'total_samples': total
    }