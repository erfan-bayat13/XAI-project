"""
MLP for Attention-to-Concept Mapping
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from config import Config


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
            # Input normalization
            nn.BatchNorm1d(input_size),

            # First layer
            nn.Linear(input_size, Config.MLP_HIDDEN_SIZE_1),
            nn.BatchNorm1d(Config.MLP_HIDDEN_SIZE_1),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT),

            # Second layer
            nn.Linear(Config.MLP_HIDDEN_SIZE_1, Config.MLP_HIDDEN_SIZE_2),
            nn.BatchNorm1d(Config.MLP_HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT),
            # Output layer
            nn.Linear(Config.MLP_HIDDEN_SIZE_2, num_concepts)
        )
        self.apply(self._init_weights)
        
        print(f"✅ MLP Attention-to-Concept created:")
        print(f"   📐 Input size: {input_size}")
        print(f"   🧠 Hidden sizes: {Config.MLP_HIDDEN_SIZE_1}, {Config.MLP_HIDDEN_SIZE_2}")
        print(f"   🎯 Output concepts: {num_concepts}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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

        # CRITICAL FIX: Make sure VIT model is in eval mode
        vit_model.eval()
        
        outputs = vit_model(image.to(device), output_attentions=True)
        attentions = outputs['attentions']

        # Extract middle layers where concepts emerge
        middle_attentions = []
        for layer_idx in middle_layers:
            if layer_idx < len(attentions):
                attention = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
                attention_avg = attention.mean(dim=0)  # Average across heads
                
                # Extract multiple types of features:
                # 1. CLS attention to patches
                cls_attention = attention_avg[0, 1:]  # [196]
                
                # 2. Attention statistics
                attention_flat = attention_avg.flatten()
                
                # FIX: Add small epsilon to prevent log(0)
                attention_flat = attention_flat + 1e-8
                # FIX: Normalize attention to make it a proper probability distribution
                attention_flat = attention_flat / attention_flat.sum()
                
                entropy = -torch.sum(attention_flat * torch.log(attention_flat))
                concentration = torch.sum(attention_flat ** 2)
                max_attention = torch.max(attention_flat)
                
                # 3. Spatial patterns
                spatial_attention = cls_attention.reshape(14, 14)
                spatial_std = spatial_attention.std()
                spatial_mean = spatial_attention.mean()
                
                # FIX: Normalize all features to similar scales
                # CLS attention is already in [0,1] range after softmax
                # Normalize statistics to [0,1] range
                entropy_norm = entropy / torch.log(torch.tensor(196.0))  # Max possible entropy
                concentration_norm = concentration  # Already in reasonable range
                max_attention_norm = max_attention  # Already in [0,1]
                spatial_std_norm = spatial_std / spatial_attention.max()  # Normalize by max
                spatial_mean_norm = spatial_mean  # Already normalized
                
                # Combine all features
                layer_features = torch.cat([
                    cls_attention,  # [196] - spatial attention
                    torch.tensor([entropy_norm, concentration_norm, max_attention_norm, 
                                spatial_std_norm, spatial_mean_norm], 
                               device=device, dtype=cls_attention.dtype)  # [5] - statistics
                ])
                middle_attentions.append(layer_features)

        # Concatenate all middle layer attentions
        if middle_attentions:
            attention_features = torch.cat(middle_attentions, dim=0)  # [603]
            return attention_features.unsqueeze(0)  # [1, 603]
        else:
            # Fallback if no attention found
            return torch.zeros(1, Config.MLP_INPUT_SIZE, device=device)


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

    # FIX: Ensure ViT model is in eval mode and on correct device
    vit_model.eval()
    vit_model.to(device)

    # Initialize MLP
    mlp = AttentionToConceptMLP().to(device)
    
    # FIX: Use different learning rate and optimizer settings
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # FIX: Use label smoothing to help with learning
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # FIX: Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Prepare training data
    print("📊 Extracting attention features from training data...")
    attention_features = []
    concept_labels = []

    # FIX: Use smaller sample size initially to debug
    num_samples = min(max_samples, len(train_dataset))
    print(f"Processing {num_samples} samples...")
    
    for i in range(num_samples):
        try:
            image, concept, task = train_dataset[i]

            # Extract attention features
            features = extract_attention_features(vit_model, image, device=device)
            
            # FIX: Check for valid features
            if features is not None and not torch.isnan(features).any():
                attention_features.append(features.squeeze(0))
                concept_labels.append(concept)
            else:
                print(f"⚠️ Invalid features at sample {i}")

            if (i + 1) % 200 == 0 or (i + 1) == num_samples:
                print(f"   Processed {i+1}/{num_samples} samples")
                
        except Exception as e:
            print(f"⚠️ Error processing sample {i}: {e}")
            continue

    if len(attention_features) == 0:
        raise ValueError("No valid features extracted! Check ViT model and feature extraction.")

    # Convert to tensors
    X = torch.stack(attention_features).to(device)
    y = torch.tensor(concept_labels, dtype=torch.long).to(device)

    print(f"✅ Training data prepared: {X.shape[0]} samples, feature shape: {X.shape[1]}")
    print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Label distribution: {torch.bincount(y)}")

    # FIX: Check for NaN or infinite values
    if torch.isnan(X).any() or torch.isinf(X).any():
        print("⚠️ NaN or Inf values in features, cleaning...")
        X = torch.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    # Training loop
    mlp.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(X.shape[0])
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        total_loss = 0
        correct = 0

        # Mini-batch training
        batch_size = 32  # FIX: Smaller batch size
        num_batches = (X.shape[0] + batch_size - 1) // batch_size

        for i in range(0, X.shape[0], batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]

            # Forward pass
            outputs = mlp(batch_X)
            loss = criterion(outputs, batch_y)

            # FIX: Check for NaN loss
            if torch.isnan(loss):
                print(f"⚠️ NaN loss at epoch {epoch}, batch {i//batch_size}")
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # FIX: Gradient clipping
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / X.shape[0]
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Update learning rate
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, LR={current_lr:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
        elif epoch > 10 and accuracy < 15:  # If stuck at random performance
            print("⚠️ Training seems stuck, stopping early")
            break

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