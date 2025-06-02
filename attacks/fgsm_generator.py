"""
FGSM Attack Generator using ART (Adversarial Robustness Toolbox) for MNIST Even-Odd Classification
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from config import Config

# ART imports
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier


class FGSMAttackGenerator:
    """
    FGSM Attack Generator using ART for MNIST Even-Odd classification
    """

    def __init__(self, model: nn.Module, device: torch.device = Config.DEVICE):
        """
        Initialize FGSM attack generator with ART integration
        
        Args:
            model: The target model to attack
            device: Device to run attacks on
        """
        self.model = model
        self.device = device
        
        # Setup ART classifier wrapper
        self.art_classifier = self._setup_art_classifier()
        
        print(f"⚡ FGSM Attack Generator (ART) initialized on {device}")

    def _setup_art_classifier(self) -> PyTorchClassifier:
        """
        Setup ART PyTorchClassifier wrapper for the model
        
        Returns:
            Configured ART PyTorchClassifier
        """
        # Create a simple loss function for ART
        criterion = nn.CrossEntropyLoss()
        
        # Create a dummy optimizer (ART requires it but we won't use it for attacks)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Wrap model with ART classifier
        art_classifier = PyTorchClassifier(
            model=self.model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE),  # 3 channels for RGB converted MNIST
            nb_classes=2,  # Even/Odd classification
            clip_values=(0.0, 1.0),  # Pixel value range after normalization
            device_type="gpu" if torch.cuda.is_available() else "cpu"
        )
        
        return art_classifier

    def generate_fgsm_attack(self, 
                           image: torch.Tensor, 
                           true_task: int, 
                           epsilon: float = Config.DEFAULT_EPSILON,
                           targeted: bool = False, 
                           target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate FGSM adversarial example using ART

        Args:
            image: Input image tensor
            true_task: True task label (0=Even, 1=Odd)
            epsilon: Attack strength
            targeted: Whether to perform targeted attack
            target_class: Target class for targeted attack

        Returns:
            Adversarial image tensor
        """
        # Prepare input for ART (expects numpy array)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Convert to numpy for ART
        image_np = image.cpu().numpy()
        
        # Create FGSM attack
        if targeted and target_class is not None:
            attack = FastGradientMethod(
                estimator=self.art_classifier, 
                eps=epsilon,
                targeted=True
            )
            # Create target labels array
            target_labels = np.array([target_class])
            adv_image_np = attack.generate(x=image_np, y=target_labels)
        else:
            attack = FastGradientMethod(
                estimator=self.art_classifier, 
                eps=epsilon,
                targeted=False
            )
            adv_image_np = attack.generate(x=image_np)
        
        # Convert back to torch tensor
        adv_image = torch.from_numpy(adv_image_np).to(self.device)
        
        return adv_image

    def generate_pgd_attack(self,
                          image: torch.Tensor,
                          true_task: int,
                          epsilon: float = Config.DEFAULT_EPSILON,
                          max_iter: int = 10,
                          eps_step: float = None) -> torch.Tensor:
        """
        Generate PGD (Projected Gradient Descent) adversarial example using ART
        
        Args:
            image: Input image tensor
            true_task: True task label
            epsilon: Attack strength (L-infinity norm bound)
            max_iter: Maximum number of iterations
            eps_step: Step size for each iteration
            
        Returns:
            Adversarial image tensor
        """
        # Prepare input for ART
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image_np = image.cpu().numpy()
        
        # Set default step size if not provided
        if eps_step is None:
            eps_step = epsilon / 4
        
        # Create PGD attack
        attack = ProjectedGradientDescent(
            estimator=self.art_classifier,
            norm=np.inf,  # L-infinity norm
            eps=epsilon,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=False
        )
        
        adv_image_np = attack.generate(x=image_np)
        adv_image = torch.from_numpy(adv_image_np).to(self.device)
        
        return adv_image

    def _safe_model_forward(self, inputs: torch.Tensor, need_attentions: bool = False) -> torch.Tensor:
        """
        Safe forward pass that works with both ARTCompatibleWrapper and MNISTViTWrapper
        
        Args:
            inputs: Input tensor
            need_attentions: Whether attentions are needed (ignored for ART wrapper)
            
        Returns:
            Logits tensor
        """
        try:
            # Try with output_attentions parameter first (for MNISTViTWrapper)
            if hasattr(self.model, 'vit_wrapper'):
                # This is ARTCompatibleWrapper - use simple forward
                return self.model(inputs)
            else:
                # This might be MNISTViTWrapper directly - try with parameters
                try:
                    return self.model(inputs, output_attentions=need_attentions)
                except TypeError:
                    # If that fails, try simple forward
                    return self.model(inputs)
        except Exception as e:
            print(f"⚠️ Error in model forward pass: {e}")
            # Fallback: try simple forward call
            return self.model(inputs)

    def test_attack_success(self, 
                          original_image: torch.Tensor, 
                          adversarial_image: torch.Tensor, 
                          true_task: int) -> Dict[str, Any]:
        """
        Test if the attack was successful
        
        Args:
            original_image: Original clean image
            adversarial_image: Generated adversarial image
            true_task: True task label
            
        Returns:
            Dictionary with attack results
        """
        with torch.no_grad():
            # Ensure both images have batch dimension
            if len(original_image.shape) == 3:
                original_image = original_image.unsqueeze(0)
            if len(adversarial_image.shape) == 3:
                adversarial_image = adversarial_image.unsqueeze(0)
            
            # Original prediction - use safe forward method
            orig_outputs = self._safe_model_forward(original_image.to(self.device))
            orig_logits = orig_outputs['logits'] if isinstance(orig_outputs, dict) else orig_outputs
            orig_pred = torch.argmax(orig_logits, dim=1).item()
            orig_confidence = torch.softmax(orig_logits, dim=1).max().item()

            # Adversarial prediction - use safe forward method
            adv_outputs = self._safe_model_forward(adversarial_image.to(self.device))
            adv_logits = adv_outputs['logits'] if isinstance(adv_outputs, dict) else adv_outputs
            adv_pred = torch.argmax(adv_logits, dim=1).item()
            adv_confidence = torch.softmax(adv_logits, dim=1).max().item()

            # Attack successful if prediction changed
            attack_success = (orig_pred != adv_pred)

            return {
                'attack_success': attack_success,
                'original_prediction': orig_pred,
                'adversarial_prediction': adv_pred,
                'original_confidence': orig_confidence,
                'adversarial_confidence': adv_confidence,
                'true_task': true_task,
                'original_correct': orig_pred == true_task,
                'adversarial_correct': adv_pred == true_task,
                'confidence_drop': orig_confidence - adv_confidence
            }

    def generate_batch_attacks(self, 
                             images: torch.Tensor, 
                             tasks: torch.Tensor,
                             epsilon: float = Config.DEFAULT_EPSILON,
                             attack_type: str = "fgsm") -> torch.Tensor:
        """
        Generate attacks for a batch of images using ART
        
        Args:
            images: Batch of input images
            tasks: Batch of true task labels
            epsilon: Attack strength
            attack_type: Type of attack ("fgsm" or "pgd")
            
        Returns:
            Batch of adversarial images
        """
        # Convert to numpy for ART
        images_np = images.cpu().numpy()
        
        if attack_type.lower() == "fgsm":
            attack = FastGradientMethod(
                estimator=self.art_classifier, 
                eps=epsilon,
                targeted=False
            )
        elif attack_type.lower() == "pgd":
            attack = ProjectedGradientDescent(
                estimator=self.art_classifier,
                norm=np.inf,
                eps=epsilon,
                eps_step=epsilon / 4,
                max_iter=10,
                targeted=False
            )
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        # Generate adversarial examples
        adversarial_images_np = attack.generate(x=images_np)
        
        # Convert back to torch
        adversarial_images = torch.from_numpy(adversarial_images_np)
        
        return adversarial_images

    def evaluate_attack_success_rate(self, 
                                   dataset: torch.utils.data.Dataset,
                                   epsilon: float = Config.DEFAULT_EPSILON,
                                   num_samples: int = 100,
                                   attack_type: str = "fgsm") -> Dict[str, float]:
        """
        Evaluate attack success rate on a dataset using ART
        
        Args:
            dataset: Dataset to evaluate on
            epsilon: Attack strength
            num_samples: Number of samples to test
            attack_type: Type of attack to use
            
        Returns:
            Dictionary with success rate metrics
        """
        print(f"🎯 Evaluating {attack_type.upper()} attack success rate (ε={epsilon})")
        
        successful_attacks = 0
        total_attacks = 0
        correctly_classified = 0
        
        self.model.eval()
        
        num_samples = min(num_samples, len(dataset))
        
        # Collect batch of samples for efficient processing
        batch_images = []
        batch_tasks = []
        sample_indices = []
        
        for i in range(num_samples):
            try:
                image, concept, task = dataset[i]
                
                # First check if model classifies correctly
                with torch.no_grad():
                    clean_input = image.unsqueeze(0).to(self.device)
                    clean_outputs = self._safe_model_forward(clean_input)
                    clean_logits = clean_outputs['logits'] if isinstance(clean_outputs, dict) else clean_outputs
                    clean_pred = torch.argmax(clean_logits, dim=1).item()
                    
                    if clean_pred == task:
                        correctly_classified += 1
                        batch_images.append(image)
                        batch_tasks.append(task)
                        sample_indices.append(i)
            
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                continue
        
        if len(batch_images) == 0:
            print("❌ No correctly classified samples found")
            return {'attack_success_rate': 0, 'clean_accuracy': 0}
        
        # Convert to batch tensors
        batch_images_tensor = torch.stack(batch_images)
        batch_tasks_tensor = torch.tensor(batch_tasks)
        
        # Generate adversarial examples for the batch
        try:
            adv_batch = self.generate_batch_attacks(
                batch_images_tensor, batch_tasks_tensor, epsilon, attack_type
            )
            
            # Test each adversarial example
            for i, (orig_img, adv_img, task) in enumerate(zip(batch_images, adv_batch, batch_tasks)):
                attack_result = self.test_attack_success(orig_img, adv_img, task)
                
                if attack_result['attack_success']:
                    successful_attacks += 1
                
                total_attacks += 1
        
        except Exception as e:
            print(f"⚠️ Error in batch attack generation: {e}")
            # Fallback to individual processing
            for i, (image, task) in enumerate(zip(batch_images, batch_tasks)):
                try:
                    if attack_type.lower() == "fgsm":
                        adv_image = self.generate_fgsm_attack(image, task, epsilon)
                    elif attack_type.lower() == "pgd":
                        adv_image = self.generate_pgd_attack(image, task, epsilon)
                    else:
                        continue
                    
                    attack_result = self.test_attack_success(image, adv_image, task)
                    
                    if attack_result['attack_success']:
                        successful_attacks += 1
                    
                    total_attacks += 1
                    
                except Exception as inner_e:
                    print(f"⚠️ Error in individual attack {i}: {inner_e}")
                    continue
        
        success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0
        clean_accuracy = correctly_classified / num_samples
        
        print(f"   Clean Accuracy: {clean_accuracy:.3f}")
        print(f"   Attack Success Rate: {success_rate:.3f}")
        print(f"   Samples: {total_attacks}/{num_samples}")
        
        return {
            'attack_success_rate': success_rate,
            'clean_accuracy': clean_accuracy,
            'successful_attacks': successful_attacks,
            'total_attacks': total_attacks,
            'samples_tested': num_samples,
            'attack_type': attack_type
        }

    def create_adversarial_dataset(self, 
                                 dataset: torch.utils.data.Dataset,
                                 epsilon: float = Config.DEFAULT_EPSILON,
                                 max_samples: int = 1000,
                                 attack_type: str = "fgsm") -> Dict[str, Any]:
        """
        Create a dataset of adversarial examples using ART
        
        Args:
            dataset: Original dataset
            epsilon: Attack strength
            max_samples: Maximum number of samples to process
            attack_type: Type of attack to use
            
        Returns:
            Dictionary containing adversarial examples and metadata
        """
        print(f"🏭 Creating adversarial dataset using {attack_type.upper()} (ε={epsilon})")
        
        adversarial_examples = []
        attack_metadata = []
        
        self.model.eval()
        
        num_samples = min(max_samples, len(dataset))
        successful_count = 0
        
        for i in range(num_samples):
            try:
                image, concept, task = dataset[i]
                
                # Generate adversarial example
                if attack_type.lower() == "fgsm":
                    adv_image = self.generate_fgsm_attack(image, task, epsilon)
                elif attack_type.lower() == "pgd":
                    adv_image = self.generate_pgd_attack(image, task, epsilon)
                else:
                    print(f"⚠️ Unsupported attack type: {attack_type}")
                    continue
                
                # Test attack success
                attack_result = self.test_attack_success(image, adv_image, task)
                
                adversarial_examples.append({
                    'original_image': image,
                    'adversarial_image': adv_image.squeeze(0) if len(adv_image.shape) == 4 else adv_image,
                    'concept': concept,
                    'task': task,
                    'attack_result': attack_result,
                    'attack_type': attack_type
                })
                
                attack_metadata.append(attack_result)
                
                if attack_result['attack_success']:
                    successful_count += 1
                
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i+1}/{num_samples} samples")
                    
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                continue
        
        success_rate = successful_count / num_samples if num_samples > 0 else 0
        print(f"✅ Adversarial dataset created:")
        print(f"   Total samples: {num_samples}")
        print(f"   Successful attacks: {successful_count}")
        print(f"   Success rate: {success_rate:.3f}")
        print(f"   Attack type: {attack_type.upper()}")
        
        return {
            'adversarial_examples': adversarial_examples,
            'attack_metadata': attack_metadata,
            'success_rate': success_rate,
            'epsilon': epsilon,
            'total_samples': num_samples,
            'successful_attacks': successful_count,
            'attack_type': attack_type
        }

    def generate_multi_epsilon_attacks(self,
                                     image: torch.Tensor,
                                     true_task: int,
                                     epsilon_values: List[float] = None,
                                     attack_type: str = "fgsm") -> Dict[float, torch.Tensor]:
        """
        Generate adversarial examples with multiple epsilon values using ART
        
        Args:
            image: Input image tensor
            true_task: True task label
            epsilon_values: List of epsilon values to test
            attack_type: Type of attack to use
            
        Returns:
            Dictionary mapping epsilon to adversarial image
        """
        if epsilon_values is None:
            epsilon_values = Config.FGSM_EPSILON_VALUES
        
        adversarial_images = {}
        
        for epsilon in epsilon_values:
            try:
                if attack_type.lower() == "fgsm":
                    adv_image = self.generate_fgsm_attack(image, true_task, epsilon)
                elif attack_type.lower() == "pgd":
                    adv_image = self.generate_pgd_attack(image, true_task, epsilon)
                else:
                    print(f"⚠️ Unsupported attack type: {attack_type}")
                    continue
                    
                adversarial_images[epsilon] = adv_image
            except Exception as e:
                print(f"⚠️ Error generating {attack_type} attack with ε={epsilon}: {e}")
                continue
        
        return adversarial_images

    def analyze_perturbation_magnitude(self,
                                     original_image: torch.Tensor,
                                     adversarial_image: torch.Tensor) -> Dict[str, float]:
        """
        Analyze the magnitude of adversarial perturbation
        
        Args:
            original_image: Original clean image
            adversarial_image: Adversarial image
            
        Returns:
            Dictionary with perturbation statistics
        """
        # Ensure same shape and device
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        if len(adversarial_image.shape) == 3:
            adversarial_image = adversarial_image.unsqueeze(0)
        
        original_image = original_image.to(self.device)
        adversarial_image = adversarial_image.to(self.device)
        
        # Compute perturbation
        perturbation = adversarial_image - original_image
        
        # Compute statistics
        l2_norm = torch.norm(perturbation, p=2).item()
        l_inf_norm = torch.norm(perturbation, p=float('inf')).item()
        l1_norm = torch.norm(perturbation, p=1).item()
        mean_abs_diff = torch.mean(torch.abs(perturbation)).item()
        max_abs_diff = torch.max(torch.abs(perturbation)).item()
        
        return {
            'l2_norm': l2_norm,
            'l_inf_norm': l_inf_norm,
            'l1_norm': l1_norm,
            'mean_absolute_difference': mean_abs_diff,
            'max_absolute_difference': max_abs_diff,
            'perturbation_percentage': (mean_abs_diff / torch.mean(original_image).item()) * 100 if torch.mean(original_image).item() != 0 else 0
        }

    def get_attack_types(self) -> List[str]:
        """
        Get list of available attack types
        
        Returns:
            List of attack type names
        """
        return ["fgsm", "pgd"]