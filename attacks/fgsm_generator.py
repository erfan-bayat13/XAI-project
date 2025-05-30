"""
FGSM Attack Generator for MNIST Even-Odd Classification
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ..config import Config


class FGSMAttackGenerator:
    """
    FGSM Attack Generator for MNIST Even-Odd classification
    """

    def __init__(self, model: nn.Module, device: torch.device = Config.DEVICE):
        """
        Initialize FGSM attack generator
        
        Args:
            model: The target model to attack
            device: Device to run attacks on
        """
        self.model = model
        self.device = device
        
        print(f"⚡ FGSM Attack Generator initialized on {device}")

    def generate_fgsm_attack(self, 
                           image: torch.Tensor, 
                           true_task: int, 
                           epsilon: float = Config.DEFAULT_EPSILON,
                           targeted: bool = False, 
                           target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate FGSM adversarial example

        Args:
            image: Input image tensor
            true_task: True task label (0=Even, 1=Odd)
            epsilon: Attack strength
            targeted: Whether to perform targeted attack
            target_class: Target class for targeted attack

        Returns:
            Adversarial image tensor
        """
        # Prepare input
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        image.requires_grad_(True)

        # Set model to evaluation mode but enable gradients
        self.model.eval()

        # Forward pass
        outputs = self.model(image, output_attentions=False)
        logits = outputs['logits']

        # Compute loss
        if targeted and target_class is not None:
            # Targeted attack: minimize loss for target class
            target_tensor = torch.tensor([target_class]).to(self.device)
            loss = -nn.CrossEntropyLoss()(logits, target_tensor)
        else:
            # Untargeted attack: maximize loss for true class
            true_task_tensor = torch.tensor([true_task]).to(self.device)
            loss = nn.CrossEntropyLoss()(logits, true_task_tensor)

        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()

        # Generate adversarial example
        data_grad = image.grad.data

        # FGSM attack
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad

        # Clamp to maintain valid pixel values
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image.detach()

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
            # Original prediction
            orig_input = original_image.unsqueeze(0).to(self.device) if len(original_image.shape) == 3 else original_image.to(self.device)
            orig_outputs = self.model(orig_input)
            orig_pred = torch.argmax(orig_outputs['logits'], dim=1).item()
            orig_confidence = torch.softmax(orig_outputs['logits'], dim=1).max().item()

            # Adversarial prediction
            adv_input = adversarial_image.to(self.device) if len(adversarial_image.shape) == 4 else adversarial_image.unsqueeze(0).to(self.device)
            adv_outputs = self.model(adv_input)
            adv_pred = torch.argmax(adv_outputs['logits'], dim=1).item()
            adv_confidence = torch.softmax(adv_outputs['logits'], dim=1).max().item()

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
                             epsilon: float = Config.DEFAULT_EPSILON) -> torch.Tensor:
        """
        Generate FGSM attacks for a batch of images
        
        Args:
            images: Batch of input images
            tasks: Batch of true task labels
            epsilon: Attack strength
            
        Returns:
            Batch of adversarial images
        """
        images = images.to(self.device)
        tasks = tasks.to(self.device)
        images.requires_grad_(True)

        # Set model to evaluation mode but enable gradients
        self.model.eval()

        # Forward pass
        outputs = self.model(images, output_attentions=False)
        logits = outputs['logits']

        # Compute loss
        loss = nn.CrossEntropyLoss()(logits, tasks)

        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()

        # Generate adversarial examples
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_images = images + epsilon * sign_data_grad

        # Clamp to maintain valid pixel values
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        return perturbed_images.detach()

    def evaluate_attack_success_rate(self, 
                                   dataset: torch.utils.data.Dataset,
                                   epsilon: float = Config.DEFAULT_EPSILON,
                                   num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate attack success rate on a dataset
        
        Args:
            dataset: Dataset to evaluate on
            epsilon: Attack strength
            num_samples: Number of samples to test
            
        Returns:
            Dictionary with success rate metrics
        """
        print(f"🎯 Evaluating FGSM attack success rate (ε={epsilon})")
        
        successful_attacks = 0
        total_attacks = 0
        correctly_classified = 0
        
        self.model.eval()
        
        num_samples = min(num_samples, len(dataset))
        
        for i in range(num_samples):
            try:
                image, concept, task = dataset[i]
                
                # First check if model classifies correctly
                with torch.no_grad():
                    clean_input = image.unsqueeze(0).to(self.device)
                    clean_outputs = self.model(clean_input)
                    clean_pred = torch.argmax(clean_outputs['logits'], dim=1).item()
                    
                    if clean_pred == task:
                        correctly_classified += 1
                        
                        # Generate adversarial example
                        adv_image = self.generate_fgsm_attack(image, task, epsilon)
                        
                        # Test if attack succeeded
                        attack_result = self.test_attack_success(image, adv_image, task)
                        
                        if attack_result['attack_success']:
                            successful_attacks += 1
                        
                        total_attacks += 1
            
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
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
            'samples_tested': num_samples
        }

    def create_adversarial_dataset(self, 
                                 dataset: torch.utils.data.Dataset,
                                 epsilon: float = Config.DEFAULT_EPSILON,
                                 max_samples: int = 1000) -> Dict[str, Any]:
        """
        Create a dataset of adversarial examples
        
        Args:
            dataset: Original dataset
            epsilon: Attack strength
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary containing adversarial examples and metadata
        """
        print(f"🏭 Creating adversarial dataset (ε={epsilon})")
        
        adversarial_examples = []
        attack_metadata = []
        
        self.model.eval()
        
        num_samples = min(max_samples, len(dataset))
        successful_count = 0
        
        for i in range(num_samples):
            try:
                image, concept, task = dataset[i]
                
                # Generate adversarial example
                adv_image = self.generate_fgsm_attack(image, task, epsilon)
                
                # Test attack success
                attack_result = self.test_attack_success(image, adv_image, task)
                
                adversarial_examples.append({
                    'original_image': image,
                    'adversarial_image': adv_image.squeeze(0),
                    'concept': concept,
                    'task': task,
                    'attack_result': attack_result
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
        
        return {
            'adversarial_examples': adversarial_examples,
            'attack_metadata': attack_metadata,
            'success_rate': success_rate,
            'epsilon': epsilon,
            'total_samples': num_samples,
            'successful_attacks': successful_count
        }

    def generate_multi_epsilon_attacks(self,
                                     image: torch.Tensor,
                                     true_task: int,
                                     epsilon_values: List[float] = None) -> Dict[float, torch.Tensor]:
        """
        Generate adversarial examples with multiple epsilon values
        
        Args:
            image: Input image tensor
            true_task: True task label
            epsilon_values: List of epsilon values to test
            
        Returns:
            Dictionary mapping epsilon to adversarial image
        """
        if epsilon_values is None:
            epsilon_values = Config.FGSM_EPSILON_VALUES
        
        adversarial_images = {}
        
        for epsilon in epsilon_values:
            try:
                adv_image = self.generate_fgsm_attack(image, true_task, epsilon)
                adversarial_images[epsilon] = adv_image
            except Exception as e:
                print(f"⚠️ Error generating attack with ε={epsilon}: {e}")
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
            'perturbation_percentage': (mean_abs_diff / torch.mean(original_image).item()) * 100
        }