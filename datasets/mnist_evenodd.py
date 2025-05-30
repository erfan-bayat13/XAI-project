"""
MNIST Even-Odd Dataset for ViT Analysis
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from ..config import Config


class MNISTEvenOdd(torchvision.datasets.MNIST):
    """
    MNIST dataset modified for Even/Odd classification with concept labels
    - Concept: Digit identity (0-9)
    - Task: Even/Odd classification
    - Perfect for testing concept-based adversarial detection!
    """

    def __init__(self, root: str, train: bool = True, download: bool = True, 
                 dataset_size: int = None):
        """
        Initialize MNIST Even-Odd dataset
        
        Args:
            root: Root directory for dataset
            train: Whether to use training or test set
            download: Whether to download if not exists
            dataset_size: Size limit for dataset
        """
        # Transform for ViT compatibility
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((Config.MNIST_MEAN,), (Config.MNIST_STD,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        ])

        # Call parent constructor
        super(MNISTEvenOdd, self).__init__(root, train=train, transform=transform, download=download)

        # Reduce dataset size if specified
        if dataset_size is not None:
            self.data = self.data[:dataset_size]
            self.targets = self.targets[:dataset_size]

        self._print_info(dataset_size or len(self.data))

    def _print_info(self, size: int):
        """Print dataset information"""
        print(f"✅ MNIST Even-Odd dataset created:")
        print(f"   📊 Size: {size} samples")
        print(f"   🎯 Task: Even/Odd classification")
        print(f"   🏷️ Concepts: Digit identity (0-9)")
        print(f"   📐 Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}x3 (ViT compatible)")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get item at index
        
        Returns:
            Tuple of (image, concept, task)
            - image: Preprocessed image tensor
            - concept: Digit identity (0-9)
            - task: Even (0) or Odd (1)
        """
        img, target = super(MNISTEvenOdd, self).__getitem__(index)
        
        concept = target  # Concept label: the actual digit (0-9)
        task = target % 2  # Task label: even (0) or odd (1)
        
        return img, concept, task

    def get_concept_name(self, concept: int) -> str:
        """Get human-readable concept name"""
        return f"Digit_{concept}"

    def get_task_name(self, task: int) -> str:
        """Get human-readable task name"""
        return "Odd" if task == 1 else "Even"

    def analyze_distribution(self, name: str = "Dataset") -> Dict[str, Any]:
        """
        Analyze the distribution of concepts and tasks in the dataset
        
        Args:
            name: Name of the dataset for display
            
        Returns:
            Dictionary with distribution statistics
        """
        concepts = []
        tasks = []

        for i in range(len(self)):
            _, concept, task = self[i]
            concepts.append(concept)
            tasks.append(task)

        concepts = np.array(concepts)
        tasks = np.array(tasks)

        print(f"\n📊 {name} Distribution Analysis:")
        print(f"   Total samples: {len(self)}")

        # Concept distribution
        concept_dist = {}
        print(f"\n   🏷️ Concept (Digit) Distribution:")
        for digit in range(10):
            count = np.sum(concepts == digit)
            percentage = count / len(self) * 100
            concept_dist[digit] = count
            print(f"     Digit {digit}: {count} samples ({percentage:.1f}%)")

        # Task distribution
        even_count = np.sum(tasks == 0)
        odd_count = np.sum(tasks == 1)
        print(f"\n   🎯 Task Distribution:")
        print(f"     Even: {even_count} samples ({even_count/len(self)*100:.1f}%)")
        print(f"     Odd: {odd_count} samples ({odd_count/len(self)*100:.1f}%)")

        # Concept-Task mapping verification
        print(f"\n   🔍 Concept-Task Mapping Verification:")
        for digit in range(10):
            digit_samples = concepts == digit
            expected_task = digit % 2
            actual_tasks = tasks[digit_samples]
            correct_mapping = np.all(actual_tasks == expected_task)
            print(f"     Digit {digit} → {'Even' if expected_task == 0 else 'Odd'}: {'✅' if correct_mapping else '❌'}")

        return {
            'concept_distribution': concept_dist,
            'task_distribution': {'even': even_count, 'odd': odd_count},
            'total_samples': len(self)
        }

    def create_concept_profiles(self) -> Dict[int, Dict[str, Any]]:
        """
        Create concept profiles for each digit (0-9)
        This will be used later for adversarial detection
        
        Returns:
            Dictionary mapping digit to its profile
        """
        concept_profiles = {}

        for digit in range(10):
            concept_profiles[digit] = {
                'digit': digit,
                'expected_task': digit % 2,
                'task_name': 'Even' if digit % 2 == 0 else 'Odd',
                'samples_count': 0
            }

        # Count samples for each concept
        for i in range(len(self)):
            _, concept, task = self[i]
            concept_profiles[concept]['samples_count'] += 1

        print(f"\n📋 Concept Profiles Created:")
        for digit, profile in concept_profiles.items():
            print(f"   Digit {digit}: {profile['task_name']} ({profile['samples_count']} samples)")

        return concept_profiles

    def visualize_samples(self, num_samples: int = 10, title: str = "MNIST Even-Odd Samples"):
        """
        Visualize samples from the dataset
        
        Args:
            num_samples: Number of samples to visualize
            title: Title for the plot
        """
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        # Get random samples
        indices = np.random.choice(len(self), num_samples, replace=False)

        for i, idx in enumerate(indices):
            image, concept, task = self[idx]

            # Convert tensor to displayable format
            img_display = image.clone()
            img_display = img_display * Config.MNIST_STD + Config.MNIST_MEAN  # Denormalize
            img_display = torch.clamp(img_display, 0, 1)

            # Convert to grayscale for display (take one channel since all 3 are the same)
            img_display = img_display[0]  # Take first channel

            axes[i].imshow(img_display.numpy(), cmap='gray')
            axes[i].set_title(f'Digit: {concept}\nTask: {self.get_task_name(task)}', fontsize=10)
            axes[i].axis('off')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def create_data_loaders(root: str = './data', 
                       train_size: int = Config.TRAIN_DATASET_SIZE,
                       test_size: int = Config.TEST_DATASET_SIZE,
                       batch_size: int = Config.BATCH_SIZE) -> Tuple[torch.utils.data.DataLoader, 
                                                                    torch.utils.data.DataLoader,
                                                                    MNISTEvenOdd, MNISTEvenOdd]:
    """
    Create train and test data loaders
    
    Args:
        root: Root directory for dataset
        train_size: Size of training dataset
        test_size: Size of test dataset
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, test_loader, train_dataset, test_dataset)
    """
    print("📁 Loading MNIST Even-Odd datasets...")
    
    train_dataset = MNISTEvenOdd(root=root, train=True, download=True, dataset_size=train_size)
    test_dataset = MNISTEvenOdd(root=root, train=False, download=True, dataset_size=test_size)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Training set: {len(train_dataset)} samples")
    print(f"✅ Test set: {len(test_dataset)} samples")
    
    return train_loader, test_loader, train_dataset, test_dataset