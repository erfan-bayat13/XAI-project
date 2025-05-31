"""
Configuration file for XAI Adversarial Detection project
"""
import torch

class Config:
    """Configuration class for the XAI Adversarial Detection project"""
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model configurations
    VIT_MODEL_NAME = "google/vit-base-patch16-224"
    VIT_HIDDEN_SIZE = 768
    VIT_NUM_LAYERS = 12
    
    # Dataset configurations
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    TRAIN_DATASET_SIZE = 10000
    TEST_DATASET_SIZE = 5000
    
    # MLP configurations
    MLP_INPUT_SIZE = 603  # 3 layers * (196 + 5) = 603 (196 spatial + 5 statistics per layer)
    MLP_HIDDEN_SIZE_1 = 256
    MLP_HIDDEN_SIZE_2 = 128
    MLP_DROPOUT = 0.3
    MLP_NUM_CONCEPTS = 10
    MLP_LEARNING_RATE = 1e-3
    
    # Training configurations
    EVEN_ODD_EPOCHS = 3
    MLP_EPOCHS = 30
    
    # Attack configurations
    FGSM_EPSILON_VALUES = [0.05, 0.1, 0.15, 0.2, 0.3]
    DEFAULT_EPSILON = 0.1
    
    # Detection configurations
    MIDDLE_LAYERS = [6, 7, 8]  # Where concepts emerge in ViT
    DETECTION_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.5
    
    # Visualization configurations
    KEY_LAYERS = [0, 3, 6, 9, 11]
    ATTENTION_COLORMAP = 'hot'
    
    # Paths
    MODEL_SAVE_PATH = 'mnist_evenodd_vit.pth'
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 Current Configuration:")
        print(f"   Device: {cls.DEVICE}")
        print(f"   ViT Model: {cls.VIT_MODEL_NAME}")
        print(f"   Image Size: {cls.IMAGE_SIZE}")
        print(f"   Batch Size: {cls.BATCH_SIZE}")
        print(f"   Training Dataset Size: {cls.TRAIN_DATASET_SIZE}")
        print(f"   Test Dataset Size: {cls.TEST_DATASET_SIZE}")
        print(f"   MLP Input Size: {cls.MLP_INPUT_SIZE}")
        print(f"   Middle Layers: {cls.MIDDLE_LAYERS}")
        print(f"   Detection Threshold: {cls.DETECTION_THRESHOLD}")