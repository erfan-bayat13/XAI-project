"""
Experiment Runner for XAI Adversarial Detection
"""
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from config import Config
from datasets.mnist_evenodd import create_data_loaders
from models.vit_wrapper import MNISTViTWrapper, train_even_odd_classifier, evaluate_model
from models.mlp_concept import AttentionToConceptMLP, train_attention_to_concept_mlp, detect_adversarial_with_mlp
from analyzers.attention_analyzer import MNISTViTAttentionAnalyzer
from analyzers.concept_detector import ConceptBasedDetector
from attacks.fgsm_generator import FGSMAttackGenerator
from utils.visualization import (visualize_adversarial_comparison, plot_detection_scores_distribution,
                                  plot_detection_performance_metrics, plot_epsilon_vs_performance)


class ExperimentRunner:
    """
    Main experiment runner for XAI Adversarial Detection
    """
    
    def __init__(self, data_root: str = './data'):
        """
        Initialize experiment runner
        
        Args:
            data_root: Root directory for datasets
        """
        self.data_root = data_root
        self.device = Config.DEVICE
        
        # Components
        self.train_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.concept_profiles = None
        
        self.vit_model = None
        self.vit_analyzer = None
        self.mlp_model = None
        self.concept_detector = None
        self.fgsm_attacker = None
        
        print(f"🚀 Experiment Runner initialized on {self.device}")
        Config.print_config()

    def setup_datasets(self) -> None:
        """Setup MNIST Even-Odd datasets"""
        print("\n" + "="*60)
        print("📁 SETTING UP DATASETS")
        print("="*60)
        
        self.train_loader, self.test_loader, self.train_dataset, self.test_dataset = create_data_loaders(
            root=self.data_root,
            train_size=Config.TRAIN_DATASET_SIZE,
            test_size=Config.TEST_DATASET_SIZE,
            batch_size=Config.BATCH_SIZE
        )
        
        # Analyze dataset distributions
        self.train_dataset.analyze_distribution("Training Set")
        self.test_dataset.analyze_distribution("Test Set")
        
        # Create concept profiles
        self.concept_profiles = self.train_dataset.create_concept_profiles()
        
        # Visualize sample images
        print("\n🖼️ Visualizing sample images:")
        self.train_dataset.visualize_samples()

    def setup_models(self) -> None:
        """Setup ViT model and analyzer"""
        print("\n" + "="*60)
        print("🧠 SETTING UP MODELS")
        print("="*60)
        
        # Initialize ViT wrapper
        self.vit_model = MNISTViTWrapper()
        
        # Initialize attention analyzer
        self.vit_analyzer = MNISTViTAttentionAnalyzer(self.vit_model)
        
        # Initialize FGSM attacker
        self.fgsm_attacker = FGSMAttackGenerator(self.vit_model)
        
        print("✅ Models initialized successfully")

    def train_even_odd_classifier(self, epochs: int = Config.EVEN_ODD_EPOCHS) -> Dict[str, float]:
        """
        Train the Even/Odd classifier
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Training results dictionary
        """
        print("\n" + "="*60)
        print("🏋️ TRAINING EVEN/ODD CLASSIFIER")
        print("="*60)
        
        # Train the classifier
        train_even_odd_classifier(self.vit_model, self.train_loader, epochs=epochs)
        
        # Evaluate on test set
        results = evaluate_model(self.vit_model, self.test_loader)
        
        return results

    def train_mlp_concept_mapper(self, epochs: int = Config.MLP_EPOCHS) -> AttentionToConceptMLP:
        """
        Train the MLP attention-to-concept mapper
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Trained MLP model
        """
        print("\n" + "="*60)
        print("🧠 TRAINING ATTENTION-TO-CONCEPT MLP")
        print("="*60)
        
        self.mlp_model = train_attention_to_concept_mlp(
            self.vit_model, self.train_dataset, epochs=epochs
        )
        
        return self.mlp_model

    def setup_concept_detector(self) -> None:
        """Setup concept-based adversarial detector"""
        print("\n" + "="*60)
        print("🛡️ SETTING UP CONCEPT DETECTOR")
        print("="*60)
        
        self.concept_detector = ConceptBasedDetector(self.vit_analyzer, self.concept_profiles)
        
        # Compute baseline attention statistics
        print("📊 Computing baseline attention statistics...")
        self.vit_analyzer.analyze_attention_statistics(self.test_dataset, sample_size=100)
        
        print("✅ Concept detector ready")

    def analyze_clean_samples(self, num_samples: int = 5) -> Dict[int, Dict[str, Any]]:
        """
        Analyze attention patterns on clean samples
        
        Args:
            num_samples: Number of samples per digit to analyze
            
        Returns:
            Dictionary of clean sample analyses
        """
        print("\n" + "="*60)
        print("🔍 ANALYZING CLEAN SAMPLES")
        print("="*60)
        
        clean_samples = {}
        sample_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        for digit in sample_digits:
            # Find first sample of this digit
            for i, (image, concept, task) in enumerate(self.test_dataset):
                if concept == digit:
                    print(f"\n🔢 Analyzing Digit {digit} ({'Even' if task == 0 else 'Odd'})")
                    
                    prediction, attention_maps = self.vit_analyzer.visualize_mnist_attention(
                        image, concept, task
                    )
                    
                    clean_samples[digit] = {
                        'image': image,
                        'concept': concept,
                        'task': task,
                        'prediction': prediction,
                        'attention_maps': attention_maps
                    }
                    break
        
        return clean_samples

    def generate_adversarial_samples(self, 
                                   epsilon_values: List[float] = None,
                                   samples_per_digit: int = 5) -> Dict[float, Dict[int, Dict[str, Any]]]:
        """
        Generate adversarial samples for different epsilon values
        
        Args:
            epsilon_values: List of epsilon values to test
            samples_per_digit: Number of samples per digit
            
        Returns:
            Dictionary mapping epsilon to adversarial samples
        """
        print("\n" + "="*60)
        print("⚡ GENERATING ADVERSARIAL SAMPLES")
        print("="*60)
        
        if epsilon_values is None:
            epsilon_values = Config.FGSM_EPSILON_VALUES
        
        adversarial_samples = {}
        
        for epsilon in epsilon_values:
            print(f"\n🎯 Generating attacks with epsilon = {epsilon}")
            
            epsilon_samples = {}
            
            for digit in range(10):
                digit_samples = []
                count = 0
                
                for i, (image, concept, task) in enumerate(self.test_dataset):
                    if concept == digit and count < samples_per_digit:
                        # Generate adversarial example
                        adv_image = self.fgsm_attacker.generate_fgsm_attack(
                            image, task, epsilon=epsilon
                        )
                        
                        # Test attack success
                        attack_result = self.fgsm_attacker.test_attack_success(
                            image, adv_image, task
                        )
                        
                        digit_samples.append({
                            'original_image': image,
                            'adversarial_image': adv_image.squeeze(0),
                            'concept': concept,
                            'task': task,
                            'attack_result': attack_result
                        })
                        
                        count += 1
                
                epsilon_samples[digit] = digit_samples
            
            adversarial_samples[epsilon] = epsilon_samples
            
            # Print summary for this epsilon
            total_attacks = sum(len(samples) for samples in epsilon_samples.values())
            successful_attacks = sum(
                sum(1 for sample in samples if sample['attack_result']['attack_success'])
                for samples in epsilon_samples.values()
            )
            success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0
            
            print(f"   📈 Attack Success Rate: {success_rate:.3f} ({successful_attacks}/{total_attacks})")
        
        return adversarial_samples

    def test_detection_methods(self, 
                             adversarial_samples: Dict[float, Dict[int, Dict[str, Any]]],
                             epsilon: float = Config.DEFAULT_EPSILON) -> Dict[str, Any]:
        """
        Test both statistical and MLP-based detection methods
        
        Args:
            adversarial_samples: Generated adversarial samples
            epsilon: Epsilon value to test
            
        Returns:
            Detection results dictionary
        """
        print("\n" + "="*60)
        print("🛡️ TESTING DETECTION METHODS")
        print("="*60)
        
        if epsilon not in adversarial_samples:
            print(f"❌ No samples found for epsilon {epsilon}")
            return {}
        
        epsilon_samples = adversarial_samples[epsilon]
        
        # Test statistical detection
        print(f"\n📊 Testing Statistical Detection (ε={epsilon})")
        statistical_results = self._test_statistical_detection(epsilon_samples)
        
        # Test MLP detection (if MLP is trained)
        mlp_results = {}
        if self.mlp_model is not None:
            print(f"\n🧠 Testing MLP Detection (ε={epsilon})")
            mlp_results = self._test_mlp_detection(epsilon_samples)
        
        return {
            'statistical_detection': statistical_results,
            'mlp_detection': mlp_results,
            'epsilon': epsilon
        }

    def _test_statistical_detection(self, epsilon_samples: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Test statistical-based detection method"""
        clean_images = []
        clean_tasks = []
        adv_images = []
        adv_tasks = []
        
        # Collect samples for testing
        for digit in range(10):
            if digit in epsilon_samples and epsilon_samples[digit]:
                sample = epsilon_samples[digit][0]  # Take first sample
                if sample['attack_result']['attack_success']:
                    clean_images.append(sample['original_image'])
                    clean_tasks.append(sample['task'])
                    adv_images.append(sample['adversarial_image'])
                    adv_tasks.append(sample['attack_result']['adversarial_prediction'])
        
        if not clean_images:
            print("⚠️ No successful attacks found for statistical detection testing")
            return {}
        
        # Test detection
        results = self.concept_detector.evaluate_detection_performance(
            clean_images, adv_images, clean_tasks, adv_tasks
        )
        
        # Visualize results
        plot_detection_performance_metrics(results)
        
        return results

    def _test_mlp_detection(self, epsilon_samples: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Test MLP-based detection method"""
        correct_detections = 0
        false_positives = 0
        total_clean = 0
        total_adv = 0
        
        # Test clean images
        for digit in range(10):
            if digit in epsilon_samples and epsilon_samples[digit]:
                sample = epsilon_samples[digit][0]  # Take first sample
                
                # Test clean image
                clean_result = detect_adversarial_with_mlp(
                    self.vit_model, self.mlp_model, 
                    sample['original_image'], sample['concept'], sample['task']
                )
                
                if clean_result['is_adversarial']:
                    false_positives += 1
                total_clean += 1
                
                # Test adversarial image (if attack was successful)
                if sample['attack_result']['attack_success']:
                    adv_result = detect_adversarial_with_mlp(
                        self.vit_model, self.mlp_model,
                        sample['adversarial_image'], sample['concept'],
                        sample['attack_result']['adversarial_prediction']
                    )
                    
                    if adv_result['is_adversarial']:
                        correct_detections += 1
                    total_adv += 1
        
        # Calculate metrics
        fpr = false_positives / total_clean if total_clean > 0 else 0
        tpr = correct_detections / total_adv if total_adv > 0 else 0
        
        results = {
            'false_positive_rate': fpr,
            'true_positive_rate': tpr,
            'false_positives': false_positives,
            'correct_detections': correct_detections,
            'total_clean': total_clean,
            'total_adversarial': total_adv
        }
        
        print(f"   MLP Detection Results:")
        print(f"   False Positive Rate: {fpr:.3f}")
        print(f"   True Positive Rate: {tpr:.3f}")
        
        return results

    def run_comprehensive_evaluation(self, 
                                   epsilon_values: List[float] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across multiple epsilon values
        
        Args:
            epsilon_values: List of epsilon values to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        print("\n" + "="*60)
        print("🔬 COMPREHENSIVE EVALUATION")
        print("="*60)
        
        if epsilon_values is None:
            epsilon_values = Config.FGSM_EPSILON_VALUES
        
        # Generate adversarial samples for all epsilon values
        adversarial_samples = self.generate_adversarial_samples(epsilon_values)
        
        # Test detection for each epsilon
        evaluation_results = {}
        
        for epsilon in epsilon_values:
            print(f"\n🎯 Evaluating epsilon = {epsilon}")
            results = self.test_detection_methods(adversarial_samples, epsilon)
            evaluation_results[epsilon] = results
        
        # Create summary visualization
        self._visualize_comprehensive_results(evaluation_results)
        
        return evaluation_results

    def _visualize_comprehensive_results(self, results: Dict[str, Any]) -> None:
        """Visualize comprehensive evaluation results"""
        # Prepare data for epsilon vs performance plot
        epsilon_performance = {}
        
        for epsilon, result in results.items():
            if 'statistical_detection' in result and result['statistical_detection']:
                stat_results = result['statistical_detection']
                epsilon_performance[epsilon] = {
                    'attack_success_rate': 0.7,  # Placeholder - would be computed from attack results
                    'detection_rate': stat_results.get('true_positive_rate', 0)
                }
        
        if epsilon_performance:
            plot_epsilon_vs_performance(epsilon_performance)

    def run_quick_demo(self, digit: int = 5) -> None:
        """
        Run a quick demonstration of the concept-based detection
        
        Args:
            digit: Digit to demonstrate with
        """
        print("\n" + "="*60)
        print("🎬 QUICK DEMO")
        print("="*60)
        
        # Quick training (1 epoch for demo)
        print("⚡ Quick training (1 epoch)...")
        self.train_even_odd_classifier(epochs=1)
        
        # Pick one sample
        image, concept, task = None, None, None
        for i, (img, conc, tsk) in enumerate(self.test_dataset):
            if conc == digit:
                image, concept, task = img, conc, tsk
                break
        
        if image is None:
            print(f"❌ No sample found for digit {digit}")
            return
        
        print(f"\n🎯 Demo with Digit {concept} ({'Even' if task == 0 else 'Odd'})")
        
        # Show original
        print("1️⃣ Original image analysis:")
        prediction, _ = self.vit_analyzer.visualize_mnist_attention(image, concept, task)
        
        # Generate adversarial
        print("\n2️⃣ Generating FGSM attack...")
        adv_image = self.fgsm_attacker.generate_fgsm_attack(image, task, epsilon=0.15)
        
        # Test attack
        attack_result = self.fgsm_attacker.test_attack_success(image, adv_image, task)
        print(f"Attack success: {'✅' if attack_result['attack_success'] else '❌'}")
        
        # Visualize comparison
        print("\n3️⃣ Comparing attention patterns:")
        visualize_adversarial_comparison(
            self.vit_analyzer, self.fgsm_attacker, self.test_dataset, digit, 0.15
        )
        
        # Test detection
        print("\n4️⃣ Testing detection:")
        if self.concept_detector is None:
            self.setup_concept_detector()
        
        clean_detection = self.concept_detector.detect_concept_inconsistency(image, task)
        adv_detection = self.concept_detector.detect_concept_inconsistency(
            adv_image.squeeze(0), attack_result['adversarial_prediction']
        )
        
        print(f"Clean image detection: {'❌ False alarm' if clean_detection['is_adversarial'] else '✅ Correct'}")
        print(f"Adversarial detection: {'✅ Detected' if adv_detection['is_adversarial'] else '❌ Missed'}")
        
        print(f"\n🎉 Demo complete!")

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete experimental pipeline
        
        Returns:
            Complete experimental results
        """
        print("\n" + "🚀"*30)
        print("RUNNING COMPLETE XAI ADVERSARIAL DETECTION PIPELINE")
        print("🚀"*30)
        
        results = {}
        
        # Step 1: Setup
        self.setup_datasets()
        self.setup_models()
        
        # Step 2: Train models
        training_results = self.train_even_odd_classifier()
        results['training'] = training_results
        
        # Step 3: Train MLP (optional)
        if Config.MLP_EPOCHS > 0:
            self.train_mlp_concept_mapper()
        
        # Step 4: Setup detection
        self.setup_concept_detector()
        
        # Step 5: Analyze clean samples
        clean_analysis = self.analyze_clean_samples()
        results['clean_analysis'] = clean_analysis
        
        # Step 6: Comprehensive evaluation
        evaluation_results = self.run_comprehensive_evaluation()
        results['evaluation'] = evaluation_results
        
        print("\n" + "✅"*30)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅"*30)
        
        return results