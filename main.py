#!/usr/bin/env python3
"""
Main entry point for XAI Adversarial Detection Project
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiments.runner import ExperimentRunner
from config import Config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='XAI Adversarial Detection for Vision Transformers')
    
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'full', 'train', 'evaluate'],
                       help='Execution mode')
    
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for datasets')
    
    parser.add_argument('--digit', type=int, default=5,
                       help='Digit to analyze in demo mode')
    
    parser.add_argument('--epsilon', type=float, default=Config.DEFAULT_EPSILON,
                       help='FGSM epsilon for demo')
    
    parser.add_argument('--epochs', type=int, default=Config.EVEN_ODD_EPOCHS,
                       help='Number of training epochs')
    
    parser.add_argument('--mlp-epochs', type=int, default=Config.MLP_EPOCHS,
                       help='Number of MLP training epochs')
    
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Disable visualizations')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Configure device
    if args.device != 'auto':
        import torch
        Config.DEVICE = torch.device(args.device)
    
    # Update config with command line arguments
    if args.epochs != Config.EVEN_ODD_EPOCHS:
        Config.EVEN_ODD_EPOCHS = args.epochs
    
    if args.mlp_epochs != Config.MLP_EPOCHS:
        Config.MLP_EPOCHS = args.mlp_epochs
    
    print("🚀 XAI Adversarial Detection for Vision Transformers")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Device: {Config.DEVICE}")
    print(f"Data root: {args.data_root}")
    print("=" * 80)
    
    # Initialize experiment runner
    runner = ExperimentRunner(data_root=args.data_root)
    
    try:
        if args.mode == 'demo':
            print("\n🎬 Running Quick Demo...")
            runner.setup_datasets()
            runner.setup_models()
            runner.setup_concept_detector()
            runner.run_quick_demo(digit=args.digit)
            
        elif args.mode == 'full':
            print("\n🔬 Running Full Pipeline...")
            results = runner.run_full_pipeline()
            print(f"\n📊 Results saved to: results_full_pipeline.json")
            
        elif args.mode == 'train':
            print("\n🏋️ Training Models Only...")
            runner.setup_datasets()
            runner.setup_models()
            
            # Train ViT classifier
            training_results = runner.train_even_odd_classifier(epochs=args.epochs)
            
            # Train MLP if requested
            if args.mlp_epochs > 0:
                runner.train_mlp_concept_mapper(epochs=args.mlp_epochs)
            
            print(f"\n✅ Training completed successfully!")
            print(f"📊 Overall accuracy: {training_results['overall_accuracy']:.2f}%")
            
        elif args.mode == 'evaluate':
            print("\n📊 Evaluation Only...")
            runner.setup_datasets()
            runner.setup_models()
            
            # Load trained model (assumes it exists)
            try:
                runner.vit_model.load_model()
                print("✅ Loaded pre-trained model")
            except:
                print("⚠️ No pre-trained model found, training first...")
                runner.train_even_odd_classifier(epochs=1)
            
            # Setup detection
            runner.setup_concept_detector()
            
            # Run evaluation
            results = runner.run_comprehensive_evaluation()
            print(f"\n📊 Evaluation completed successfully!")
            
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 Execution completed successfully!")


if __name__ == '__main__':
    main()