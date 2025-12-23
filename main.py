#!/usr/bin/env python3
"""
Main application for differentiable IMU state estimation
"""

import argparse
import jax
import time
import numpy as np
from pathlib import Path

from dynamics.rigid_body import RigidBodyDynamics
from estimator.ekf import DifferentiableEKF
from models.bias_net import BiasCorrectionNet, LearnedEKF
from training.trainer import EKFTrainer
from deployment.realtime import RealTimeEstimator
from data.collector import DataCollector
from utils.logging_config import setup_logging
from config.config_manager import config_manager
from utils.exceptions import KalmanFilterError
from testing.test_environment import run_tests
from utils.model_utils import save_model, load_model, find_latest_model
from utils.data_utils import load_training_data


def main():
    # Setup logging first
    config = config_manager.get_config()
    logger = setup_logging(
        level=config.logging.LEVEL,
        log_file="main.log",
        component="main"
    )
    
    parser = argparse.ArgumentParser(
        description="Professional Differentiable IMU State Estimation"
    )
    parser.add_argument("--mode", choices=["collect", "train", "deploy", "test"],
                       required=True, help="Operation mode")
    parser.add_argument("--duration", type=float, default=30.0,
                       help="Duration for collection/deployment")
    parser.add_argument("--epochs", type=int, default=config.training.MAX_EPOCHS,
                       help="Training epochs")
    args = parser.parse_args()
    
    try:
        if args.mode == "collect":
            logger.info("=== Data Collection Mode ===")
            collector = DataCollector()
            
            collector.collect_stationary(duration=60.0)
            collector.collect_rotation(duration=30.0)
            collector.collect_trajectory(duration=120.0)
            
            logger.info("Data collection complete!")

        elif args.mode == "train":
            logger.info("=== Training Mode ===")
            
            dynamics = RigidBodyDynamics()
            ekf = DifferentiableEKF(dynamics)
            bias_net = BiasCorrectionNet()
            learned_ekf = LearnedEKF(ekf, bias_net)
            
            trainer = EKFTrainer(
                learned_ekf, 
                learning_rate=config.training.LEARNING_RATE,
                batch_size=config.training.BATCH_SIZE,
                validation_split=config.training.VALIDATION_SPLIT
            )
            
            logger.info("Loading dataset...")
            dataset = load_training_data()  
            
            if dataset is None:
                logger.error("No valid training data available")
                return 1
            
            logger.info(f"Starting training for {args.epochs} epochs...")
            losses = trainer.train(dataset, num_epochs=args.epochs)
            
            model_path = f"models/trained_model_{int(time.time())}.pkl"
            save_model(trainer.state.params, model_path)
            logger.info(f"Training complete. Model saved to {model_path}")

        elif args.mode == "deploy":
            logger.info("=== Deployment Mode ===")
            
            model_path = find_latest_model()
            if not model_path:
                logger.error("No trained model found. Run training first.")
                return 1
            
            params = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            
            dynamics = RigidBodyDynamics()
            ekf = DifferentiableEKF(dynamics)
            bias_net = BiasCorrectionNet()
            learned_ekf = LearnedEKF(ekf, bias_net)
            
            estimator = RealTimeEstimator(learned_ekf, params)
            estimator.run(duration=args.duration, visualize=True)
        
        elif args.mode == "test":
            logger.info("=== Test Mode ===")
            test_results = run_tests()
            
            if test_results["overall_status"] == "PASSED":
                logger.info("All tests passed - system ready for production")
                return 0
            else:
                logger.error(f"Tests failed: {test_results['overall_status']}")
                return 1
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    exit(main())
