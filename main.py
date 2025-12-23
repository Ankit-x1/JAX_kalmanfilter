#!/usr/bin/env python3
"""
Main application for differentiable IMU state estimation
"""

import argparse
import jax
import numpy as np
from pathlib import Path

from dynamics.rigid_body import RigidBodyDynamics
from estimator.ekf import DifferentiableEKF
from models.bias_net import BiasCorrectionNet, LearnedEKF
from training.trainer import EKFTrainer
from deployment.realtime import RealTimeEstimator
from data.collector import DataCollector

def main():
    parser = argparse.ArgumentParser(
        description="Differentiable IMU State Estimation"
    )
    parser.add_argument("--mode", choices=["collect", "train", "deploy"],
                       required=True, help="Operation mode")
    parser.add_argument("--duration", type=float, default=30.0,
                       help="Duration for collection/deployment")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs")
    args = parser.parse_args()
    
    if args.mode == "collect":
        print("=== Data Collection Mode ===")
        collector = DataCollector()
        
        collector.collect_stationary(duration=60.0)
        collector.collect_rotation(duration=30.0)
        collector.collect_trajectory(duration=120.0)
        
        print("Data collection complete!")
    
    elif args.mode == "train":
        print("=== Training Mode ===")
        
        dynamics = RigidBodyDynamics()
        
        Q = np.eye(13) * 1e-6
        R = np.eye(6) * 1e-4
        
        ekf = DifferentiableEKF(dynamics, Q, R)
        
        bias_net = BiasCorrectionNet()
        
        learned_ekf = LearnedEKF(ekf, bias_net)
        
        trainer = EKFTrainer(learned_ekf, learning_rate=1e-3)
        
        print("Loading dataset...")
        dataset = load_training_data()  
        
        print("Starting training...")
        losses = trainer.train(dataset, num_epochs=args.epochs)
        
        save_model(trainer.state.params, "trained_model.pkl")
        print(f"Training complete. Model saved.")
    
    elif args.mode == "deploy":
        print("=== Deployment Mode ===")
        
        params = load_model("trained_model.pkl")
        
        dynamics = RigidBodyDynamics()
        Q = np.eye(13) * 1e-6
        R = np.eye(6) * 1e-4
        ekf = DifferentiableEKF(dynamics, Q, R)
        bias_net = BiasCorrectionNet()
        learned_ekf = LearnedEKF(ekf, bias_net)
        
        estimator = RealTimeEstimator(learned_ekf, params)
        estimator.run(duration=args.duration, visualize=True)

def load_training_data():
    """Load and preprocess training data"""
    import json
    from pathlib import Path
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"Warning: Data directory {data_dir} not found. Using dummy data.")
        # Return dummy data for testing
        n_samples = 1000
        states = np.random.randn(n_samples, 13)
        measurements = np.random.randn(n_samples, 6)
        targets = np.random.randn(n_samples, 13)
        return states, measurements, targets
    
    # Load all JSON files in data directory
    all_data = []
    for file_path in data_dir.glob("*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.extend(data['samples'])
    
    if not all_data:
        print("Warning: No training data found. Using dummy data.")
        n_samples = 1000
        states = np.random.randn(n_samples, 13)
        measurements = np.random.randn(n_samples, 6)
        targets = np.random.randn(n_samples, 13)
        return states, measurements, targets
    
    # Convert to numpy arrays
    n_samples = len(all_data)
    measurements = np.array([[s['gyro'] + s['accel'] for s in all_data]]).reshape(n_samples, 6)
    
    # Create dummy states and targets (in real implementation, these would be computed)
    states = np.zeros((n_samples, 13))
    states[:, 0:4] = 1.0  # Identity quaternion
    targets = states.copy()
    
    print(f"Loaded {n_samples} training samples")
    return states, measurements, targets

def save_model(params, filename):
    """Save model parameters"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def load_model(filename):
    """Load model parameters"""
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()