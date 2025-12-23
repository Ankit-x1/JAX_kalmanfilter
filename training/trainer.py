import jax
import jax.numpy as jnp
from jax import jit
import optax
from flax.training import train_state
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import time

from estimator.ekf import EKFState
from models.bias_net import LearnedEKF
from utils.logging_config import ComponentLogger
from config.config_manager import config_manager
from utils.exceptions import TrainingError, ModelError


class TrainState(train_state.TrainState):
    """Training state with additional metrics"""

    batch_stats: Dict = None


class EKFTrainer:
    """Professional training of differentiable EKF"""

    def __init__(
        self,
        learned_ekf: LearnedEKF,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        validation_split: Optional[float] = None,
    ):
        self.logger = ComponentLogger("trainer")
        self.learned_ekf = learned_ekf

        # Get configuration
        self.config = config_manager.get_config().training
        self.learning_rate = learning_rate or self.config.LEARNING_RATE
        self.batch_size = batch_size or self.config.BATCH_SIZE
        self.validation_split = validation_split or self.config.VALIDATION_SPLIT

        # Initialize training state
        key = jax.random.PRNGKey(int(time.time()) % 2**32)
        init_state = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        init_params = self.learned_ekf.model.init(key, init_state)

        tx = optax.adam(self.learning_rate)

        self.state = TrainState.create(
            apply_fn=self.learned_ekf.model.apply,
            params=init_params,
            tx=tx,
            batch_stats={},
        )

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = []

        self.logger.info(
            f"EKFTrainer initialized with lr={self.learning_rate}, batch_size={self.batch_size}"
        )

    @jit
    def loss_fn(self, params, batch):
        """Loss function: MSE between predicted and measured accelerations"""
        states, measurements, targets = batch
        n_samples = states.shape[0]

        total_loss = 0.0

        x0 = states[0]
        P0 = jnp.eye(13) * config_manager.get_config().ekf.INITIAL_COVARIANCE
        ekf_state = EKFState(x0, P0, 0.0)

        def body_fn(i, carry):
            ekf_state, total_loss = carry

            u = measurements[i, :6]
            z_accel = measurements[i, 3:6]
            dt = 0.01

            ekf_state = self.learned_ekf.step(params, ekf_state, u, z_accel, u[0:3], dt)

            accel_pred, _ = self.learned_ekf.ekf.dynamics.measurement_model(ekf_state.x)
            loss = jnp.mean((accel_pred - z_accel) ** 2)

            gyro_corr, accel_corr = self.learned_ekf.model.apply(params, ekf_state.x)
            reg_loss = 0.001 * (jnp.sum(gyro_corr**2) + jnp.sum(accel_corr**2))

            total_loss += loss + reg_loss

            return (ekf_state, total_loss)

        final_state, total_loss = jax.lax.fori_loop(
            0, n_samples, body_fn, (ekf_state, 0.0)
        )

        return total_loss / n_samples

    @jit
    def train_step(self, state, batch):
        """Single training step"""
        loss_fn = lambda params: self.loss_fn(params, batch)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        state = state.apply_gradients(grads=grads)

        return state, loss

    def train(self, dataset, num_epochs: int = 100, batch_size: int = 32):
        """Professional training loop with validation and early stopping"""
        num_epochs = num_epochs or self.config.MAX_EPOCHS
        losses = []
        val_losses = []
        
        # Split dataset
        train_data, val_data = self._split_dataset(dataset)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training samples: {len(train_data[0])}, Validation samples: {len(val_data[0])}")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            epoch_loss = 0.0
            num_batches = 0
            
            batches = self._create_batches(train_data, self.batch_size)
            
            try:
                with tqdm(batches, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=False) as pbar:
                    for batch in pbar:
                        self.state, loss = self.train_step(self.state, batch)
                        epoch_loss += loss
                        num_batches += 1
                        
                        pbar.set_postfix({
                            "loss": f"{loss:.6f}",
                            "lr": f"{self.learning_rate:.2e}"
                        })
                
                avg_train_loss = epoch_loss / num_batches
                losses.append(avg_train_loss)
                
                # Validation phase
                val_loss = self._validate(val_data)
                val_losses.append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Save best model
                    checkpoint_path = Path("models/best_model.pkl")
                    self._save_checkpoint(checkpoint_path)
                    self.logger.debug(f"New best model saved: {val_loss:.6f}")
                else:
                    self.patience_counter += 1
                
                epoch_time = time.time() - epoch_start
                self.logger.info(f"Epoch {epoch + 1}: Train={avg_train_loss:.6f}, "
                               f"Val={val_loss:.6f}, Time={epoch_time:.2f}s")
                
                # Early stopping
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Save checkpoint
                if (epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                    checkpoint_path = Path(f"models/checkpoint_epoch_{epoch + 1}.pkl")
                    self._save_checkpoint(checkpoint_path)
                
            except Exception as e:
                self.logger.error(f"Training epoch {epoch + 1} failed: {e}")
                raise TrainingError(f"Epoch {epoch + 1} failed: {e}")
        
        # Load best model for final state
        best_checkpoint = Path("models/best_model.pkl")
        if best_checkpoint.exists():
            self._load_checkpoint(best_checkpoint)
        
        self.training_history = {
            'train_losses': losses,
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': epoch + 1
        }
        
        self.logger.info(f"Training complete. Best validation loss: {self.best_val_loss:.6f}")
        return self.training_history

    def _split_dataset(self, dataset):
        """Split dataset into training and validation"""
        states, measurements, targets = dataset
        n_samples = states.shape[0]
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * (1 - self.validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_data = (
            states[train_indices],
            measurements[train_indices],
            targets[train_indices]
        )
        
        val_data = (
            states[val_indices],
            measurements[val_indices],
            targets[val_indices]
        )
        
        return train_data, val_data
    
    def _validate(self, val_data):
        """Validate model on validation set"""
        val_loss = 0.0
        num_batches = 0
        
        batches = self._create_batches(val_data, self.batch_size)
        
        for batch in batches:
            loss = self.loss_fn(self.state.params, batch)
            val_loss += loss
            num_batches += 1
        
        return val_loss / num_batches if num_batches > 0 else float('inf')
    
    def _save_checkpoint(self, checkpoint_path: Path):
        """Save model checkpoint"""
        try:
            import pickle
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(self.state.params, f)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        try:
            import pickle
            
            with open(checkpoint_path, 'rb') as f:
                params = pickle.load(f)
            
            self.state = self.state.replace(params=params)
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")

    def _create_batches(self, dataset, batch_size):
        """Create training batches from dataset"""
        states, measurements, targets = dataset

        n_samples = states.shape[0]
        indices = np.arange(n_samples - batch_size)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples - batch_size, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            batch_states = jnp.stack([states[i] for i in batch_indices])
            batch_meas = jnp.stack([measurements[i] for i in batch_indices])
            batch_targets = jnp.stack([targets[i] for i in batch_indices])

            yield (batch_states, batch_meas, batch_targets)
