import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple

class TrainState(train_state.TrainState):
    """Training state with additional metrics"""
    batch_stats: Dict = None

class EKFTrainer:
    """End-to-end training of differentiable EKF"""
    
    def __init__(self, learned_ekf: LearnedEKF, 
                 learning_rate: float = 1e-3):
        self.learned_ekf = learned_ekf
        self.learning_rate = learning_rate
        
        key = jax.random.PRNGKey(42)
        dummy_state = jnp.ones(13)  
        dummy_params = self.learned_ekf.model.init(key, dummy_state)
        
        tx = optax.adam(learning_rate)
        
        self.state = TrainState.create(
            apply_fn=self.learned_ekf.model.apply,
            params=dummy_params,
            tx=tx,
            batch_stats={}
        )
        
    @jit
    def loss_fn(self, params, batch):
        """Loss function: MSE between predicted and measured accelerations"""
        states, measurements, targets = batch
        n_samples = states.shape[0]
        
        total_loss = 0.0
        
        x0 = states[0]
        P0 = jnp.eye(13) * 1e-4
        ekf_state = EKFState(x0, P0, 0.0)
        
        def body_fn(i, carry):
            ekf_state, total_loss = carry
            
            u = measurements[i, :6]  
            z_accel = measurements[i, 3:6]  
            dt = 0.01 
            
            ekf_state = self.learned_ekf.step(
                params, ekf_state, u, z_accel, u[0:3], dt
            )
            
            accel_pred, _ = self.learned_ekf.ekf.dynamics.measurement_model(ekf_state.x)
            loss = jnp.mean((accel_pred - z_accel) ** 2)
            
            gyro_corr, accel_corr = self.learned_ekf.model.apply(params, ekf_state.x)
            reg_loss = 0.001 * (jnp.sum(gyro_corr ** 2) + jnp.sum(accel_corr ** 2))
            
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
    
    def train(self, dataset, num_epochs: int = 100, 
              batch_size: int = 32):
        """Full training loop"""
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            batches = self._create_batches(dataset, batch_size)
            
            with tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch in pbar:
                    self.state, loss = self.train_step(self.state, batch)
                    epoch_loss += loss
                    num_batches += 1
                    
                    pbar.set_postfix({"loss": loss})
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        return losses
    
    def _create_batches(self, dataset, batch_size):
        """Create training batches from dataset"""
        states, measurements, targets = dataset
        
        n_samples = states.shape[0]
        indices = np.arange(n_samples - batch_size)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples - batch_size, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_states = jnp.stack([states[i] for i in batch_indices])
            batch_meas = jnp.stack([measurements[i] for i in batch_indices])
            batch_targets = jnp.stack([targets[i] for i in batch_indices])
            
            yield (batch_states, batch_meas, batch_targets)