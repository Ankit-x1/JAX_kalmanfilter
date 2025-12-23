import jax
import jax.numpy as jnp
from jax import jit
from flax import linen as nn
from typing import Tuple
from estimator.ekf import DifferentiableEKF, EKFState

class BiasCorrectionNet(nn.Module):
    """Tiny MLP for bias and noise correction"""
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Input: state vector [13,]
        Output: (gyro_bias_correction [3,], accel_bias_correction [3,])
        """
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        
        gyro_correction = nn.Dense(3)(x) * 0.01
        accel_correction = nn.Dense(3)(x) * 0.01
        
        return gyro_correction, accel_correction

class LearnedEKF:
    """EKF with learned bias corrections"""
    
    def __init__(self, ekf: DifferentiableEKF, model: BiasCorrectionNet):
        self.ekf = ekf
        self.model = model
        
    def apply_corrections(self, params, state: EKFState, u: jnp.ndarray):
        """Apply neural corrections to measurements"""
        gyro_corr, accel_corr = self.model.apply(params, state.x)
        
        u_corrected = jnp.concatenate([
            u[0:3] + gyro_corr,  
            u[3:6] + accel_corr  
        ])
        
        return u_corrected
    
    @jit
    def step(self, params, state: EKFState, u: jnp.ndarray, 
             z_accel: jnp.ndarray, z_gyro: jnp.ndarray, dt: float) -> EKFState:
        """Single filter step with learned corrections"""
        u_corrected = self.apply_corrections(params, state, u)
        
        state_pred = self.ekf.predict(state, u_corrected, dt)
        
        state_accel = self.ekf.update_accel(state_pred, z_accel)
        
        state_final = self.ekf.update_gyro(state_accel, z_gyro)
        
        return state_final