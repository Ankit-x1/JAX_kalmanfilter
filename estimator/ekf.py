import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Tuple, NamedTuple

class EKFState(NamedTuple):
    """Extended Kalman Filter state"""
    x: jnp.ndarray  
    P: jnp.ndarray  
    t: float        

class DifferentiableEKF:
    """Fully differentiable EKF implementation in JAX"""
    
    def __init__(self, dynamics, Q: jnp.ndarray, R: jnp.ndarray):
        """
        Args:
            dynamics: Dynamics model instance
            Q: Process noise covariance
            R: Measurement noise covariance
        """
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
        
    @jit
    def predict(self, state: EKFState, u: jnp.ndarray, dt: float) -> EKFState:
        """Prediction step"""
        x, P, t = state
        
        f = self.dynamics.dynamics(x, u, dt)
        
       
        x_pred = x + f * dt
        
        F = jax.jacfwd(lambda x: self.dynamics.dynamics(x, u, dt))(x)
        
        Q_d = self.Q * dt
        
        P_pred = F @ P @ F.T + Q_d
        
        return EKFState(x_pred, P_pred, t + dt)
    
    @jit
    def update_accel(self, state: EKFState, z_accel: jnp.ndarray) -> EKFState:
        """Update step using accelerometer measurement"""
        x, P, t = state
        
        h_accel, _ = self.dynamics.measurement_model(x)
        
        H = jax.jacfwd(lambda x: self.dynamics.measurement_model(x)[0])(x)
        
        y = z_accel - h_accel
        
        S = H @ P @ H.T + self.R[:3, :3]  # R for accelerometer
        
        K = P @ H.T @ jnp.linalg.inv(S)
        
        x_new = x + K @ y
        
        I = jnp.eye(P.shape[0])
        P_new = (I - K @ H) @ P
        
        return EKFState(x_new, P_new, t)
    
    @jit
    def update_gyro(self, state: EKFState, z_gyro: jnp.ndarray) -> EKFState:
        """Update step using gyroscope measurement"""
        x, P, t = state
        
        _, h_gyro = self.dynamics.measurement_model(x)
        
        H = jax.jacfwd(lambda x: self.dynamics.measurement_model(x)[1])(x)
        
        y = z_gyro - h_gyro
        
        S = H @ P @ H.T + self.R[3:, 3:]  # R for gyroscope
        
        K = P @ H.T @ jnp.linalg.inv(S)
        
        x_new = x + K @ y
        
        I = jnp.eye(P.shape[0])
        P_new = (I - K @ H) @ P
        
        return EKFState(x_new, P_new, t)