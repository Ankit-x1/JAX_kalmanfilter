import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Tuple, NamedTuple
from utils.logging_config import ComponentLogger
from utils.exceptions import NumericalInstabilityError, CovarianceError
from config.config_manager import config_manager


class EKFState(NamedTuple):
    """Extended Kalman Filter state"""

    x: jnp.ndarray
    P: jnp.ndarray
    t: float


class DifferentiableEKF:
    """Fully differentiable EKF implementation in JAX"""

    def __init__(self, dynamics):
        """
        Args:
            dynamics: Dynamics model instance
        """
        self.logger = ComponentLogger("estimator")
        self.dynamics = dynamics
        self.config = config_manager.get_config().ekf

        self.Q = jnp.diag(jnp.array(self.config.PROCESS_NOISE_DIAG))
        self.R = jnp.diag(jnp.array(self.config.MEASUREMENT_NOISE_DIAG))
        self.numerical_tolerance = self.config.NUMERICAL_TOLERANCE

    @jit
    def predict(self, state: EKFState, u: jnp.ndarray, dt: float) -> EKFState:
        """Prediction step with numerical stability checks"""
        x, P, t = state

        # Check state validity
        if jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)):
            raise NumericalInstabilityError("Invalid state vector in prediction")

        f = self.dynamics.dynamics(x, u, dt)
        x_pred = x + f * dt

        F = jax.jacfwd(lambda x: self.dynamics.dynamics(x, u, dt))(x)
        Q_d = self.Q * dt

        P_pred = F @ P @ F.T + Q_d

        # Ensure covariance matrix is symmetric and positive definite
        P_pred = 0.5 * (P_pred + P_pred.T)

        diag_P = jnp.diag(P_pred)
        if jnp.any(diag_P < self.numerical_tolerance):
            P_pred = P_pred + jnp.eye(P_pred.shape[0]) * self.numerical_tolerance

        return EKFState(x_pred, P_pred, t + dt)

    @jit
    def update_accel(self, state: EKFState, z_accel: jnp.ndarray) -> EKFState:
        """Update step using accelerometer measurement with robust inversion"""
        x, P, t = state

        h_accel, _ = self.dynamics.measurement_model(x)

        H = jax.jacfwd(lambda x: self.dynamics.measurement_model(x)[0])(x)

        y = z_accel - h_accel

        R_accel = self.R[:3, :3]
        S = H @ P @ H.T + R_accel

        # Use pseudo-inverse for numerical stability
        try:
            S_inv = jnp.linalg.inv(S)
        except jnp.linalg.LinAlgError:
            S_inv = jnp.linalg.pinv(S)

        # Check innovation for outliers
        innovation_magnitude = jnp.sqrt(jnp.sum(y**2))
        if innovation_magnitude > 3.0:  # 3-sigma threshold
            # Reduce Kalman gain for large innovations
            K = P @ H.T @ S_inv * 0.5
        else:
            K = P @ H.T @ S_inv

        x_new = x + K @ y

        # Joseph form update for numerical stability
        I = jnp.eye(P.shape[0])
        P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R_accel @ K.T

        return EKFState(x_new, P_new, t)

    @jit
    def update_gyro(self, state: EKFState, z_gyro: jnp.ndarray) -> EKFState:
        """Update step using gyroscope measurement with robust inversion"""
        x, P, t = state

        _, h_gyro = self.dynamics.measurement_model(x)

        H = jax.jacfwd(lambda x: self.dynamics.measurement_model(x)[1])(x)

        y = z_gyro - h_gyro

        R_gyro = self.R[3:, 3:]
        S = H @ P @ H.T + R_gyro

        # Use pseudo-inverse for numerical stability
        try:
            S_inv = jnp.linalg.inv(S)
        except jnp.linalg.LinAlgError:
            S_inv = jnp.linalg.pinv(S)

        # Check innovation for outliers
        innovation_magnitude = jnp.sqrt(jnp.sum(y**2))
        if innovation_magnitude > 3.0:
            K = P @ H.T @ S_inv * 0.5
        else:
            K = P @ H.T @ S_inv

        x_new = x + K @ y

        # Joseph form update
        I = jnp.eye(P.shape[0])
        P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R_gyro @ K.T

        return EKFState(x_new, P_new, t)
