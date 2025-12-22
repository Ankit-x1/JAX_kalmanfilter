import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from typing import Tuple

@jit
def quaternion_multiply(q: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product of two quaternions"""
    return jnp.array([
        q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
        q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],
        q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],
        q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]
    ])

@jit
def quaternion_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """Quaternion conjugate"""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

@jit
def quaternion_from_axis_angle(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Create quaternion from axis-angle representation"""
    axis = axis / (jnp.linalg.norm(axis) + 1e-8)
    s = jnp.sin(angle / 2.0)
    return jnp.array([
        jnp.cos(angle / 2.0),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s
    ])

@jit
def quaternion_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion to 3x3 rotation matrix"""
    q0, q1, q2, q3 = q
    return jnp.array([
        [1 - 2*q2*q2 - 2*q3*q3, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1*q1 - 2*q3*q3, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1*q1 - 2*q2*q2]
    ])

class RigidBodyDynamics:
    """Continuous-time rigid body dynamics with quaternions"""
    
    def __init__(self, g: float = 9.80665):
        self.g = g
        self.gravity_vector = jnp.array([0.0, 0.0, g])
        
    @jit
    def omega_matrix(self, omega: jnp.ndarray) -> jnp.ndarray:
        """Skew-symmetric matrix for quaternion derivative"""
        wx, wy, wz = omega
        return 0.5 * jnp.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
    
    @jit
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Continuous-time dynamics: dx/dt = f(x, u)
        
        State x: [q0, q1, q2, q3, ωx, ωy, ωz, bgx, bgy, bgz, bax, bay, baz]
                   0   1   2   3   4   5   6   7    8    9    10   11   12
        Input u: [ω_meas_x, ω_meas_y, ω_meas_z, a_meas_x, a_meas_y, a_meas_z]
        """
        # Unpack state
        q = x[0:4]
        omega = x[4:7]
        gyro_bias = x[7:10]
        accel_bias = x[10:13]
        
        # Unpack measurements
        omega_meas = u[0:3]
        accel_meas = u[3:6]
        
        # Gyroscope model: ω_true = ω_meas - bias
        omega_true = omega_meas - gyro_bias
        
        # Quaternion kinematics: dq/dt = 0.5 * Ω(ω_true) * q
        q_dot = self.omega_matrix(omega_true) @ q
        
        # Simple gyro dynamics: dω/dt = 0 (assume constant between measurements)
        omega_dot = jnp.zeros_like(omega)
        
        # Bias modeled as random walk: db/dt = 0 + noise
        gyro_bias_dot = jnp.zeros_like(gyro_bias)
        accel_bias_dot = jnp.zeros_like(accel_bias)
        
        # Combine derivatives
        x_dot = jnp.concatenate([q_dot, omega_dot, gyro_bias_dot, accel_bias_dot])
        
        # Return state derivative
        return x_dot
    
    @jit
    def measurement_model(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Measurement models for accelerometer and gyroscope
        
        Returns: (accel_predicted, gyro_predicted)
        """
        q = x[0:4]
        omega = x[4:7]
        gyro_bias = x[7:10]
        accel_bias = x[10:13]
        
        # Rotate gravity vector to body frame
        R = quaternion_to_rotation_matrix(q)
        gravity_body = R.T @ self.gravity_vector
        
        # Accelerometer model: a_meas = gravity_body + bias + noise
        accel_pred = gravity_body + accel_bias
        
        # Gyroscope model: ω_meas = ω_true + bias + noise
        gyro_pred = omega + gyro_bias
        
        return accel_pred, gyro_pred