import jax
import jax.numpy as jnp
import time
from queue import Queue
from threading import Thread
import matplotlib.pyplot as plt
from drivers.mpu6050 import MPU6050
from models.bias_net import LearnedEKF
from estimator.ekf import EKFState

class RealTimeEstimator:
    """Real-time state estimation on Raspberry Pi"""
    
    def __init__(self, learned_ekf: LearnedEKF, model_params):
        self.learned_ekf = learned_ekf
        self.params = model_params
        self.imu = MPU6050()
        self.imu.calibrate()
        
        x0 = jnp.array([1.0, 0.0, 0.0, 0.0,  
                        0.0, 0.0, 0.0,       
                        0.0, 0.0, 0.0,      
                        0.0, 0.0, 0.0])      
        P0 = jnp.eye(13) * 1e-4
        
        self.state = EKFState(x0, P0, time.monotonic())
        
        self.estimates = []
        self.timestamps = []
        
        self._step_jit = jax.jit(self._step)
    
    def _step(self, state, gyro, accel, dt):
        """Single estimation step (JIT compiled)"""
        u = jnp.concatenate([gyro, accel])
        return self.learned_ekf.step(
            self.params, state, u, accel, gyro, dt
        )
    
    def run(self, duration: float = 30.0, 
            visualize: bool = True):
        """Run real-time estimation"""
        print("Starting real-time estimation...")
        print("Press Ctrl+C to stop")
        
        start_time = time.monotonic()
        last_time = start_time
        
        try:
            while time.monotonic() - start_time < duration:
                gyro, accel, timestamp = self.imu.read_calibrated()
                
                current_time = timestamp
                dt = current_time - last_time
                
                gyro_jax = jnp.array(gyro)
                accel_jax = jnp.array(accel)
                
                self.state = self._step_jit(
                    self.state, gyro_jax, accel_jax, dt
                )
                
                self.estimates.append(self.state.x.copy())
                self.timestamps.append(current_time)
                
                if len(self.timestamps) % 10 == 0:
                    q = self.state.x[0:4]
                    euler = self._quaternion_to_euler(q)
                    print(f"Roll: {euler[0]:.2f}°, "
                          f"Pitch: {euler[1]:.2f}°, "
                          f"Yaw: {euler[2]:.2f}°")
                
                last_time = current_time
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\nStopping estimation...")
        
        finally:
            if visualize:
                self._visualize()
    
    def _quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        w, x, y, z = q
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        sinp = jnp.clip(sinp, -1.0, 1.0)
        pitch = jnp.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)
        
        return jnp.degrees(jnp.array([roll, pitch, yaw]))
    
    def _visualize(self):
        """Visualize estimation results"""
        estimates = jnp.stack(self.estimates)
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        eulers = jnp.array([self._quaternion_to_euler(q) 
                           for q in estimates[:, 0:4]])
        
        axes[0, 0].plot(self.timestamps, eulers[:, 0], label='Roll')
        axes[0, 0].plot(self.timestamps, eulers[:, 1], label='Pitch')
        axes[0, 0].plot(self.timestamps, eulers[:, 2], label='Yaw')
        axes[0, 0].set_title('Orientation (Euler Angles)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Angle (deg)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot angular velocity
        axes[0, 1].plot(self.timestamps, estimates[:, 4], label='ωx')
        axes[0, 1].plot(self.timestamps, estimates[:, 5], label='ωy')
        axes[0, 1].plot(self.timestamps, estimates[:, 6], label='ωz')
        axes[0, 1].set_title('Angular Velocity')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('rad/s')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot gyro bias estimates
        axes[1, 0].plot(self.timestamps, estimates[:, 7], label='bgx')
        axes[1, 0].plot(self.timestamps, estimates[:, 8], label='bgy')
        axes[1, 0].plot(self.timestamps, estimates[:, 9], label='bgz')
        axes[1, 0].set_title('Gyroscope Bias Estimates')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Bias (rad/s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot accel bias estimates
        axes[1, 1].plot(self.timestamps, estimates[:, 10], label='bax')
        axes[1, 1].plot(self.timestamps, estimates[:, 11], label='bay')
        axes[1, 1].plot(self.timestamps, estimates[:, 12], label='baz')
        axes[1, 1].set_title('Accelerometer Bias Estimates')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Bias (m/s²)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot quaternion norm (should be ~1)
        q_norm = jnp.linalg.norm(estimates[:, 0:4], axis=1)
        axes[2, 0].plot(self.timestamps, q_norm)
        axes[2, 0].axhline(y=1.0, color='r', linestyle='--')
        axes[2, 0].set_title('Quaternion Norm')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Norm')
        axes[2, 0].grid(True)
        
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('estimation_results.png', dpi=150)
        print("Results saved to estimation_results.png")
        
        if not plt.get_backend().startswith('agg'):
            plt.show()