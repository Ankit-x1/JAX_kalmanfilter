import jax
import jax.numpy as jnp
import time
import signal
import atexit
from queue import Queue
from threading import Thread, Event
import matplotlib.pyplot as plt
from drivers.mpu6050 import MPU6050
from models.bias_net import LearnedEKF
from estimator.ekf import EKFState
from utils.logging_config import ComponentLogger
from utils.exceptions import SensorError, EstimationError
from config.config_manager import config_manager


class RealTimeEstimator:
    """Real-time state estimation with professional error handling"""

    def __init__(self, learned_ekf: LearnedEKF, model_params):
        self.logger = ComponentLogger("deployment")
        self.learned_ekf = learned_ekf
        self.params = model_params
        self.config = config_manager.get_config()

        # Shutdown handling
        self._shutdown_event = Event()
        self._running = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)

        try:
            self.imu = MPU6050()
            self.imu.calibrate()
            self.logger.info("IMU initialization successful")
        except Exception as e:
            self.logger.error(f"IMU initialization failed: {e}")
            raise

        x0 = jnp.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        P0 = jnp.eye(13) * self.config.ekf.INITIAL_COVARIANCE

        self.state = EKFState(x0, P0, time.monotonic())

        self.estimates = []
        self.timestamps = []
        self._max_estimates = 10000  # Prevent memory leaks

        self._step_jit = jax.jit(self._step)
        self.logger.info("RealTimeEstimator initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self._shutdown_event.set()
    
    def _cleanup(self):
        """Cleanup resources"""
        if self._running:
            self.logger.info("Performing cleanup...")
            self._running = False
            try:
                if hasattr(self.imu, 'bus'):
                    self.imu.bus.close()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
    
    def _step(self, state, gyro, accel, dt):
        """Single estimation step with validation"""
        try:
            u = jnp.concatenate([gyro, accel])
            new_state = self.learned_ekf.step(
                self.params, state, u, accel, gyro, dt
            )
            
            # Validate new state
            if jnp.any(jnp.isnan(new_state.x)) or jnp.any(jnp.isinf(new_state.x)):
                raise EstimationError("Invalid state produced")
            
            return new_state
        except Exception as e:
            self.logger.error(f"Estimation step failed: {e}")
            return state  # Return previous state on failure

    def run(self, duration: float = 30.0, visualize: bool = True):
        """Run real-time estimation for specified duration"""
        self._running = True
        self.logger.info(f"Starting real-time estimation for {duration}s")
        
        start_time = time.monotonic()
        last_time = start_time
        
        try:
            while self._running and (time.monotonic() - start_time < duration):
                if self._shutdown_event.is_set():
                    break
                
                current_time = time.monotonic()
                dt = current_time - last_time
                
                if dt < 0.001:  # Minimum dt to avoid division issues
                    time.sleep(0.001)
                    continue
                
                try:
                    gyro, accel, timestamp = self.imu.read_calibrated()
                    
                    self.state = self._step_jit(self.state, gyro, accel, dt)
                    
                    # Store estimates
                    if len(self.estimates) < self._max_estimates:
                        self.estimates.append(self.state.x)
                        self.timestamps.append(current_time - start_time)
                    
                except SensorError as e:
                    self.logger.warning(f"Sensor read failed: {e}")
                    time.sleep(0.01)
                    continue
                
                last_time = current_time
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
        
        finally:
            self._running = False
            self.logger.info("Estimation stopped")
            
            if visualize and self.estimates:
                try:
                    self._visualize()
                except Exception as e:
                    self.logger.error(f"Visualization failed: {e}")

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

        eulers = jnp.array([self._quaternion_to_euler(q) for q in estimates[:, 0:4]])

        axes[0, 0].plot(self.timestamps, eulers[:, 0], label="Roll")
        axes[0, 0].plot(self.timestamps, eulers[:, 1], label="Pitch")
        axes[0, 0].plot(self.timestamps, eulers[:, 2], label="Yaw")
        axes[0, 0].set_title("Orientation (Euler Angles)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Angle (deg)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot angular velocity
        axes[0, 1].plot(self.timestamps, estimates[:, 4], label="ωx")
        axes[0, 1].plot(self.timestamps, estimates[:, 5], label="ωy")
        axes[0, 1].plot(self.timestamps, estimates[:, 6], label="ωz")
        axes[0, 1].set_title("Angular Velocity")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("rad/s")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot gyro bias estimates
        axes[1, 0].plot(self.timestamps, estimates[:, 7], label="bgx")
        axes[1, 0].plot(self.timestamps, estimates[:, 8], label="bgy")
        axes[1, 0].plot(self.timestamps, estimates[:, 9], label="bgz")
        axes[1, 0].set_title("Gyroscope Bias Estimates")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Bias (rad/s)")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot accel bias estimates
        axes[1, 1].plot(self.timestamps, estimates[:, 10], label="bax")
        axes[1, 1].plot(self.timestamps, estimates[:, 11], label="bay")
        axes[1, 1].plot(self.timestamps, estimates[:, 12], label="baz")
        axes[1, 1].set_title("Accelerometer Bias Estimates")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Bias (m/s²)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Plot quaternion norm (should be ~1)
        q_norm = jnp.linalg.norm(estimates[:, 0:4], axis=1)
        axes[2, 0].plot(self.timestamps, q_norm)
        axes[2, 0].axhline(y=1.0, color="r", linestyle="--")
        axes[2, 0].set_title("Quaternion Norm")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_ylabel("Norm")
        axes[2, 0].grid(True)

        axes[2, 1].axis("off")

        plt.tight_layout()
        plt.savefig("estimation_results.png", dpi=150)
        self.logger.info("Results saved to estimation_results.png")

        if not plt.get_backend().startswith("agg"):
            plt.show()
