"""
Professional real-world testing environment
"""

import time
import signal
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json

from utils.logging_config import ComponentLogger
from utils.exceptions import HardwareError, ConfigurationError
from config.config_manager import config_manager
from drivers.mpu6050 import MPU6050
from deployment.realtime import RealTimeEstimator


@dataclass
class TestConfig:
    """Test configuration parameters"""

    DURATION: float = 300.0  # 5 minutes default
    SAMPLE_RATE: int = 100  # Hz
    LOG_INTERVAL: int = 100  # Log every N samples
    HEALTH_CHECK_INTERVAL: int = 50  # Health check interval
    MAX_ERRORS: int = 10  # Max consecutive errors before abort
    OUTPUT_DIR: str = "test_results"


class TestEnvironment:
    """Professional testing environment for real hardware"""

    def __init__(self, test_config: Optional[TestConfig] = None):
        self.logger = ComponentLogger("test_env")
        self.test_config = test_config or TestConfig()
        self.config = config_manager.get_config()

        # Test state
        self._running = False
        self._shutdown_event = threading.Event()
        self._test_results = {}
        self._error_count = 0
        self._sample_count = 0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Create output directory
        self.output_dir = Path(self.test_config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Test interrupted by signal {signum}")
        self._shutdown_event.set()

    def run_hardware_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive hardware diagnostics"""
        self.logger.info("Starting hardware diagnostics...")

        diagnostics = {
            "imu_connection": False,
            "i2c_bus": False,
            "sensor_calibration": False,
            "data_quality": False,
            "errors": [],
        }

        try:
            # Test I2C bus communication
            self.logger.info("Testing I2C bus...")
            imu = MPU6050()
            diagnostics["imu_connection"] = True
            diagnostics["i2c_bus"] = True

            # Test sensor calibration
            self.logger.info("Testing sensor calibration...")
            if imu.calibrate(samples=100):
                diagnostics["sensor_calibration"] = True

            # Test data quality
            self.logger.info("Testing data quality...")
            valid_samples = 0
            for i in range(50):
                try:
                    gyro, accel, timestamp = imu.read_calibrated()
                    if not (any(g for g in gyro) or any(a for a in accel)):
                        continue
                    valid_samples += 1
                    time.sleep(0.01)
                except Exception as e:
                    diagnostics["errors"].append(f"Sample {i}: {e}")

            if valid_samples >= 40:  # 80% success rate
                diagnostics["data_quality"] = True

            # Close connection
            imu.bus.close()

        except Exception as e:
            diagnostics["errors"].append(f"Hardware test failed: {e}")
            self.logger.error(f"Hardware diagnostics failed: {e}")

        self._save_diagnostics(diagnostics)
        return diagnostics

    def run_performance_test(self, estimator: RealTimeEstimator) -> Dict[str, Any]:
        """Run performance benchmarking test"""
        self.logger.info("Starting performance test...")

        performance = {
            "start_time": time.time(),
            "samples_processed": 0,
            "processing_times": [],
            "memory_usage": [],
            "errors": 0,
            "avg_processing_time": 0.0,
            "max_processing_time": 0.0,
            "min_processing_time": float("inf"),
        }

        start_time = time.monotonic()
        last_time = start_time

        while not self._shutdown_event.is_set() and (
            time.monotonic() - start_time < self.test_config.DURATION
        ):
            try:
                loop_start = time.perf_counter()

                # Simulate one estimation step
                gyro, accel, timestamp = estimator.imu.read_calibrated()

                processing_time = time.perf_counter() - loop_start
                performance["processing_times"].append(processing_time)
                performance["samples_processed"] += 1

                # Update stats
                if processing_time > performance["max_processing_time"]:
                    performance["max_processing_time"] = processing_time
                if processing_time < performance["min_processing_time"]:
                    performance["min_processing_time"] = processing_time

                # Log progress
                if (
                    performance["samples_processed"] % self.test_config.LOG_INTERVAL
                    == 0
                ):
                    self.logger.info(
                        f"Processed {performance['samples_processed']} samples"
                    )

                # Target sample rate
                target_dt = 1.0 / self.test_config.SAMPLE_RATE
                elapsed = time.monotonic() - last_time
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

                last_time = time.monotonic()

            except Exception as e:
                performance["errors"] += 1
                self._error_count += 1

                if self._error_count >= self.test_config.MAX_ERRORS:
                    self.logger.error("Too many errors, aborting test")
                    break

                time.sleep(0.1)  # Backoff

        # Calculate final statistics
        if performance["processing_times"]:
            performance["avg_processing_time"] = sum(
                performance["processing_times"]
            ) / len(performance["processing_times"])

        performance["end_time"] = time.time()
        performance["duration"] = performance["end_time"] - performance["start_time"]
        performance["samples_per_second"] = (
            performance["samples_processed"] / performance["duration"]
        )

        self._save_performance(performance)
        return performance

    def run_endurance_test(self, estimator: RealTimeEstimator) -> Dict[str, Any]:
        """Run long-duration endurance test"""
        self.logger.info("Starting endurance test...")

        endurance = {
            "start_time": time.time(),
            "health_checks": [],
            "sensor_health": True,
            "memory_growth": [],
            "recovery_events": 0,
            "total_runtime": 0.0,
        }

        start_time = time.monotonic()
        check_interval = 10.0  # Health check every 10 seconds
        last_health_check = start_time

        while not self._shutdown_event.is_set() and (
            time.monotonic() - start_time < self.test_config.DURATION
        ):
            try:
                # Normal operation
                gyro, accel, timestamp = estimator.imu.read_calibrated()

                # Periodic health checks
                if time.monotonic() - last_health_check >= check_interval:
                    health = estimator.imu.health_check()
                    endurance["health_checks"].append(
                        {"timestamp": time.time(), "status": health}
                    )

                    if not health["sensor_connected"]:
                        endurance["sensor_health"] = False
                        self.logger.warning("Sensor disconnected during endurance test")

                    last_health_check = time.monotonic()

                time.sleep(0.01)

            except Exception as e:
                self.logger.warning(f"Endurance test error: {e}")
                endurance["recovery_events"] += 1
                time.sleep(1.0)  # Recovery delay

        endurance["end_time"] = time.time()
        endurance["total_runtime"] = endurance["end_time"] - endurance["start_time"]

        self._save_endurance(endurance)
        return endurance

    def _save_diagnostics(self, diagnostics: Dict[str, Any]):
        """Save diagnostic results"""
        timestamp = int(time.time())
        file_path = self.output_dir / f"diagnostics_{timestamp}.json"

        with open(file_path, "w") as f:
            json.dump(diagnostics, f, indent=2, default=str)

        self.logger.info(f"Diagnostics saved to {file_path}")

    def _save_performance(self, performance: Dict[str, Any]):
        """Save performance results"""
        timestamp = int(time.time())
        file_path = self.output_dir / f"performance_{timestamp}.json"

        with open(file_path, "w") as f:
            json.dump(performance, f, indent=2, default=str)

        self.logger.info(f"Performance results saved to {file_path}")

    def _save_endurance(self, endurance: Dict[str, Any]):
        """Save endurance test results"""
        timestamp = int(time.time())
        file_path = self.output_dir / f"endurance_{timestamp}.json"

        with open(file_path, "w") as f:
            json.dump(endurance, f, indent=2, default=str)

        self.logger.info(f"Endurance results saved to {file_path}")

    def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite"""
        self.logger.info("=== Starting Complete Test Suite ===")

        results = {
            "test_start": time.time(),
            "diagnostics": None,
            "performance": None,
            "endurance": None,
            "overall_status": "UNKNOWN",
        }

        try:
            # Phase 1: Hardware Diagnostics
            results["diagnostics"] = self.run_hardware_diagnostics()

            if not all(
                [
                    results["diagnostics"]["imu_connection"],
                    results["diagnostics"]["data_quality"],
                ]
            ):
                self.logger.error("Hardware diagnostics failed - aborting test suite")
                results["overall_status"] = "FAILED"
                return results

            # Phase 2: Performance Testing
            try:
                # Setup estimator for testing
                from models.bias_net import BiasCorrectionNet, LearnedEKF
                from dynamics.rigid_body import RigidBodyDynamics
                from estimator.ekf import DifferentiableEKF

                # Use a simple test model
                dynamics = RigidBodyDynamics()
                ekf = DifferentiableEKF(dynamics)
                bias_net = BiasCorrectionNet()
                learned_ekf = LearnedEKF(ekf, bias_net)

                # Initialize proper model parameters for testing
                import jax
                import jax.numpy as jnp

                test_key = jax.random.PRNGKey(int(time.time()) % 2**32)
                init_state = jnp.array(
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                )
                test_params = learned_ekf.model.init(test_key, init_state)

                estimator = RealTimeEstimator(learned_ekf, test_params)
                results["performance"] = self.run_performance_test(estimator)

            except Exception as e:
                self.logger.error(f"Performance test failed: {e}")
                results["performance"] = {"error": str(e)}

            # Phase 3: Endurance Testing
            if results["performance"] and "error" not in results["performance"]:
                try:
                    results["endurance"] = self.run_endurance_test(estimator)
                except Exception as e:
                    self.logger.error(f"Endurance test failed: {e}")
                    results["endurance"] = {"error": str(e)}

            # Determine overall status
            if (
                results["diagnostics"]["sensor_health"]
                and results["performance"]
                and "error" not in results["performance"]
                and results["performance"]["samples_per_second"] > 50
            ):
                results["overall_status"] = "PASSED"
            else:
                results["overall_status"] = "FAILED"

        except Exception as e:
            self.logger.error(f"Test suite error: {e}")
            results["overall_status"] = "ERROR"

        results["test_end"] = time.time()
        results["total_duration"] = results["test_end"] - results["test_start"]

        # Save complete results
        timestamp = int(time.time())
        file_path = self.output_dir / f"complete_test_{timestamp}.json"
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"=== Test Suite Complete: {results['overall_status']} ===")
        self.logger.info(f"Results saved to {file_path}")

        return results


def run_tests():
    """Main test runner"""
    test_env = TestEnvironment()
    return test_env.run_complete_test_suite()
