"""
Data validation utilities for professional data handling
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional
from utils.exceptions import ConfigurationError
from utils.logging_config import ComponentLogger


class DataValidator:
    """Professional data validation and processing"""

    def __init__(self):
        self.logger = ComponentLogger("validator")

        # Validation thresholds
        self.GYRO_MAX = 10.0  # rad/s
        self.GYRO_MIN = -10.0
        self.ACCEL_MAX = 50.0  # m/s²
        self.ACCEL_MIN = -50.0
        self.MAX_NAN_RATIO = 0.1  # 10% NaN tolerance
        self.MIN_SAMPLES = 100

    def validate_json_structure(self, data: Dict[str, Any]) -> bool:
        """Validate JSON file structure"""
        try:
            required_keys = ["samples", "metadata"]
            if not all(key in data for key in required_keys):
                return False

            samples = data["samples"]
            if not isinstance(samples, list) or len(samples) == 0:
                return False

            # Check first sample structure
            sample_keys = ["timestamp", "gyro", "accel"]
            first_sample = samples[0]

            return all(key in first_sample for key in sample_keys)

        except Exception as e:
            self.logger.error(f"JSON validation error: {e}")
            return False

    def validate_sensor_reading(self, reading: np.ndarray, reading_type: str) -> bool:
        """Validate individual sensor reading"""
        if len(reading) != 3:
            return False

        if np.any(np.isnan(reading)) or np.any(np.isinf(reading)):
            return False

        if reading_type == "gyro":
            return np.all(reading >= self.GYRO_MIN) and np.all(reading <= self.GYRO_MAX)
        elif reading_type == "accel":
            return np.all(reading >= self.ACCEL_MIN) and np.all(
                reading <= self.ACCEL_MAX
            )

        return False

    def process_samples(
        self, raw_samples: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Process raw samples into training data"""
        if len(raw_samples) < self.MIN_SAMPLES:
            raise ConfigurationError(
                f"Insufficient samples: {len(raw_samples)} < {self.MIN_SAMPLES}"
            )

        gyro_data = []
        accel_data = []
        timestamps = []

        # Collect valid sensor data
        for i, sample in enumerate(raw_samples):
            try:
                gyro = np.array(sample["gyro"], dtype=np.float64)
                accel = np.array(sample["accel"], dtype=np.float64)

                if self.validate_sensor_reading(
                    gyro, "gyro"
                ) and self.validate_sensor_reading(accel, "accel"):
                    gyro_data.append(gyro)
                    accel_data.append(accel)
                    timestamps.append(sample.get("timestamp", i))

            except Exception as e:
                self.logger.warning(f"Skipping invalid sample {i}: {e}")
                continue

        if len(gyro_data) < self.MIN_SAMPLES:
            raise ConfigurationError(f"Insufficient valid samples after filtering")

        # Convert to arrays
        gyro_data = np.array(gyro_data)
        accel_data = np.array(accel_data)
        measurements = np.hstack([gyro_data, accel_data])

        # Generate states (simplified - in real deployment, would use ground truth)
        n_samples = len(gyro_data)
        states = np.zeros((n_samples, 13))
        states[:, 0:4] = 1.0  # Identity quaternion as initial guess

        # Integrate gyro data for angular velocity estimates
        if n_samples > 1:
            dt = np.diff(timestamps)
            states[1:, 4:7] = gyro_data[:-1]  # Angular velocity

        # Generate targets (for supervised learning)
        targets = states.copy()

        self.logger.info(f"Processed {n_samples} valid samples")

        return {
            "states": states,
            "measurements": measurements,
            "targets": targets,
            "timestamps": np.array(timestamps),
        }

    def validate_batch(
        self, states: np.ndarray, measurements: np.ndarray, targets: np.ndarray
    ) -> bool:
        """Validate complete training batch"""
        try:
            # Shape validation
            expected_state_dim = 13
            expected_measurement_dim = 6

            if (
                states.shape[1] != expected_state_dim
                or measurements.shape[1] != expected_measurement_dim
                or targets.shape[1] != expected_state_dim
            ):
                return False

            # Check for NaN/Inf
            for arr, name in [
                (states, "states"),
                (measurements, "measurements"),
                (targets, "targets"),
            ]:
                nan_ratio = np.isnan(arr).sum() / arr.size
                inf_count = np.isinf(arr).sum()

                if nan_ratio > self.MAX_NAN_RATIO:
                    self.logger.error(f"Too many NaN values in {name}: {nan_ratio:.2%}")
                    return False

                if inf_count > 0:
                    self.logger.error(f"Infinite values found in {name}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Batch validation error: {e}")
            return False
