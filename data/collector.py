import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from drivers.mpu6050 import MPU6050
from utils.logging_config import ComponentLogger
from config.config_manager import config_manager


class DataCollector:
    """Professional IMU data collection with metadata"""

    def __init__(self, output_dir: str = None):
        self.logger = ComponentLogger("collector")
        self.config = config_manager.get_config()

        output_dir = output_dir or "data/raw"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.imu = MPU6050()
            self.imu.calibrate()
            self.logger.info("DataCollector initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize DataCollector: {e}")
            raise

    def collect_stationary(self, duration: float = 60.0, name: str = "stationary"):
        return self._collect(duration, name, "stationary")

    def collect_rotation(self, duration: float = 30.0, name: str = "rotation"):
        return self._collect(duration, name, "rotation")

    def collect_trajectory(self, duration: float = 120.0, name: str = "trajectory"):
        return self._collect(duration, name, "trajectory")

    def _collect(self, duration: float, name: str, motion_type: str):
        sample_rate = int(1.0 / self.config.imu.TIMEOUT)  # Hz
        n_samples = int(duration * sample_rate)

        data = {
            "metadata": {
                "name": name,
                "motion_type": motion_type,
                "duration": duration,
                "sample_rate": sample_rate,
                "timestamp": datetime.now().isoformat(),
            },
            "samples": [],
        }

        self.logger.info(f"Collecting {name} data for {duration}s...")

        for i in range(n_samples):
            gyro, accel, timestamp = self.imu.read_calibrated()

            data["samples"].append(
                {"t": float(timestamp), "gyro": gyro.tolist(), "accel": accel.tolist()}
            )

            if i % (n_samples // 10) == 0:
                progress = (i / n_samples) * 100
                self.logger.debug(f"Progress: {i}/{n_samples} ({progress:.1f}%)")

            time.sleep(1.0 / sample_rate)

        filename = (
            self.output_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved {len(data['samples'])} samples to {filename}")
        return filename
