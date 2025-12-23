try:
    import smbus2
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False
    smbus2 = None

import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from utils.logging_config import ComponentLogger
from utils.exceptions import I2CError, CalibrationError, SensorError
from config.config_manager import config_manager
from drivers.mpu6050_constants import (
    REG_PWR_MGMT_1, REG_GYRO_CONFIG, REG_ACCEL_CONFIG, REG_SMPLRT_DIV,
    REG_ACCEL_XOUT_H, REG_ACCEL_YOUT_H, REG_ACCEL_ZOUT_H,
    REG_GYRO_XOUT_H, REG_GYRO_YOUT_H, REG_GYRO_ZOUT_H,
    MPU6050_ADDRESS, WAKE_UP_VALUE, GYRO_CONFIG_VALUE, ACCEL_CONFIG_VALUE,
    MAX_RAW_VALUE, MIN_VALID_SAMPLES, MAX_CONSECUTIVE_ERRORS, SENSOR_HEALTH_DEGRADED_THRESHOLD
)


class MPU6050:
    def __init__(self):
        self.logger = ComponentLogger("driver")
        self.config = config_manager.get_config().imu

        if not SMBUS_AVAILABLE:
            raise I2CError(
                "smbus2 not available. Install with: pip install smbus2 (Linux only)",
                details={"platform": "non-linux", "missing_dependency": "smbus2"}
            )

        try:
            self.bus = smbus2.SMBus(self.config.BUS_NUM)
        except Exception as e:
            raise I2CError(
                f"Failed to initialize I2C bus {self.config.BUS_NUM}: {e}",
                details={"bus_num": self.config.BUS_NUM},
            )

        self.address = MPU6050_ADDRESS
        self._initialize_sensor()

        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self._last_read_time: Optional[float] = None
        self._read_failures = 0
        self._max_failures = MAX_CONSECUTIVE_ERRORS

    def _initialize_sensor(self):
        """Initialize sensor with error handling"""
        try:
            self._write_byte_safe(REG_PWR_MGMT_1, WAKE_UP_VALUE)  # Wake up
            self._write_byte_safe(REG_GYRO_CONFIG, GYRO_CONFIG_VALUE)  # Gyro config
            self._write_byte_safe(REG_ACCEL_CONFIG, ACCEL_CONFIG_VALUE)  # Accel config
            self._write_byte_safe(REG_SMPLRT_DIV, self.config.SMPLRT_DIV)
            self.logger.info("MPU6050 initialized successfully")
        except Exception as e:
            raise I2CError(
                f"Failed to initialize MPU6050: {e}", details={"address": self.address}
            )

    def _write_byte_safe(self, reg: int, value: int):
        """Safe byte write with timeout and error handling"""
        try:
            self.bus.write_byte_data(self.address, reg, value)
        except OSError as e:
            raise I2CError(
                f"I2C write failed at register 0x{reg:02X}: {e}",
                details={"register": reg, "value": value, "address": self.address},
            )

    def _read_word_safe(self, reg: int) -> int:
        """Safe word read with timeout and error handling"""
        try:
            high = self.bus.read_byte_data(self.address, reg)
            low = self.bus.read_byte_data(self.address, reg + 1)
            value = (high << 8) | low
            if value >= 0x8000:
                value -= 0x10000
            return value
        except OSError as e:
            self._read_failures += 1
            if self._read_failures >= self._max_failures:
                raise I2CError(
                    f"I2C read failed after {self._read_failures} attempts: {e}",
                    details={"register": reg, "address": self.address},
                )
            return 0

    def calibrate(self, samples: int = None) -> bool:
        """Calibrate sensor with error handling"""
        samples = samples or self.config.CALIBRATION_SAMPLES
        gyro_sum = np.zeros(3)
        accel_sum = np.zeros(3)
        valid_samples = 0

        self.logger.info(f"Starting calibration with {samples} samples")

        for i in range(samples):
            try:
                g, a = self.read_raw()
                if np.any(np.isnan(g)) or np.any(np.isnan(a)):
                    continue
                gyro_sum += g
                accel_sum += a
                valid_samples += 1
                time.sleep(0.001)
            except Exception as e:
                self.logger.warning(f"Calibration sample {i} failed: {e}")
                continue

        if valid_samples < samples * MIN_VALID_SAMPLES:
            raise CalibrationError(
                f"Insufficient valid samples: {valid_samples}/{samples}"
            )

        self.gyro_bias = gyro_sum / valid_samples
        self.accel_bias = (accel_sum / valid_samples) - np.array(
            [0, 0, self.config.ACCEL_SCALE]
        )

        self.logger.info(f"Calibration complete. Valid samples: {valid_samples}")
        self.logger.debug(f"Gyro bias: {self.gyro_bias}")
        self.logger.debug(f"Accel bias: {self.accel_bias}")
        return True

    def read_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read raw sensor data with validation"""
        try:
            accel = np.array(
                [
                    self._read_word_safe(REG_ACCEL_XOUT_H),
                    self._read_word_safe(REG_ACCEL_YOUT_H),
                    self._read_word_safe(REG_ACCEL_ZOUT_H),
                ],
                dtype=np.float32,
            )

            gyro = np.array(
                [
                    self._read_word_safe(REG_GYRO_XOUT_H),
                    self._read_word_safe(REG_GYRO_YOUT_H),
                    self._read_word_safe(REG_GYRO_ZOUT_H),
                ],
                dtype=np.float32,
            )

            # Validate sensor data
            if np.any(np.abs(accel) > MAX_RAW_VALUE) or np.any(np.abs(gyro) > MAX_RAW_VALUE):
                raise SensorError("Sensor reading out of valid range")

            self._read_failures = max(0, self._read_failures - 1)
            return gyro, accel

        except Exception as e:
            self.logger.error(f"Raw sensor read failed: {e}")
            raise SensorError(f"Failed to read sensor data: {e}")

    def read_calibrated(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Read calibrated sensor data with health monitoring"""
        try:
            gyro_raw, accel_raw = self.read_raw()
            gyro_cal = (gyro_raw - self.gyro_bias) / self.config.GYRO_SCALE
            accel_cal = (accel_raw - self.accel_bias) / self.config.ACCEL_SCALE

            gyro_rad = np.deg2rad(gyro_cal)
            accel_ms2 = accel_cal * 9.80665

            current_time = time.monotonic()

            # Validate calibrated data
            if np.any(np.abs(gyro_rad) > 10) or np.any(np.abs(accel_ms2) > 50):
                self.logger.warning("Sensor data appears invalid")

            self._last_read_time = current_time
            return gyro_rad, accel_ms2, current_time

        except Exception as e:
            self.logger.error(f"Calibrated sensor read failed: {e}")
            raise SensorError(f"Failed to read calibrated data: {e}")

    def health_check(self) -> dict:
        """Perform sensor health check"""
        status = {
            "sensor_connected": True,
            "last_read_time": self._last_read_time,
            "read_failures": self._read_failures,
            "bias_stable": True,
        }

        if self._last_read_time and (time.monotonic() - self._last_read_time) > 1.0:
            status["sensor_connected"] = False

        if self._read_failures > SENSOR_HEALTH_DEGRADED_THRESHOLD:
            status["sensor_health"] = "degraded"
        else:
            status["sensor_health"] = "healthy"

        return status
