import smbus2
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class IMUConfig:
    GYRO_SCALE: float = 131.0       
    ACCEL_SCALE: float = 16384.0    
    SMPLRT_DIV: int = 7             # Sample rate = 1kHz / (1 + SMPLRT_DIV)
    CALIBRATION_SAMPLES: int = 1000

class MPU6050:
    def __init__(self, bus_num: int = 1, address: int = 0x68):
        self.bus = smbus2.SMBus(bus_num)
        self.address = address
        self.config = IMUConfig()
        
        # Initialize sensor
        self._write_byte(0x6B, 0x00) 
        self._write_byte(0x1B, 0x00)  
        self._write_byte(0x1C, 0x00)  
        self._write_byte(0x19, self.config.SMPLRT_DIV)
        
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
    
    def _write_byte(self, reg: int, value: int):
        self.bus.write_byte_data(self.address, reg, value)
    
    def _read_word(self, reg: int) -> int:
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        value = (high << 8) | low
        if value >= 0x8000:  # signed conversion
            value -= 0x10000
        return value
    
    def calibrate(self, samples: int = None):
        samples = samples or self.config.CALIBRATION_SAMPLES
        gyro_sum = np.zeros(3)
        accel_sum = np.zeros(3)
        
        print("Calibrating IMU... Keep sensor stationary.")
        for _ in range(samples):
            g, a = self.read_raw()
            gyro_sum += g
            accel_sum += a
            time.sleep(0.001)
        
        self.gyro_bias = gyro_sum / samples
        self.accel_bias = (accel_sum / samples) - np.array([0, 0, self.config.ACCEL_SCALE])
        print(f"Gyro bias: {self.gyro_bias}")
        print(f"Accel bias: {self.accel_bias}")
    
    def read_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        accel = np.array([
            self._read_word(0x3B),
            self._read_word(0x3D),
            self._read_word(0x3F)
        ], dtype=np.float32)
        
        gyro = np.array([
            self._read_word(0x43),
            self._read_word(0x45),
            self._read_word(0x47)
        ], dtype=np.float32)
        
        return gyro, accel
    
    def read_calibrated(self) -> Tuple[np.ndarray, np.ndarray, float]:
        gyro_raw, accel_raw = self.read_raw()
        gyro_cal = (gyro_raw - self.gyro_bias) / self.config.GYRO_SCALE
        accel_cal = (accel_raw - self.accel_bias) / self.config.ACCEL_SCALE
        
        gyro_rad = np.deg2rad(gyro_cal)
        accel_ms2 = accel_cal * 9.80665
        
        return gyro_rad, accel_ms2, time.monotonic()
