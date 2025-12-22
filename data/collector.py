import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from drivers.mpu6050 import MPU6050

class DataCollector:
    """Synchronized IMU data collection with metadata"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.imu = MPU6050()
        self.imu.calibrate()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_stationary(self, duration: float = 60.0, name: str = "stationary"):
        return self._collect(duration, name, "stationary")
    
    def collect_rotation(self, duration: float = 30.0, name: str = "rotation"):
        return self._collect(duration, name, "rotation")
    
    def collect_trajectory(self, duration: float = 120.0, name: str = "trajectory"):
        return self._collect(duration, name, "trajectory")
    
    def _collect(self, duration: float, name: str, motion_type: str):
        sample_rate = 100  # Hz
        n_samples = int(duration * sample_rate)
        
        data = {
            'metadata': {
                'name': name,
                'motion_type': motion_type,
                'duration': duration,
                'sample_rate': sample_rate,
                'timestamp': datetime.now().isoformat()
            },
            'samples': []
        }
        
        print(f"Collecting {name} data for {duration}s...")
        
        for i in range(n_samples):
            gyro, accel, timestamp = self.imu.read_calibrated()
            
            data['samples'].append({
                't': float(timestamp),
                'gyro': gyro.tolist(),
                'accel': accel.tolist()
            })
            
            if i % (n_samples // 10) == 0:
                print(f"  Progress: {i}/{n_samples} ({(i/n_samples)*100:.1f}%)")
            
            time.sleep(1.0 / sample_rate)
        
        filename = self.output_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data['samples'])} samples to {filename}")
        return filename
