#!/usr/bin/env python3
"""
Quick test script for MPU6050 connectivity and basic functionality
"""

import sys
import time
import numpy as np

def test_imu_basic():
    """Test basic IMU functionality"""
    print("=== MPU6050 Basic Test ===")
    
    try:
        from drivers.mpu6050 import MPU6050
        print(" MPU6050 module imported successfully")
    except ImportError as e:
        print(f" Failed to import MPU6050: {e}")
        print("Make sure you're on Raspberry Pi with smbus2 installed")
        return False
    
    try:
        # Initialize IMU
        print("Initializing MPU6050...")
        imu = MPU6050()
        print(" MPU6050 initialized")
        
        # Calibrate
        print("Calibrating... Keep sensor still!")
        imu.calibrate()
        print(" Calibration complete")
        
        # Read samples
        print("\nReading 10 samples:")
        print("Sample | Gyro (rad/s)              | Accel (m/s²)")
        print("-------|---------------------------|---------------------------")
        
        for i in range(10):
            gyro, accel, timestamp = imu.read_calibrated()
            
            gyro_str = f"[{gyro[0]:7.3f}, {gyro[1]:7.3f}, {gyro[2]:7.3f}]"
            accel_str = f"[{accel[0]:7.2f}, {accel[1]:7.2f}, {accel[2]:7.2f}]"
            
            print(f"  {i+1:2d}   | {gyro_str} | {accel_str}")
            
            time.sleep(0.1)
        
        # Test data ranges
        print("\n=== Data Validation ===")
        
        # Collect more samples for validation
        samples = []
        for _ in range(100):
            gyro, accel, _ = imu.read_calibrated()
            samples.append((gyro.copy(), accel.copy()))
            time.sleep(0.01)
        
        gyros = np.array([s[0] for s in samples])
        accels = np.array([s[1] for s in samples])
        
        print(f"Gyro range: [{gyros.min():.3f}, {gyros.max():.3f}] rad/s")
        print(f"Accel range: [{accels.min():.2f}, {accels.max():.2f}] m/s²")
        
        # Check if accelerometer detects gravity (~9.8 m/s²)
        accel_magnitude = np.linalg.norm(accels, axis=1).mean()
        if 8.0 <= accel_magnitude <= 11.0:
            print(f" Gravity detected: {accel_magnitude:.2f} m/s²")
        else:
            print(f" Unexpected gravity reading: {accel_magnitude:.2f} m/s²")
        
        # Check if gyroscope is stable when stationary
        gyro_std = np.std(gyros, axis=0)
        if np.all(gyro_std < 0.1):
            print(f" Gyroscope stable (std: {gyro_std})")
        else:
            print(f" Gyroscope unstable (std: {gyro_std})")
        
        print("\n All tests passed!")
        return True
        
    except Exception as e:
        print(f" Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_collection():
    """Test data collection functionality"""
    print("\n=== Data Collection Test ===")
    
    try:
        from data.collector import DataCollector
        
        print("Creating DataCollector...")
        collector = DataCollector(output_dir="test_data")
        
        # Test short collection
        print("Collecting 5 seconds of stationary data...")
        filename = collector.collect_stationary(duration=5.0, name="test_stationary")
        
        if filename.exists():
            print(f"✓ Data saved to {filename}")
            
            # Verify data format
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
            
            print(f"✓ Collected {len(data['samples'])} samples")
            print(f"  Sample rate: {data['metadata']['sample_rate']} Hz")
            print(f"  Duration: {data['metadata']['duration']} s")
            
            # Check first sample
            sample = data['samples'][0]
            print(f"  First sample: gyro={sample['gyro']}, accel={sample['accel']}")
            
            return True
        else:
            print("✗ Data file not created")
            return False
            
    except Exception as e:
        print(f"✗ Data collection error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Raspberry Pi MPU6050 Test Suite")
    print("================================")
    
    # Test 1: Basic IMU functionality
    imu_ok = test_imu_basic()
    
    if imu_ok:
        # Test 2: Data collection
        data_ok = test_data_collection()
        
        if data_ok:
            print("\n🎉 All tests passed! Ready for EKF training.")
            print("\nNext steps:")
            print("1. Run: python3 main.py --mode collect")
            print("2. Run: python3 main.py --mode train --epochs 20")
            print("3. Run: python3 main.py --mode deploy --duration 30")
        else:
            print("\n Data collection failed. Check file permissions.")
            sys.exit(1)
    else:
        print("\n IMU test failed. Check hardware connections.")
        print("\nTroubleshooting:")
        print("1. Verify I2C wiring: VCC->3.3V, GND->GND, SCL->GPIO3, SDA->GPIO2")
        print("2. Run: sudo i2cdetect -y 1 (should show device at 0x68)")
        print("3. Enable I2C: sudo raspi-config -> Interfacing Options -> I2C")
        sys.exit(1)
