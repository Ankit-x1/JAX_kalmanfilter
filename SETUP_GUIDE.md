# Raspberry Pi + MPU6050 Setup Guide

## Hardware Setup

### 1. Wiring Connections
Connect MPU6050 to Raspberry Pi GPIO pins:

```
MPU6050    ->  Raspberry Pi
VCC        ->  3.3V (Pin 1)
GND        ->  GND (Pin 6)
SCL        ->  GPIO3/SCL (Pin 5)
SDA        ->  GPIO2/SDA (Pin 3)
```

### 2. Enable I2C on Raspberry Pi
```bash
sudo raspi-config
# Select: Interfacing Options -> I2C -> Enable
sudo reboot
```

### 3. Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python packages
pip3 install jax[cpu] flax optax numpy scipy matplotlib smbus2 tqdm

# Install I2C tools
sudo apt install -y i2c-tools
```

### 4. Verify I2C Connection
```bash
# Scan for I2C devices
sudo i2cdetect -y 1
# You should see device at address 0x68
```

## Testing Steps

### Step 1: Test IMU Connection
```bash
cd /path/to/JAX_kalmanfilter
python3 -c "
from drivers.mpu6050 import MPU6050
import time

print('Testing MPU6050 connection...')
try:
    imu = MPU6050()
    imu.calibrate()
    print('✓ IMU connected and calibrated')
    
    # Read a few samples
    for i in range(5):
        gyro, accel, timestamp = imu.read_calibrated()
        print(f'Sample {i+1}: Gyro={gyro}, Accel={accel}')
        time.sleep(0.1)
except Exception as e:
    print(f'✗ Error: {e}')
"
```

### Step 2: Collect Training Data
```bash
# Collect stationary data (keep IMU still)
python3 main.py --mode collect

# This will create data in data/raw/ directory
```

### Step 3: Train the EKF
```bash
# Train with collected data
python3 main.py --mode train --epochs 50

# This will:
# 1. Load IMU data from data/raw/
# 2. Train the neural bias correction
# 3. Save model to trained_model.pkl
```

### Step 4: Deploy Real-time Estimation
```bash
# Run real-time state estimation
python3 main.py --mode deploy --duration 60

# This will:
# 1. Load trained model
# 2. Read live IMU data
# 3. Run EKF estimation
# 4. Display orientation angles
# 5. Save results to estimation_results.png
```

## Expected Output

### During Data Collection:
```
=== Data Collection Mode ===
Calibrating IMU... Keep sensor stationary.
Gyro bias: [0.01 -0.02 0.00]
Accel bias: [0.1 0.05 9.8]
Collecting stationary data for 60.0s...
  Progress: 0/6000 (0.0%)
...
Saved 6000 samples to data/raw/stationary_20231223_123456.json
```

### During Training:
```
=== Training Mode ===
Loading dataset...
Loaded 18000 training samples
Starting training...
Training complete. Model saved.
```

### During Deployment:
```
=== Deployment Mode ===
Starting real-time estimation...
Press Ctrl+C to stop
...
Results saved to estimation_results.png
```

## Troubleshooting

### Common Issues:

1. **I2C Device Not Found (0x68)**
   - Check wiring connections
   - Ensure 3.3V power (not 5V)
   - Enable I2C in raspi-config

2. **Permission Denied on I2C**
   ```bash
   sudo usermod -a -G i2c $USER
   sudo reboot
   ```

3. **Import Errors**
   ```bash
   # Reinstall packages
   pip3 install --upgrade jax[cpu] flax optax smbus2
   ```

4. **Calibration Issues**
   - Keep sensor perfectly still during calibration
   - Ensure sensor is on a flat surface

### Performance Tips:

1. **Reduce Sample Rate** if CPU usage is high:
   - Edit `data/collector.py`: change `sample_rate = 100` to `50`

2. **Disable Visualization** for faster real-time:
   ```bash
   python3 main.py --mode deploy --duration 60 --no-visualize
   ```

3. **Use GPU Acceleration** (if available):
   ```bash
   pip3 install jax[cuda]  # For NVIDIA GPU
   ```


## Next Steps:

1. **Start with Step 1** to verify hardware connection
2. **Collect diverse data** (stationary, rotation, movement)
3. **Train for more epochs** for better accuracy
4. **Experiment with different motions** to test robustness
