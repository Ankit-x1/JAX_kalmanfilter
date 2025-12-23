# Professional JAX Kalman Filter

Differentiable Extended Kalman Filter for real-time IMU state estimation with neural network bias correction and production-grade deployment infrastructure.

## Abstract

This project implements a fully differentiable Extended Kalman Filter (EKF) using JAX for IMU sensor fusion, incorporating neural network-based bias correction and comprehensive error handling suitable for real-world hardware deployment. The system supports MPU6050 IMU sensors and provides robust data collection, training, and real-time estimation capabilities.

## System Requirements

- Python 3.11+
- JAX >= 0.4.0
- Flax >= 0.7.0
- NumPy >= 1.21.0
- PyYAML >= 6.0
- Raspberry Pi hardware (for deployment)
- MPU6050 IMU sensor

## Installation

### Development Setup
```bash
git clone <https://Ankit-x1/JAX_kalmanfilter.git>
cd JAX_kalmanfilter
pip install -r requirements.txt
pip install -e .
```

### Docker Deployment
```bash
docker build -t jax-kalman-filter .
docker-compose up -d
```

## Configuration

System behavior is controlled through `config.yaml`:

```yaml
imu:
  GYRO_SCALE: 131.0
  ACCEL_SCALE: 16384.0
  CALIBRATION_SAMPLES: 1000
  
ekf:
  PROCESS_NOISE_DIAG: [1e-6, ...]
  MEASUREMENT_NOISE_DIAG: [1e-4, ...]
  INITIAL_COVARIANCE: 1e-4
  
training:
  LEARNING_RATE: 1e-3
  BATCH_SIZE: 32
  MAX_EPOCHS: 100
```

## Operation Modes

### Data Collection
Collects real sensor data for training:
```bash
python main.py --mode collect
```

### Model Training
Trains bias correction neural network:
```bash
python main.py --mode train --epochs 50 --validation-split 0.2
```

### Real-time Deployment
Deploys trained model for live estimation:
```bash
python main.py --mode deploy --duration 3600
```

### System Testing
Validates hardware and performance:
```bash
python main.py --mode test
```



## Key Features

### Algorithmic Components
- Differentiable Extended Kalman Filter with JAX JIT compilation
- Neural network bias correction for sensor drift
- Quaternion-based attitude representation
- Numerical stability with Joseph form updates

### Production Infrastructure
- Component-based structured logging
- Comprehensive error handling and recovery
- Hardware health monitoring
- Graceful shutdown and resource cleanup
- Data validation and integrity checks

### Real-world Deployment
- I2C communication with error recovery
- Sensor calibration and drift compensation
- Performance benchmarking and endurance testing
- Docker containerization for deployment
- Configuration management for different environments

## Performance Characteristics

- Sample Rate: Up to 1kHz
- Processing Latency: <1ms per estimation step
- Memory Footprint: <100MB for real-time deployment
- Accuracy: Sub-degree attitude estimation with proper calibration

## Hardware Requirements

### Minimum Requirements
- Raspberry Pi 3B+ or equivalent
- MPU6050 IMU sensor on I2C bus
- 8GB microSD storage
- 5V power supply

### Recommended Configuration
- Raspberry Pi 4B (4GB RAM)
- High-quality I2C cabling
- 32GB microSD storage
- Stable power supply with filtering

## Testing and Validation

The system includes comprehensive testing:

1. **Hardware Diagnostics**: I2C connectivity, sensor health
2. **Performance Testing**: Processing speed, memory usage
3. **Endurance Testing**: Long-duration reliability
4. **Data Validation**: Integrity checks, outlier detection

Run tests with: `python main.py --mode test`

## Troubleshooting

### Common Issues

**I2C Connection Failed**
- Verify wiring: VCC->3.3V, GND->GND, SCL->GPIO3, SDA->GPIO2
- Check device address: `sudo i2cdetect -y 1`
- Enable I2C: `sudo raspi-config -> Interfacing Options`

**Poor Estimation Quality**
- Ensure sensor calibration is complete
- Verify training data quality and diversity
- Check EKF noise parameters in configuration

**Memory Issues**
- Reduce history buffer size in configuration
- Monitor log file size in `/app/logs`
- Validate data collection frequency

## Model Training Guidelines

### Data Collection Requirements
- Stationary data: 60 seconds minimum
- Rotation data: Multiple axis movements
- Trajectory data: Realistic motion patterns
- Total dataset: >10,000 samples recommended

### Training Parameters
- Learning rate: 1e-3 (adjustable)
- Batch size: 32 (memory dependent)
- Validation split: 20%
- Early stopping: 10 epochs patience

## Security Considerations

- Model parameters are stored with integrity checksums
- I2C access requires appropriate permissions
- Logs may contain sensitive system information
- Docker containers run with minimal privileges

## Contributing

1. Follow existing code structure and naming conventions
2. Add appropriate error handling for all hardware interactions
3. Include comprehensive logging for debugging
4. Update configuration schema for new parameters
5. Add tests for new functionality

## References

- Barfoot, T.D. "State Estimation for Robotics"
- Simon, D. "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches"
- JAX Documentation: https://jax.readthedocs.io/

## License

MIT License - see LICENSE file for details.

## Version History

- v2.0.0: Production-ready deployment infrastructure
- v1.0.0: Initial implementation of differentiable EKF