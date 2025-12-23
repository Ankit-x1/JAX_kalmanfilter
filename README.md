# JAX Kalman Filter

Differentiable Extended Kalman Filter for IMU state estimation with learned bias correction.

## Overview

Implementation of a differentiable EKF using JAX for IMU sensor fusion, featuring neural network-enhanced bias correction and real-time deployment capabilities.

## Requirements

- Python 3.12+
- JAX >= 0.4.0
- Flax >= 0.7.0

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Data Collection
```bash
python main.py --mode collect
```

### Training
```bash
python main.py --mode train --epochs 100
```

### Deployment
```bash
python main.py --mode deploy --duration 30
```

## Architecture

- `dynamics/` - Rigid body dynamics models
- `estimator/` - Differentiable EKF implementation
- `models/` - Neural network components
- `drivers/` - Sensor interfaces (MPU6050)
- `data/` - Collection and preprocessing
- `training/` - Training pipeline
- `deployment/` - Real-time execution

## License

MIT License