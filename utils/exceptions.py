"""
Custom exceptions for JAX Kalman Filter system
"""

from typing import Optional, Any


class KalmanFilterError(Exception):
    """Base exception for Kalman Filter system"""

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.component = component
        self.details = details


class HardwareError(KalmanFilterError):
    """Hardware-related errors"""

    pass


class SensorError(HardwareError):
    """Sensor-specific errors"""

    pass


class I2CError(SensorError):
    """I2C communication errors"""

    pass


class CalibrationError(SensorError):
    """Sensor calibration errors"""

    pass


class EstimationError(KalmanFilterError):
    """State estimation errors"""

    pass


class NumericalInstabilityError(EstimationError):
    """Numerical stability issues in estimation"""

    pass


class CovarianceError(NumericalInstabilityError):
    """Covariance matrix issues"""

    pass


class ConfigurationError(KalmanFilterError):
    """Configuration-related errors"""

    pass


class ModelError(KalmanFilterError):
    """Model-related errors"""

    pass


class TrainingError(ModelError):
    """Training process errors"""

    pass
