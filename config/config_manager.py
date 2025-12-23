"""
Centralized configuration management for JAX Kalman Filter
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class IMUConfig:
    GYRO_SCALE: float = 131.0
    ACCEL_SCALE: float = 16384.0
    SMPLRT_DIV: int = 7
    CALIBRATION_SAMPLES: int = 1000
    BUS_NUM: int = 1
    ADDRESS: int = 0x68
    TIMEOUT: float = 0.1


@dataclass
class EKFConfig:
    PROCESS_NOISE_DIAG: list = None
    MEASUREMENT_NOISE_DIAG: list = None
    INITIAL_COVARIANCE: float = 1e-4
    NUMERICAL_TOLERANCE: float = 1e-10

    def __post_init__(self):
        if self.PROCESS_NOISE_DIAG is None:
            self.PROCESS_NOISE_DIAG = [1e-6] * 13
        if self.MEASUREMENT_NOISE_DIAG is None:
            self.MEASUREMENT_NOISE_DIAG = [1e-4] * 6


@dataclass
class TrainingConfig:
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2
    MAX_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10
    CHECKPOINT_INTERVAL: int = 5


@dataclass
class LoggingConfig:
    LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    COMPONENTS: list = None

    def __post_init__(self):
        if self.COMPONENTS is None:
            self.COMPONENTS = ["main", "driver", "estimator", "trainer"]


@dataclass
class SystemConfig:
    """Master configuration class"""

    imu: IMUConfig = None
    ekf: EKFConfig = None
    training: TrainingConfig = None
    logging: LoggingConfig = None

    def __post_init__(self):
        if self.imu is None:
            self.imu = IMUConfig()
        if self.ekf is None:
            self.ekf = EKFConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


class ConfigManager:
    """Configuration manager with file support"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self.config = SystemConfig()
        self._load_config()

    def _load_config(self):
        """Load configuration from file"""
        if not self.config_path.exists():
            logger.info(f"Config file {self.config_path} not found, using defaults")
            self._save_config()
            return

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)

            if data:
                self._update_config_from_dict(data)
                logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")

    def _update_config_from_dict(self, data: Dict[str, Any]):
        """Update config from dictionary"""
        if "imu" in data:
            imu_data = data["imu"]
            self.config.imu = IMUConfig(
                **{k: v for k, v in imu_data.items() if hasattr(IMUConfig, k)}
            )

        if "ekf" in data:
            ekf_data = data["ekf"]
            self.config.ekf = EKFConfig(
                **{k: v for k, v in ekf_data.items() if hasattr(EKFConfig, k)}
            )

        if "training" in data:
            training_data = data["training"]
            self.config.training = TrainingConfig(
                **{k: v for k, v in training_data.items() if hasattr(TrainingConfig, k)}
            )

        if "logging" in data:
            logging_data = data["logging"]
            self.config.logging = LoggingConfig(
                **{k: v for k, v in logging_data.items() if hasattr(LoggingConfig, k)}
            )

    def _save_config(self):
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            with open(self.config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get_config(self) -> SystemConfig:
        """Get current configuration"""
        return self.config

    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for section, updates in kwargs.items():
            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                for key, value in updates.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        logger.info(f"Updated {section}.{key} = {value}")
        self._save_config()


# Global configuration instance
config_manager = ConfigManager()
