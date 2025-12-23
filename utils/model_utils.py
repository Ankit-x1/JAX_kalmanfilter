
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Dict
from utils.logging_config import ComponentLogger

logger = ComponentLogger("model_utils")

def save_model(params: Any, filename: str) -> None:
    """Save model parameters to file"""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        raise

def load_model(filename: str) -> Any:
    """Load model parameters from file"""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise

def find_latest_model(model_dir: str = "models") -> Optional[str]:
    """Find the most recently modified model file"""
    path = Path(model_dir)
    if not path.exists():
        return None
        
    files = list(path.glob("*.pkl"))
    if not files:
        return None
        
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)
