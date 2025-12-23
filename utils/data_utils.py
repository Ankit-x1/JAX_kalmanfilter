
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from utils.logging_config import ComponentLogger
from utils.data_validator import DataValidator

logger = ComponentLogger("data_utils")

def load_training_data(data_dir: str = "data/raw") -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load and preprocess training data.
    Returns: (states, measurements, targets) or None if failure
    """
    path = Path(data_dir)
    if not path.exists():
        logger.warning(f"Data directory {path} not found.")
        return None
    
    validator = DataValidator()
    all_samples = []
    
    # Load all JSON files
    json_files = list(path.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {path}")
        return None

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if validator.validate_json_structure(data):
                     all_samples.extend(data['samples'])
                else:
                    logger.warning(f"Skipping invalid file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            
    if not all_samples:
         logger.warning("No valid samples found.")
         return None

    try:
        # Process using validator logic which handles filtering and array conversion
        processed = validator.process_samples(all_samples)
        return processed['states'], processed['measurements'], processed['targets']
    except Exception as e:
        logger.error(f"Error processing samples: {e}")
        return None
