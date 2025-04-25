import pickle
from pathlib import Path
import logging
from typing import Any


def save_data(experiment_dir: Path, filename: str, data: Any):
    """Сохраняет данные эксперимента в файл с использованием pickle."""
    logger = logging.getLogger()
    filepath = (experiment_dir / filename).with_suffix(".pkl")
    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to: {filepath.resolve()} (format: pickle)")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath} using pickle: {e}")


def load_data(experiment_dir: Path, filename: str) -> Any:
    """Загружает данные эксперимента из файла pickle."""
    logger = logging.getLogger()
    filepath = (experiment_dir / filename).with_suffix(".pkl")
    if not filepath.exists():
        logger.error(f"Cannot load data. File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Data loaded from: {filepath.resolve()} (format: pickle)")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {filepath} using pickle: {e}")
        raise
