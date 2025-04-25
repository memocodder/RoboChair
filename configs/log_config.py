import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Tuple
from pathlib import Path


@dataclass
class LoggingConfig:
    """Базовая конфигурация логирования экспериментов."""

    base_log_dir: str = "results"
    experiment_name: Optional[str] = None
    log_level: str = "info"
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"
    log_to_console: bool = True
    log_to_file: bool = True
    overwrite_existing: bool = False
    experiment_description: str = "No description."
    save_config: bool = True

    _experiment_dir_path: Path = field(init=False, repr=False)
    _log_file_path: Path = field(init=False, repr=False)
    _logger_configured: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"

        self._experiment_dir_path = (
            Path(self.base_log_dir) / self.experiment_name
        )
        self._log_file_path = self._experiment_dir_path / "experiment.log"

    @property
    def experiment_dir(self) -> Path:
        if not self._logger_configured:
            raise RuntimeError(
                "Cannot access experiment_dir. Logger not configured yet. "
                "Call setup_logger(config) first."
            )
        return self._experiment_dir_path


def setup_logger(config: LoggingConfig) -> logging.Logger:
    """
    Настраивает логгер и создает папку эксперимента.

    Возвращает только логгер. Путь к папке доступен через config.experiment_dir.
    """
    exp_dir = config._experiment_dir_path

    if exp_dir.exists() and not config.overwrite_existing:
        raise FileExistsError(
            f"Directory '{exp_dir}' already exists. Use overwrite_existing=True to ignore."
        )

    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "description.txt").write_text(
        config.experiment_description, encoding="utf-8"
    )

    logger = logging.getLogger(config.experiment_name)
    logger.setLevel(config.log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(config.log_format)
    handlers = []

    if config.log_to_file:
        file_handler = logging.FileHandler(
            config._log_file_path, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if not handlers:
        logger.addHandler(logging.NullHandler())
    else:
        for handler in handlers:
            logger.addHandler(handler)

    config._logger_configured = True

    logger.info(
        f"Logger setup complete. Experiment: '{config.experiment_name}'"
    )
    logger.info(f"Log directory: {config.experiment_dir.resolve()}")

    return logger


