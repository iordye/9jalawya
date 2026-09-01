# utils/logging_config.py

import logging
import sys
from datetime import datetime 
from pathlib import Path

CONSOLE_FORMAT = (
    "%(asctime)s.%(msecs)03d |"
    "%(levelname)-8s |"
    "%(name)s |"
    "%(message)s"
)

FILE_FORMAT = (
    "%(asctime)s.%(msecs)03d |"
    "%(levelname)-8s |"
    "%(name)s |"
    "%(threadName)-15s |"
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(
        console_level: int = logging.INFO,
        log_to_file: bool = False,
        log_dir: str = "logs",
) -> None:
    """
    Set up logging configuration.

    Args:
        console_level (int): Logging level for console output.
        log_to_file (bool): Whether to log to a file.
        log_dir (str): Directory to save log files if log_to_file is True.
    """
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels at the root

    root_logger.handlers.clear()  # Clear existing handlers if called multiple times

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)  # Create log directory if it doesn't exist

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_dir_path / f"log_{timestamp}.log"

        file_handler = logging.FileHandler(log_file_path, encoding = "utf-8")
        file_handler.setLevel(logging.DEBUG)  # Log all levels to file
        file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_file_path}")

    # Silence noisy third-party libraries
    # These libraries log at DEBUG constantly and would drown out your logs
    _silence_noisy_libraries()

def _silence_noisy_libraries() -> None:
    """
    Silence noisy third-party libraries that log excessively at DEBUG level.
    """
    noisy_loggers = [
        "langchain",
        "langchain_huggingface",
        "langchain_text_splitters",
        "langchain_community",
        "langchain_astradb",
        "docx",
        "urllib3",          # HTTP library — logs every request/response
        "httpx",            # Another HTTP library
        "httpcore",
        "multipart",
        "filelock",         # Model download locking
        "transformers",     # Hugging Face — logs model loading details
        "torch",            # PyTorch — logs CUDA initialization
        "numba",   
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Call at the TOP of every module file, like this:
        logger = get_logger(__name__)

    Args:
        name: The logger name. 

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)

class TimingLogger:
    """
    Context manager for timing code execution and logging the duration.

    Usage:
        with TimingLogger("MyTask"):
            # Code to time
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation_name: str,
        level: int = logging.DEBUG,
    ):
        self.logger = logger
        self.operation_name = operation_name
        self.level = level
        self._start_time: float = 0.0

    def __enter__(self) -> "TimingLogger":
        import time
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        import time
        self.elapsed_ms = (time.perf_counter() - self._start_time) * 1000

        if exc_type is not None:
            # An exception occurred inside the 'with' block
            self.logger.error(
                f"{self.operation_name} FAILED after {self.elapsed_ms:.1f}ms | "
                f"error={exc_type.__name__}: {exc_val}"
            )
        else:
            self.logger.log(
                self.level,
                f"{self.operation_name} completed | duration_ms={self.elapsed_ms:.1f}"
            )

        # Return False: don't suppress the exception if one occurred
        return False