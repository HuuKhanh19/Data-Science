"""
Project-wide logging setup.

Implements IMPLEMENTATION.md section 9.

Format: ``[timestamp] [level] [module] message``
Default level: INFO. Override via env var ``LOG_LEVEL=DEBUG``.

Critical events (refit, inference, errors) WILL also be logged to the
``events`` table in SQLite (see src/storage/db.py, implemented in Session 13).
This module only handles stdout/stderr logging.

Usage
-----
>>> from src.utils.logging import get_logger
>>> log = get_logger(__name__)
>>> log.info("Starting data acquisition")
"""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Get a project-configured logger.

    Idempotent: calling multiple times with the same name returns the same
    logger without duplicating handlers.

    Parameters
    ----------
    name : str
        Logger name. Convention: pass ``__name__`` from caller.

    Returns
    -------
    logging.Logger
        Configured logger with stdout handler. Does NOT propagate to root
        logger (to avoid duplicate output).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured by previous get_logger call

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    logger.propagate = False  # Don't double-log via root logger

    return logger
