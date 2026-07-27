"""Canonical dataset packaging and release verification for CVBench."""

from .errors import DatasetError
from .initializer import init_dataset
from .intake import import_contribution
from .manifest import build_release, verify_release
from .validator import validate_dataset

__all__ = [
    "DatasetError",
    "build_release",
    "init_dataset",
    "import_contribution",
    "validate_dataset",
    "verify_release",
]
__version__ = "0.1.0"
