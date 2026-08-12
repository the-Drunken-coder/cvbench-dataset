"""Canonical dataset packaging and release verification for CVBench."""

from .errors import DatasetError
from .initializer import init_dataset
from .intake import import_contribution
from .manifest import build_release, verify_release
from .source_recipe import hydrate_source_recipe, validate_source_recipe
from .validator import validate_dataset

__all__ = [
    "DatasetError",
    "build_release",
    "init_dataset",
    "import_contribution",
    "hydrate_source_recipe",
    "validate_dataset",
    "validate_source_recipe",
    "verify_release",
]
__version__ = "0.1.0"
