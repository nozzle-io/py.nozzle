from importlib.metadata import distribution, version
from pathlib import Path
import os

import nozzle

DIST_NAME = "nozzle-io"


def import_origin() -> Path:
    return Path(nozzle.__file__).resolve()


def forbidden_root() -> Path:
    return Path(os.environ["NOZZLE_FORBIDDEN_IMPORT_ROOT"]).resolve()


def test_distribution_metadata_matches_import_version():
    dist = distribution(DIST_NAME)
    assert dist.metadata["Name"] == DIST_NAME
    assert version(DIST_NAME) == nozzle.__version__


def test_nozzle_import_comes_from_installed_artifact_not_checkout():
    origin = import_origin()
    root = forbidden_root()
    print(f"import_origin={origin}")
    print(f"forbidden_import_root={root}")
    assert not origin.is_relative_to(root), f"imported nozzle from checkout: {origin}"
