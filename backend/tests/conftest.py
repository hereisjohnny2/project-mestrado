"""Makes the legacy CLI importable as a plain Python package for the parity
test, without touching its code (it uses bare imports like
``from rock_model import RockNetModel``, so it needs its own directory on
``sys.path``, exactly as if it were being run as a script from inside
``legacy/rock-nn/``)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROCKNN_DIR = REPO_ROOT / "legacy" / "rock-nn"

if str(LEGACY_ROCKNN_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROCKNN_DIR))
