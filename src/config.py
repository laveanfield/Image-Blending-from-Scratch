from pathlib import Path
from typing import Optional

#=================================================
# Path config
#=================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT   = PROJECT_ROOT / "data"
SOURCE_DIR  = DATA_ROOT / "source"
MASK_DIR    = DATA_ROOT / "mask"
TARGET_DIR  = DATA_ROOT / "target"
RESULT_DIR  = DATA_ROOT / "result"

#=================================================
# Blending config
#=================================================

GRAD_MIX    = True     # Poisson only: mixed gradients
NUM_LEVELS  = 3        # Laplacian only: pyramid levels
IMAGE_NAME  = "03.jpg"

#=================================================
# Manual offset
#=================================================

USE_MANUAL_OFFSET = False
MANUAL_OFFSETS = {
    "01.jpg": [200,  21,  1.0],
    "02.jpg": [-300, -100, 1.0],
    "03.jpg": [-150, 10,  1.0],
}


def get_offset(image_name: str) -> tuple[Optional[list[int]], Optional[float]]:
    """Return (offset, scale) for a given image."""
    if not USE_MANUAL_OFFSET:
        return None, None
    entry = MANUAL_OFFSETS.get(image_name)
    if entry is None:
        return None, None
    return entry[:2], entry[2]