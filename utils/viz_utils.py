import numpy as np
import tifffile
from matplotlib.colors import to_rgb
from utils import temporal_utils

# ---------------------------------------------------------------------------
# Slice / stack mapping  (1-based, matching ImageJ stack order)
# ---------------------------------------------------------------------------

SLICE_NAMES = [
    "eve",    # 1
    "engril", # 2  (the TIFF slice name; "en" in stage_utils is aliased here)
    "mlpt",   # 3
    "hb",     # 4
    "kr",     # 5
    "gt",     # 6
    "svb",    # 7
    "cad",    # 8
    "runt",   # 9
    "odd",    # 10
]

# Map gene names (including stage_utils alias "en") to 1-based stack numbers
GENE_TO_STACK = {name: idx + 1 for idx, name in enumerate(SLICE_NAMES)}
GENE_TO_STACK["en"] = GENE_TO_STACK["engril"]   # stage_utils uses "en"

# ---------------------------------------------------------------------------
# Colours — taken directly from temporal_utils.COLORS (no lightening)
# "engril" slice uses the "en" colour entry
# ---------------------------------------------------------------------------

STACK_COLORS = {}
for _name in SLICE_NAMES:
    _lookup = "en" if _name == "engril" else _name
    _color  = temporal_utils.COLORS.get(_lookup, "gray")
    STACK_COLORS[GENE_TO_STACK[_name]] = _color


# ---------------------------------------------------------------------------
# TIFF I/O
# ---------------------------------------------------------------------------

def load_tiff(filename):
    with tifffile.TiffFile(filename) as tif:
        return tif.asarray()


def get_stack(image, stack_number=1):
    """Extract one slice from a multi-stack TIFF (1-based)."""
    if image.ndim == 2:
        return image
    total = image.shape[0]
    if not (1 <= stack_number <= total):
        raise ValueError(f"stack_number must be 1–{total}, got {stack_number}.")
    return image[stack_number - 1]


# ---------------------------------------------------------------------------
# Display mapping
# ---------------------------------------------------------------------------

def apply_window(img, lo, hi):
    """Linear stretch: [lo, hi] → [0, 1].  Below lo → black, above hi → white."""
    img   = img.astype(np.float32)
    denom = hi - lo
    if denom <= 0:
        return np.zeros_like(img)
    return np.clip((img - lo) / denom, 0.0, 1.0)


def autoadjust(img, saturation=0.35):
    """Auto-stretch: clip saturation% of pixels at each tail (ImageJ Auto)."""
    lo = float(np.percentile(img, saturation / 2.0))
    hi = float(np.percentile(img, 100.0 - saturation / 2.0))
    if hi <= lo:
        hi = lo + 1.0
    return apply_window(img, lo, hi)


# ---------------------------------------------------------------------------
# Colorisation
# ---------------------------------------------------------------------------

def apply_color(img_norm, stack_number):
    """Multiply a normalised [0,1] grayscale image by the gene's RGB colour."""
    color   = np.array(to_rgb(STACK_COLORS[stack_number]), dtype=np.float32)
    colored = np.zeros((*img_norm.shape, 3), dtype=np.float32)
    for i in range(3):
        colored[..., i] = img_norm * color[i]
    return colored


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_stack_number(gene):
    """Return 1-based stack number for a gene name, or None if not mapped."""
    return GENE_TO_STACK.get(gene, None)