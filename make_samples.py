"""Generate synthetic demo scalp images for the static Space.

Procedural only: gradient scalp-colored background + follicle clusters drawn as
circles-of-hairs, seeded deterministically. NOT real scalp photos — labeled synthetic so
the privacy rule (no patient images, ever) holds by construction.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "samples"


def scalp_bg(w, h, rng):
    img = Image.new("RGB", (w, h))
    px = img.load()
    base_r, base_g, base_b = 214 + rng.randint(-8, 8), 178 + rng.randint(-8, 8), 152 + rng.randint(-8, 8)
    for y in range(h):
        for x in range(w):
            n = rng.randint(-14, 14)
            px[x, y] = (base_r + n, base_g + n - 4, base_b + n - 8)
    return img


def draw_unit(d, cx, cy, n_hairs, rng, minaturized=False):
    """Draw one follicular unit: n dark shafts fanning from one pore."""
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(70, 50, 45))
    for _ in range(n_hairs):
        ang = rng.uniform(0, 6.283)
        ln = rng.uniform(3, 9 if minaturized else 14)
        d.line([(cx, cy), (cx + ln * __import__("math").cos(ang), cy + ln * __import__("math").sin(ang))],
               fill=(38, 28, 24), width=1)


def make(seed, w=640, h=640, minaturize_top=False):
    rng = random.Random(seed)
    img = scalp_bg(w, h, rng)
    d = ImageDraw.Draw(img)
    for cy in range(28, h, 26):
        for cx in range(28, w, 26):
            mini = minaturize_top and cy < h / 3
            n = rng.choice([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5]) if not mini else rng.choice([1, 1, 1, 2])
            draw_unit(d, cx, cy, n, rng, minaturized=mini)
    return img


OUT.mkdir(parents=True, exist_ok=True)
make(7).save(OUT / "synthetic-1.png")
make(42, minaturize_top=True).save(OUT / "synthetic-2.png")
print("wrote", list(OUT.glob("*.png")))
