# Image Blending from Scratch

From-scratch implementation of three image blending methods — rebuilding the core mathematics of each approach rather than relying on existing libraries. Includes naive alpha blending, multi-resolution Laplacian pyramid blending, and gradient-domain Poisson blending.

---

## Overview

Given a **source** image, a **mask**, and a **target** image, the goal is to seamlessly composite the masked region of the source onto the target — minimising visible seams at the boundary.

| Method | Idea |
|---|---|
| Naive copy | Alpha blending: `C = α·F + (1−α)·B` |
| Laplacian blending | Blend each frequency band separately via Gaussian/Laplacian pyramids |
| Poisson blending | Solve `Lf = b` to match source gradients while respecting target boundary conditions |

## Project Structure

```
gradient-domain-editing/
├── src/
│   ├── config.py           # Paths and run settings
│   ├── utils.py            # load_img, preprocess_images, display_images, select_offset
│   ├── naive_blend.py
│   ├── laplacian_blend.py
│   └── poisson_blend.py
├── data/
│   ├── source/
│   ├── mask/
│   ├── target/
│   └── result/
├── demo.ipynb              # Run and compare all three methods
└── requirements.txt
```

## Setup
```bash
py -3.12 -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

## Usage

**1. Prepare your images** — place `source`, `mask`, and `target` images with the same filename into their respective folders under `data/`.

To create a mask, use [GIMP](https://www.gimp.org/): open the source image, use the free select or fuzzy select tool to outline the region you want to blend, then fill the selection with white on a black background and export as the mask image.

**2. Configure** `src/config.py`:

```python
IMAGE_NAME        = "01.jpg"
GRAD_MIX          = True    # Poisson only: use mixed gradients
NUM_LEVELS        = 3       # Laplacian only: pyramid levels
USE_MANUAL_OFFSET = False   # True: use MANUAL_OFFSETS | False: interactive selector
```

**3. Run** `demo.ipynb` — the notebook loads the images, runs all three methods, displays a side-by-side comparison, and saves results to `data/result/`.

### Interactive offset selector

When `USE_MANUAL_OFFSET = False`, an OpenCV window opens for you to position the source on the target interactively:

| Control | Action |
|---|---|
| Mouse move / left-click | Set placement position |
| Scroll up / `+` | Scale source up (+5%) |
| Scroll down / `-` | Scale source down (-5%) |
| Enter | Confirm |

## Methods

Three methods are implemented to compare their trade-offs in quality, smoothness, and computational cost. The full mathematical derivation of each method is documented with worked examples in `demo.ipynb`.

### Naive copy
Direct alpha blend using the mask. Fast but produces visible colour discontinuities at the boundary.

### Laplacian blending
Decomposes source, target, and mask into Gaussian/Laplacian pyramids, then blends each frequency band independently. Coarser levels use a progressively blurred mask, giving smooth colour transitions without sacrificing high-frequency detail. Does not operate in the gradient domain.

### Poisson blending
Gradient-domain method that finds a result image whose gradients best match the source gradients inside the mask, subject to the target's boundary conditions. Reduces to solving a sparse linear system `Lf = b` per colour channel. Supports mixed gradients (`GRAD_MIX=True`) for better handling of semi-transparent objects.

## Results

<p align="center">
  <img src="data/result/naive.jpg" width="30%" />
  <img src="data/result/laplacian.jpg" width="30%" />
  <img src="data/result/poisson.jpg" width="30%" />
</p>

<p align="center">
  Naive &nbsp;&nbsp;&nbsp; Laplacian &nbsp;&nbsp;&nbsp; Poisson
</p>

## References

- Pérez, P., Gangnet, M., & Blake, A. (2003). *Poisson image editing*. ACM SIGGRAPH 2003.
- Burt, P., & Adelson, E. (1983). *The Laplacian pyramid as a compact image code*. IEEE Transactions on Communications.
- Arnebäck, E. *Laplacian Pyramids*.
- Salazar Cavazos, J. *Image Blending*.