import cv2
import numpy as np

#=================================================
# Convolution
#=================================================

def _conv(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.astype(np.float64)
    ksize = len(k)
    pad = ksize // 2

    def _conv1d_axis(arr: np.ndarray, axis: int) ->np.ndarray:
        arr = np.moveaxis(arr, axis, 0)
        N = arr.shape[0]

        left = arr[pad : 0 : -1]
        right = arr[N-2 : N-2-pad : -1]
        padded = np.concatenate([left, arr, right], axis=0)

        out = np.zeros_like(arr, dtype=np.float64)
        for i in range(N):
            window = padded[i : i+ksize]
            out[i] = np.tensordot(k, window, axes=([0], [0]))
        
        return np.moveaxis(out, 0, axis)
    
    img_f = img.astype(np.float64)
    img_f = _conv1d_axis(img_f, axis=1)
    img_f = _conv1d_axis(img_f, axis=0)
    return img_f



_GAUSS5 = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

#=================================================
# Pyramid operations
#=================================================

def pyr_down(img: np.ndarray) ->np.ndarray:
    blurred = _conv(img, _GAUSS5)
    return blurred[::2, ::2].astype(img.dtype)

def pyr_up(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    has_color = (img.ndim == 3)
    C = img.shape[2] if has_color else 1

    if has_color:
        up = np.zeros((H*2, W*2, C), dtype=np.float64)
        up[::2, ::2] =img.astype(np.float64)
    else:
        up = np.zeros((H*2, W*2), dtype=np.float64)
        up[::2, ::2] =img.astype(np.float64)

    result = _conv(up, _GAUSS5) * 4.0
    return result.astype(img.dtype)


#=================================================
# Gaussian pyramid
#=================================================

def build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    gp, cur = [img.copy()], img.copy()
    for _ in range(levels):
        cur = pyr_down(cur)
        gp.append(cur)
    return gp  # index 0=finest, index levels=coarsest

#=================================================
# Laplacian pyramid
#=================================================

def build_laplacian_pyramid(gp: list, levels: int) -> list:
    lp = [gp[levels].astype(np.float32)]   # coarsest giữ nguyên
    for i in range(levels, 0, -1):
        up  = pyr_up(gp[i])
        up  = cv2.resize(up, (gp[i-1].shape[1], gp[i-1].shape[0]))
        lap = gp[i-1].astype(np.float32) - up.astype(np.float32)
        lp.append(lap)
    return lp  # index 0=coarsest, index levels=finest

#=================================================
# Laplacian blending
#=================================================

def laplacian_blend(image_data: dict, num_levels: int = 6):
    source = image_data['source']
    mask   = image_data['mask']
    target = image_data['target']
    H_min, H_max, W_min, W_max = image_data['dims']

    target_region = target[H_min:H_max, W_min:W_max].copy()

    src_u8 = (source        * 255).astype(np.uint8)
    tgt_u8 = (target_region * 255).astype(np.uint8)
    msk_u8 = (mask          * 255).astype(np.uint8)

    mask_bin = (msk_u8 > 128).astype(np.uint8)
    if mask_bin.ndim == 2:
        mask_bin = mask_bin[:, :, np.newaxis]
    src_u8 = src_u8 * mask_bin + tgt_u8 * (1 - mask_bin)

    # Gaussian pyramid
    gp_src  = build_gaussian_pyramid(src_u8,  num_levels)
    gp_tgt  = build_gaussian_pyramid(tgt_u8,  num_levels)
    gp_mask = build_gaussian_pyramid(msk_u8,  num_levels)

    # Laplacian pyramid

    lp_src = build_laplacian_pyramid(gp_src, num_levels)
    lp_tgt = build_laplacian_pyramid(gp_tgt, num_levels)

    # Blend
    blended_pyramid = []
    for i, (lap_s, lap_t) in enumerate(zip(lp_src, lp_tgt)):
        # lp index 0=coarsest → mask index num_levels=coarsest
        m = gp_mask[num_levels - i].astype(np.float32) / 255.0
        m = cv2.resize(m, (lap_s.shape[1], lap_s.shape[0]))
        if m.ndim == 2:
            m = m[:, :, np.newaxis]
        blended_pyramid.append(lap_s * m + lap_t * (1.0 - m))

    # Reconstruct
    result = blended_pyramid[0]
    for i in range(1, num_levels + 1):
        result = cv2.pyrUp(result)
        result = cv2.resize(result, (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
        result += blended_pyramid[i]

    result = np.clip(result / 255.0, 0.0, 1.0)

    output = target.copy()
    output[H_min:H_max, W_min:W_max] = result
    return output