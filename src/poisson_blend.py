import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse import diags, csr_matrix

#=================================================
# Helper functions
#=================================================

def get_subimg(image: np.ndarray, dims: tuple) -> np.ndarray:
    return image[dims[0]:dims[1], dims[2]:dims[3]]

def linear_solver(A: csr_matrix, b: np.ndarray, dims: tuple) -> np.ndarray:
    x = linalg.spsolve(A.tocsc(), b)
    return np.reshape(x, (dims[0], dims[1]))

def stitch_img(source: np.ndarray, target: np.ndarray, dims: tuple) -> np.ndarray:
    target[dims[0]:dims[1], dims[2]:dims[3], :] = source
    return target

#=================================================
# Build sparse Laplacian matrix and guidance vector
#=================================================

def poisson_solver(image_data: dict, grad_mix: bool) -> list[tuple[csr_matrix, np.ndarray]]:
    """
    Find L, b of the equation: L.f = b
    """
    mask = image_data['mask']
    Hs, Ws = mask.shape
    num_pixels = Hs * Ws

    source = image_data['source'].flatten(order='C')
    target_subimg = get_subimg(image_data['target'], image_data['dims']).flatten(order='C')
    mask_flat = mask.flatten(order='C')
    mask_bool = mask_flat > 0.99
    # Build the sparse Laplacian block matrix 
    diag_main = np.where(mask_bool, 4.0, 1.0) # 4 in masked regions, 1 in unmasked regions

    diag_up = np.zeros(num_pixels - Ws)
    diag_down = np.zeros(num_pixels - Ws)
    diag_left = np.zeros(num_pixels - 1)
    diag_right = np.zeros(num_pixels - 1)

    ## Neighbor up
    valid_up = mask_bool[Ws:]
    diag_up = np.where(valid_up, -1.0, 0.0)

    ## Neighbor down
    valid_down = mask_bool[:num_pixels - Ws]
    diag_down = np.where(valid_down, -1.0, 0.0)

    ## Neighbor left
    valid_left = mask_bool[1:].copy()
    valid_left[Ws-1::Ws] = False  # Invalidate left neighbors at the start of each row
    diag_left = np.where(valid_left, -1.0, 0.0)

    ## Neighbor right
    valid_right = mask_bool[:num_pixels - 1].copy()
    valid_right[Ws-1::Ws] = False  # Invalidate right neighbors at the start of each row
    diag_right = np.where(valid_right, -1.0, 0.0)


    laplacian = diags([diag_main, diag_up, diag_down, diag_left, diag_right], 
                      [0, -Ws, Ws, -1, 1], 
                      shape=(num_pixels, num_pixels), format='csr')
    
    # Build guidance field
    guidance_field = target_subimg.copy()

    src_2d = source.reshape(Hs, Ws)
    tgt_2d = target_subimg.reshape(Hs, Ws)

    ## Gradient source
    grad_up_s = src_2d - np.roll(src_2d, shift=1, axis=0); grad_up_s[0, :] = src_2d[0, :]
    grad_down_s = src_2d - np.roll(src_2d, shift=-1, axis=0); grad_down_s[-1, :] = src_2d[-1, :]
    grad_left_s = src_2d - np.roll(src_2d, shift=1, axis=1); grad_left_s[:, 0] = src_2d[:, 0]
    grad_right_s =src_2d - np.roll(src_2d, shift=-1, axis=1); grad_right_s[:, -1] = src_2d[:, -1]
    
    if grad_mix:
        grad_up_t = tgt_2d - np.roll(tgt_2d, shift=1, axis=0); grad_up_t[0, :] = tgt_2d[0, :]
        grad_down_t = tgt_2d - np.roll(tgt_2d, shift=-1, axis=0); grad_down_t[-1, :] = tgt_2d[-1, :]
        grad_left_t = tgt_2d - np.roll(tgt_2d, shift=1, axis=1); grad_left_t[:, 0] = tgt_2d[:, 0]
        grad_right_t = tgt_2d - np.roll(tgt_2d, shift=-1, axis=1); grad_right_t[:, -1] = tgt_2d[:, -1]

        _compare = lambda x, y: np.where(np.abs(x) > np.abs(y), x, y)
        g_up = _compare(grad_up_s, grad_up_t)
        g_down = _compare(grad_down_s, grad_down_t)
        g_left = _compare(grad_left_s, grad_left_t)
        g_right = _compare(grad_right_s, grad_right_t)
    else:
        g_up = grad_up_s
        g_down = grad_down_s
        g_left = grad_left_s
        g_right = grad_right_s

    div = (g_up + g_down + g_left + g_right).flatten(order='C')
    guidance_field = np.where(mask_bool, div, target_subimg)

    return [laplacian, guidance_field]

#=================================================
# Poisson blending
#=================================================

def poisson_blend(image_data: dict, grad_mix: bool = False) -> np.ndarray:
    equation_param = []
    ch_data = {}
    # Construct Poisson equation for each channel
    for ch in range(3):
        ch_data['source'] = image_data['source'][:, :, ch]
        ch_data['mask'] = image_data['mask'][:, :, ch]
        ch_data['target'] = image_data['target'][:, :, ch]
        ch_data['dims'] = image_data['dims']
        equation_param.append(poisson_solver(ch_data, grad_mix))

    # Solve the Poisson equation for each channel
    image_result = np.empty_like(image_data['source'])
    for i in range(3):
        image_result[:, :, i] = linear_solver(equation_param[i][0], equation_param[i][1], (image_data['source'].shape[0], image_data['source'].shape[1]))
        
    image_result = stitch_img(image_result, image_data['target'].copy(), image_data['dims'])

    return np.clip(image_result, 0, 1)