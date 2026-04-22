import numpy as np

def naive_blend(image_data: dict) -> np.ndarray:
    source = image_data['source']
    mask = image_data['mask']
    target = image_data['target'].copy()
    dims = image_data['dims']

    target[dims[0]:dims[1], dims[2]:dims[3], :] = target[dims[0]:dims[1],dims[2]:dims[3],:] * (1 - mask) + source * mask

    return target