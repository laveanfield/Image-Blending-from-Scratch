from pathlib import Path
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
from config import SOURCE_DIR, MASK_DIR, TARGET_DIR

#=================================================
# Loading and showing image
#=================================================

def load_img(image_name: str, target_offset=None, scale=None) -> dict:
    image_data = {}
    source = cv2.imread(str(SOURCE_DIR / image_name))
    mask   = cv2.imread(str(MASK_DIR   / image_name))
    target = cv2.imread(str(TARGET_DIR / image_name))

    if source is None: raise FileNotFoundError(f"Source not found: {SOURCE_DIR / image_name}")
    if mask is None: raise FileNotFoundError(f"Mask not found: {MASK_DIR   / image_name}")
    if target is None: raise FileNotFoundError(f"Target not found: {TARGET_DIR / image_name}")

    # Normalize the images to [0, 1]
    image_data['source'] = cv2.normalize(source.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image_data['mask'] = cv2.normalize(mask.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image_data['target'] = cv2.normalize(target.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    if target_offset is None:
        dims, scale = select_offset(image_name)
        image_data['dims'] = dims
    else:
        image_data['dims'] = target_offset

    # Scale source and mask if user adjusted size
    if scale is not None and scale != 1.0:
            Hs, Ws = image_data['source'].shape[:2]
            new_H = max(1, int(round(Hs * scale)))
            new_W = max(1, int(round(Ws * scale)))
            image_data['source'] = cv2.resize(image_data['source'], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            image_data['mask']   = cv2.resize(image_data['mask'],   (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            print(f"Scaled source & mask by {scale:.2f}x → {new_H}x{new_W}")
    return image_data

def display_images(image_data: dict, title: str = "") -> None:
    keys = [k for k in image_data.keys() if k != "dims"]
    plt.figure(figsize=(6 * len(keys) , 6))
    if title:
        plt.suptitle(title)
    for i, key in enumerate(keys):
        img = image_data[key]
        plt.subplot(1, len(keys), i + 1)
        plt.title(key)
        plt.imshow(np.clip(img, 0, 1)[:, :, [2, 1, 0]]) # Convert BGR to RGB
        # plt.axis("off")
    plt.tight_layout()
    plt.show()

#=================================================
# Preprocessing
#=================================================

def preprocess_images(image_data: dict) -> dict:
    source = image_data['source'].copy()
    mask = image_data['mask'].copy()
    target = image_data['target'].copy()
    
    # Get image shapes and offset dimensions
    Hs, Ws, _ = source.shape
    Ht, Wt, _ = target.shape
    Ho, Wo = image_data['dims']

    print(f"Source shape: {Hs}x{Ws}")
    print(f"Target shape: {Ht}x{Wt}")
    print(f"Offset (Ho, Wo): ({Ho}, {Wo})")
    print(f"Target region: H[{Ho}:{min(Ho+Hs,Ht)}] W[{Wo}:{min(Wo+Ws,Wt)}]")

    # Adjust the source and mask images based on the offsets (if negative)
    if Ho < 0:
        mask = np.roll(mask, shift=Ho, axis=0)
        source = np.roll(source, shift=Ho, axis=0)
        mask[Ho + Hs:,:,:] = 0
        source[Ho + Hs:,:,:] = 0
        Ho = 0
        
    
    if Wo < 0:
        mask = np.roll(mask, shift=Wo, axis=1)
        source = np.roll(source, shift=Wo, axis=1)
        mask[:, Wo + Ws:,:] = 0
        source[:, Wo + Ws:,:] = 0
        Wo = 0
    
    # Mask region on target
    H_min = Ho
    H_max = min(Ho + Hs, Ht)
    W_min = Wo
    W_max = min(Wo + Ws, Wt)

    # Crop source and mask if they exceed target dimensions
    source = source[0:min(Hs, Ht - Ho), 0:min(Ws, Wt - Wo), :]
    mask = mask[0:min(Hs, Ht - Ho), 0:min(Ws, Wt - Wo), :]

    return {'source': source, 'mask': mask, 'target': target, 'dims': (H_min, H_max, W_min, W_max)}

#=================================================
# Select offset
#=================================================

def select_offset(image_name: str) -> tuple[list[int], float]:
    """
    Controls:
      - Mouse move / Left-click : choose placement position
      - Scroll UP / press '+'   : scale source+mask UP  (+5% per step)
      - Scroll DOWN / press '-' : scale source+mask DOWN (-5% per step)
      - Enter                   : confirm
    """
    target_path = str(TARGET_DIR / image_name)
    source_path = str(SOURCE_DIR / image_name)
    mask_path   = str(MASK_DIR   / image_name)
    
    target = cv2.imread(target_path)
    source_orig = cv2.imread(source_path)
    mask_orig   = cv2.imread(mask_path)

    Ht, Wt = target.shape[:2]
    Hs_orig, Ws_orig = source_orig.shape[:2]

    state = {
        'offset': [0, 0],
        'scale': 1.0,
        'source': source_orig.copy(),
        'mask': mask_orig.copy(),
    }

    WIN = "Select Offset & Scale (scroll=resize, Enter=confirm)"

    def _redraw(y, x):
        src  = state['source']
        msk  = state['mask']
        Hs, Ws = src.shape[:2]
        display = target.copy()
        Hmax = min(y + Hs, Ht)
        Wmax = min(x + Ws, Wt)
        if Hmax > y and Wmax > x:
            src_crop  = src[0:Hmax - y, 0:Wmax - x]
            mask_crop = msk[0:Hmax - y, 0:Wmax - x] / 255.0
            roi = display[y:Hmax, x:Wmax]
            display[y:Hmax, x:Wmax] = (1 - mask_crop * 0.7) * roi + (mask_crop * 0.7) * src_crop
        cv2.rectangle(display, (x, y), (min(x+Ws,Wt), min(y+Hs,Ht)), (0, 255, 0), 2)
        sc = state['scale']
        cv2.putText(display,
                    f"Offset:({y},{x})  Scale:{sc:.2f}x ({int(Hs_orig*sc)}x{int(Ws_orig*sc)})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow(WIN, display)

    def _rescale(delta):
        state["scale"] = max(0.05, state["scale"] + delta)
        new_H = max(1, int(round(Hs_orig* state["scale"])))
        new_W = max(1, int(round(Ws_orig * state["scale"])))
        state["source"] = cv2.resize(source_orig, (new_W,  new_H), interpolation=cv2.INTER_LINEAR)
        state["mask"]   = cv2.resize(mask_orig,   (new_W,  new_H), interpolation=cv2.INTER_LINEAR)
        _redraw(*state["offset"])

    def mouse_callback(event, x, y, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            if event == cv2.EVENT_LBUTTONDOWN:
                state['offset'] = [y, x]
                # print(f"Selected offset: H0={y}, W0={x}")
            _redraw(state['offset'][0], state['offset'][1])

        elif event == cv2.EVENT_MOUSEWHEEL:
            _rescale(0.05 if flags > 0 else -0.05)

    cv2.imshow(WIN, target)
    cv2.setMouseCallback(WIN, mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:   # Enter to confirm
            break
        elif key == ord('+') or key == ord('='): 
            _rescale(+0.05)
        elif key == ord('-'): 
            _rescale(-0.05)

    cv2.destroyAllWindows()
    print(f"Confirmed — offset: {state['offset']}, scale: {state['scale']:.2f}x")
    return state['offset'], state['scale']