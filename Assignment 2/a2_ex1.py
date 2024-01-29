import numpy as np
from PIL import Image

def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:    #we check the shape of the input
        return pil_image[np.newaxis, :, :] #if the image is already grayscale, we return this
    elif pil_image.ndim == 3 and pil_image.shape[2] == 3: #if the image is an RGB image:
        pass
    else:
        raise ValueError

    pil_image = pil_image / 255.0   #we normalize RGB values

    Clinear = np.where(pil_image <= 0.04045, pil_image / 12.92, ((pil_image + 0.055) / 1.055) ** 2.4)   #we calculate linear RGB values
    Rlinear, Glinear, Blinear = Clinear[:, :, 0], Clinear[:, :, 1], Clinear[:, :, 2]

    Ylinear = 0.2126 * Rlinear + 0.7152 * Glinear + 0.0722 * Blinear      #we calculate linear grayscale values

    Y = np.where(Ylinear <= 0.0031308, 12.92 * Ylinear, 1.055 * Ylinear ** (1 / 2.4) - 0.055)    #we calculate non-linear RGB values

    Y = Y[np.newaxis, :, :]   #we add a brightness chanell

    if np.issubdtype(pil_image.dtype, np.integer): #back th the original data type
        Y = np.round(Y).astype(pil_image.dtype)
    else:
        Y = Y.astype(pil_image.dtype)

    return Y
