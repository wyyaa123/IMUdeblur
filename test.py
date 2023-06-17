import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import include.PSF as _PSF
import include.Image as _Image

if __name__ == '__main__':

    kernel = _PSF.PSFFunction(10, 30)
    kernel.calculate_h()
    PSF = _Image.kernel_compliant(kernel.hh)
    # PSF *= 2550
    # PSF = PSF.astype(np.uint8)
    plt.imshow(PSF, cmap='gray', vmin=0, vmax=0.1)
    plt.axis('off')
    plt.show()