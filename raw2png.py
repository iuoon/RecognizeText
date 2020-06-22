from PIL import Image, ImageFilter
import numpy as np
from scipy import ndimage, misc
import imageio

if __name__ == "__main__":
    rawfile = np.fromfile('d:\\terrain.raw', "uint16")
    rawfile.shape = (1025,1025)
    b=rawfile.astype(np.uint16)
    # misc.imsave("d:\\terrain.png", b)
    imageio.imwrite("d:\\terrain.png", b)
