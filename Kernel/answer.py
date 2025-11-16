from typing import List
from Kernel import Kernel, init

import numpy as np
from scipy.signal import convolve2d

def filter_function(image: List[List[int]], kernel: List[List[int]]):
    kernel = np.array(kernel)
    image = np.array(image)
    filtered = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered

Kernel = Kernel("minion.png", filter_function)
# Kernel = Kernel("bill.png", filter_function)

init()