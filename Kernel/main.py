from typing import List
from Kernel import Kernel, init

def filter_function(image: List[List[int]], kernel: List[List[int]]):
    return image[:]

Kernel = Kernel("minion.jfif", filter_function)

init()