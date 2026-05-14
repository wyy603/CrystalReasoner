import re

import numpy as np


def simple_cif_positions(prompt):
    pattern = r"<CIF>(.*?)</CIF>"
    matches = re.finditer(pattern, prompt, re.DOTALL)
    indices = [(m.start(1) - 5, m.end(1) + 6) for m in matches]
    return indices


def getseg(x, arr):
    for i in range(0, len(arr) - 1):
        if arr[i] <= x and x <= arr[i + 1]:
            return arr[i], arr[i + 1]


def strseg(tuple, exp=1):
    l, r = tuple
    if l != -np.inf:
        l = int(round(l * exp))
    if r != np.inf:
        r = int(round(r * exp))
    if l == -np.inf:
        return f"less or equal than {r}"
    if r == np.inf:
        return f"greater or equal than {l}"
    return f"in [{l}, {r}]"
