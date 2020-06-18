import numpy as np
import math

def nearest_neighbour_topography(
        shape: (int, int), 
        self_coupling: float, 
        neighbour_coupling: float) -> np.ndarray:
    """
    Creates a matrix representing a topography where each point
    affects itself and its nearest neighbours.
    """
    rows = shape[0]
    cols = shape[1]
    size = rows * cols
    result = np.zeros((size, size))

    for i in range(size):
        result[i, i] = self_coupling
        col_i = i % cols
        row_i = i // cols

        for row_offset in range(-1, 2):
            for col_offset in range(-1, 2):
                if row_offset != 0 or col_offset != 0:
                    row = row_i + row_offset
                    col = col_i + col_offset
                    if row >= 0 and row < rows and col >= 0 and col < cols:
                        src = row * cols + col
                        result[src, i] = neighbour_coupling

    return result

def exponential_topography(
        shape: (int, int),
        self_coupling: float, 
        decay: float) -> np.ndarray:
    """
    Creates a matrix representing a topography where each point
    affects itself and its nearest neighbours.
    """
    rows = shape[0]
    cols = shape[1]
    size = rows * cols
    result = np.zeros((size, size))

    for i in range(size):
        col_i = i % cols
        row_i = i // cols
        for row_offset in range(-rows, rows):
            for col_offset in range(-cols, cols):
                row = row_i + row_offset
                col = col_i + col_offset
                if row >= 0 and row < rows and col >= 0 and col < cols:
                    src = row * cols + col
                    distance = math.sqrt(row_offset * row_offset + col_offset * col_offset)
                    coupling = self_coupling * math.exp(-distance * decay)
                    result[src, i] = coupling

    return result

def fast_exponential_topography(
        shape: (int, int),
        self_coupling: float, 
        decay: float) -> np.ndarray:
    """
    Creates a matrix representing a topography where each point
    affects itself and its nearest neighbours.
    """
    rows = shape[0]
    cols = shape[1]
    size = rows * cols
    result = np.zeros((size, size))

    # generate the top row
    for row in range(0, rows):
        for col in range(0, cols):
            src = row * cols + col
            distance = math.sqrt(row * row + col * col)
            coupling = self_coupling * math.exp(-distance * decay)
            result[0, src] = coupling

    # Generate an upper triangular matrix by copying the top row
    for i in range(1, size):
        result[i, i:] = result[0, 0:-i]

    # Make it symmetric (we know that every value is non-negative)
    return np.maximum(result, result.transpose())

def stratified_topography(
        shape: (int, int),
        coupling_multiplier: float,
        distance_decay: float,
        coupling_decay: float) -> np.ndarray:
    """
    Creates a matrix representing a one-dimensional topography where each point
    affects itself and an exponentially decreasing area around. Also, the self-
    coupling is much greater for small i:

    coupling(i, j) = exp(-abs(i - j) * decay) * exp(i + j)
    """
    rows = shape[0]
    cols = shape[1]
    size = rows * cols
    result = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            average = (i + j) * 0.5
            distance = abs(i - j)
            coupling = coupling_multiplier * math.exp(-distance * distance_decay - average * coupling_decay)
            result[i, j] = coupling

    return result

def test_nearest_neighbour():
    topography = nearest_neighbour_topography((4, 4), 1.0, 0.1)
    print(f"{topography}")

    src = np.zeros((4,4))
    src[0, 1] = 3.0
    src[2, 2] = 2.0
    print(f"{src}")

    shape = src.shape
    src.shape = (1, 16)
    dest = src.dot(topography)
    dest.shape = shape
    print(f"{dest}")

def test_exponential():
    topography = exponential_topography((6, 6), 1.0, 1.0)
    print("test_exponential")
    print(f"{topography}")

    # src = np.zeros((6,6))
    # src[0, 1] = 3.0
    # print(f"{src}")

    # shape = src.shape
    # src.shape = (1, 36)
    # dest = src.dot(topography)
    # dest.shape = shape
    # print(f"{dest}")

def test_fast_exponential():
    topography = fast_exponential_topography((6, 6), 1.0, 1.0)
    print("test_fast_exponential")
    print(f"{topography}")

    slow = exponential_topography((6, 6), 1.0, 1.0)

    for row1, row2 in zip(slow, topography):
        assert np.allclose(row1, row2) 

    assert np.allclose(slow, topography)

def test_stratified_topography():
    topography = stratified_topography((6, 6), 1.0, 1.0, 1.0)
    print("test_stratified_topography")
    print(f"{topography}")

if __name__ == "__main__":
    test_nearest_neighbour()
    test_exponential()
    # test_fast_exponential()
    test_stratified_topography()


