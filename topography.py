import numpy as np

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

if __name__ == "__main__":
    test_nearest_neighbour()


