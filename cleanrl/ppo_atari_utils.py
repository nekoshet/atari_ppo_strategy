import numpy as np


def get_surrounding_window(array, i, j, k):
    """
    Extract a kxkxc window around point (i,j) from a 3D array.

    Parameters:
    array (numpy.ndarray): Input 3D array of shape (n, n, c)
    i (int): Row coordinate of center point
    j (int): Column coordinate of center point
    k (int): Size of window (must be odd)

    Returns:
    numpy.ndarray: kxkxc window around (i,j), padded with zeros if out of bounds
    """
    if k % 2 == 0:
        raise ValueError("k must be odd")

    n, _, c = array.shape
    offset = k // 2

    # Create empty window
    window = np.zeros((k, k, c), dtype=array.dtype)

    # Calculate boundaries for source array
    row_start = max(0, i - offset)
    row_end = min(n, i + offset + 1)
    col_start = max(0, j - offset)
    col_end = min(n, j + offset + 1)

    # Calculate boundaries for target window
    win_row_start = offset - (i - row_start)
    win_row_end = offset + (row_end - i)
    win_col_start = offset - (j - col_start)
    win_col_end = offset + (col_end - j)

    # Copy valid region
    window[win_row_start:win_row_end,
    win_col_start:win_col_end,
    :] = array[row_start:row_end,
         col_start:col_end,
         :]

    return window


if __name__ == '__main__':
    arr = np.arange(50).reshape(5, 5, 2)
    print(arr[..., 1])
    i, j = 4, 1
    k = 3
    win = get_surrounding_window(arr, i, j, 3)
    print(win[..., 1])

    out = np.zeros_like(arr)
    out[:win.shape[0], :win.shape[1]] = win
    print(out[..., 1])
