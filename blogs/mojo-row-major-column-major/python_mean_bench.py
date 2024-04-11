from timeit import timeit
import numpy as np

def row_major_mean(np_mat: np.ndarray, mean_idx=[]):
    if len(mean_idx) == 0:
        mean_idx = slice(None)
    return 1000*timeit(lambda: np_mat[:,mean_idx].mean(axis=0), number=10) / 10

def col_major_mean(np_mat: np.ndarray, mean_idx=[]):
    if len(mean_idx) == 0:
        mean_idx = slice(None)
    np_mat = np.asfortranarray(np_mat)
    return 1000*timeit(lambda: np_mat[:,mean_idx].mean(axis=0), number=10) / 10

if __name__ == "__main__":
    np_mat = np.random.rand(10000,10000)
    print()
    print("Mean of all columns, time (ms)")
    print("------------------------------")
    print("Row-major:", row_major_mean(np_mat))
    print("Col-major:",col_major_mean(np_mat))
    row_major_mean_vals = np_mat.mean(axis=0)
    col_major_mean_vals = np.asfortranarray(np_mat).mean(axis=0)
    np.testing.assert_almost_equal(row_major_mean_vals, col_major_mean_vals)
    print("Accuracy comparision (2-Norm of difference):", np.linalg.norm(row_major_mean_vals - col_major_mean_vals))
    print()
    print("Mean of 1000 random columns, time (ms)")
    print("--------------------------------------")
    mean_idx = np.random.randint(0,10000,1000)
    print("Row-major:", row_major_mean(np_mat,mean_idx))
    print("Col-major:",col_major_mean(np_mat,mean_idx))
    row_major_mean_vals = np_mat[:,mean_idx].mean(axis=0)
    col_major_mean_vals = np.asfortranarray(np_mat[:,mean_idx]).mean(axis=0)
    np.testing.assert_almost_equal(row_major_mean_vals, col_major_mean_vals)
    print("Accuracy comparision (2-Norm of difference):", np.linalg.norm(row_major_mean_vals - col_major_mean_vals))
    print()