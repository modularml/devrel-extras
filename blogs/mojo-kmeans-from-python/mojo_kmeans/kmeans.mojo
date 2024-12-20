from .matrix import Matrix
from utils.numerics import max_or_inf
from .utils import list_to_matrix, kmeans_plus_plus, distance_norm
from algorithm import vectorize, parallelize
from random import random_si64, random_float64, seed
from memory import memcpy
from sys import info
from time import now
from python import Python

struct Kmeans[dtype: DType=DType.float64]():
    var k: Int
    var max_iterations: Int
    var tol: Scalar[dtype]
    var centroids: List[Matrix[dtype]]
    var inertia: Scalar[dtype]
    var verbose: Bool
    var run_till_max_iter: Bool
    alias simd_width: Int=4*simdwidthof[dtype]()

    fn __init__(inout self,
                centroids: List[Matrix[dtype]],
                max_iterations: Int=300, 
                tol: Scalar[dtype]=1e-4, 
                rng_seed: Int=42,
                verbose: Bool=True,
                run_till_max_iter: Bool=False):
        self.k=len(centroids)  # Number of clusters
        self.max_iterations=max_iterations # Maximum number of iterations to run the algorithm
        self.tol=tol  # Tolerance for convergence. Stop if the change in inertia is less than tol.
        self.centroids=centroids # List to store the centroids of clusters
        self.inertia=0.0  # Measure of the total distance of each point to its assigned centroid
        self.verbose=verbose  # Controls whether to print detailed debug statements
        self.run_till_max_iter=run_till_max_iter  # Run till max_iterations even if converged
        seed(rng_seed) # Seed for random number generation, for reproducibility

    # Lloyd's algorithm: Recompute centroids and assign points to the nearest cluster
    fn _lloyds_iteration(inout self, data: Matrix[dtype], 
                                    inout centroids_distances: Matrix[dtype]) raises:
        # Update distances based on current centroids
        distance_norm[dtype, self.simd_width](self.centroids, data, centroids_distances)
        # Assign each point to the nearest centroid
        var labels=centroids_distances.argmin()

        # Recalculate centroids as the mean of all points assigned to each centroid
        for idx in range(self.k):
            # Check if any points are assigned to this cluster
            self.centroids[idx]=data.mean(where=labels==idx) if (labels==idx).sum() > 0 else self.centroids[idx]
        # Compute new inertia as the sum of squared distances to the nearest centroid
        self.inertia=(centroids_distances.min[axis=1]()).sum()
        if self.verbose:
            print(", inertia:", self.inertia)

    # Main method to fit the k-means model to the provided data
    fn fit(inout self, data: Matrix[dtype]) raises -> List[Matrix[dtype]]:

        # Initialize distance matrix
        var centroids_distances=Matrix[dtype](data.rows, self.k, max_or_inf[dtype]())

        var previous_inertia=self.inertia

        for i in range(self.max_iterations):
            if self.verbose:
                print("Iteration:", i, end='')
            previous_inertia=self.inertia

            # Perform an iteration of Lloyd's algorithm
            self._lloyds_iteration(data, centroids_distances)

            if abs(previous_inertia - self.inertia) < self.tol and not self.run_till_max_iter:
                print("Converged at iteration:",i,
                "Inertia change less than tol:",self.tol,
                "\n Final inertia:",self.inertia)
                break
            if i == self.max_iterations:
                print("Converged at iteration:",i,"Max iterations reached",self.max_iterations) 
        return self.centroids