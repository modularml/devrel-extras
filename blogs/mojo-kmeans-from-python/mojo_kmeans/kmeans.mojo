from .matrix import Matrix
from .utils import list_to_matrix
from algorithm import vectorize, parallelize
from random import random_si64, random_float64, seed
from memory import memcpy
from sys import info
import math
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
                k: Int=8, 
                max_iterations: Int=300, 
                tol: Scalar[dtype]=1e-4, 
                rng_seed: Int=42,
                verbose: Bool=True,
                run_till_max_iter: Bool=False):
        self.k=k  # Number of clusters
        self.max_iterations=max_iterations # Maximum number of iterations to run the algorithm
        self.tol=tol  # Tolerance for convergence. Stop if the change in inertia is less than tol.
        self.centroids=List[Matrix[dtype]](capacity=k) # List to store the centroids of clusters
        self.inertia=0.0  # Measure of the total distance of each point to its assigned centroid
        self.verbose=verbose  # Controls whether to print detailed debug statements
        self.run_till_max_iter=run_till_max_iter  # Run till max_iterations even if converged
        seed(rng_seed) # Seed for random number generation, for reproducibility

    # Calculate squared Euclidean distance from each data point to each centroid
    fn distance_norm(self, data: Matrix[dtype], 
                    inout centroids_distances: Matrix[dtype]):
        alias simd_width=self.simd_width
        var simd_multiple=math.align_down(data.cols, simd_width)
        @parameter
        fn parallel_sqnorm(idx_centroid: Int):
            var centroid=self.centroids[idx_centroid]
            for idx_mat_row in range(data.rows):
                var sq_norm=SIMD[dtype, simd_width].splat(0)
                for idx_col in range(0, simd_multiple, simd_width):
                    sq_norm += math.pow(
                        data._matPtr.load[width=simd_width](idx_mat_row*data.cols+idx_col) - 
                        centroid._matPtr.load[width=simd_width](idx_col),2)
                for idx_col in range(simd_multiple, data.cols):
                    sq_norm[0] += math.pow(
                        data._matPtr.load[width=1](idx_mat_row*data.cols+idx_col) - 
                        centroid._matPtr.load[width=1](idx_col),2
                        )
                centroids_distances[idx_mat_row, idx_centroid]=sq_norm.reduce_add()
        parallelize[parallel_sqnorm](len(self.centroids), info.num_performance_cores())

    # Initialize centroids using the k-means++ algorithm to improve cluster quality
    fn _kmeans_plus_plus(inout self, data: Matrix[dtype]) raises:
        # Declare temporary variables
        var probs: Matrix[dtype]
        var cumulative_probs: Matrix[dtype]

        # Add the first centroid randomly chosen from the data
        self.centroids.append(data[int(random_si64(0,data.rows)),:])

        # Create a full matrix for distance calculations; start with infinity
        var centroids_distances=Matrix[dtype](data.rows, self.k, math.limit.inf[dtype]())

        for idx_c in range(1, self.k):
            # Update distances for all points relative to the new set of centroids
            self.distance_norm(data, centroids_distances)
            # Find the minimum distance to any centroid for each point
            var min_distances=centroids_distances.min[axis=1]() 

            # Probability of selecting next centroid is proportional to squared distance
            probs=min_distances / min_distances.sum()
            # Cumulative probabilities for selection
            cumulative_probs=probs.cumsum() 

            # Select the next centroid based on cumulative probabilities
            var rand_prob=random_float64().cast[dtype]()
            for i in range(len(cumulative_probs)):
                if rand_prob < cumulative_probs[i]:
                    self.centroids.append(data[i,:])
                    break

    # Lloyd's algorithm: Recompute centroids and assign points to the nearest cluster
    fn _lloyds_iteration(inout self, data: Matrix[dtype], 
                                    inout centroids_distances: Matrix[dtype]) raises:
        # Update distances based on current centroids
        self.distance_norm(data, centroids_distances)
        # Assign each point to the nearest centroid
        var labels=centroids_distances.argmin(axis=1)

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
        var centroids_distances=Matrix[dtype](data.rows, self.k, math.limit.inf[dtype]())

        # Initialize centroids using k-means++
        self._kmeans_plus_plus(data)
        var previous_inertia=self.inertia

        for i in range(self.max_iterations):
            if self.verbose:
                print("Iteration:", i, end='')
            previous_inertia=self.inertia

            # Perform an iteration of Lloyd's algorithm
            self._lloyds_iteration(data, centroids_distances)

            if math.abs(previous_inertia - self.inertia) < self.tol and not self.run_till_max_iter:
                print("Converged at iteration:",i,
                "Inertia change less than tol:",self.tol,
                "\n Final inertia:",self.inertia)
                break
            if i == self.max_iterations:
                print("Converged at iteration:",i,"Max iterations reached",self.max_iterations) 
        return self.centroids