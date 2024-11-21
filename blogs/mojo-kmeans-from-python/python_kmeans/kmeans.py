import numpy as np
from numpy.random import seed

class Kmeans:
    def __init__(self, 
                 centroids, 
                 max_iterations=300, 
                 tol=1e-4, 
                 rng_seed=42, 
                 verbose=True, 
                 run_till_max_iter=False):
        self.k=len(centroids)  # Number of clusters
        self.max_iterations=max_iterations # Maximum number of iterations to run the algorithm
        self.tol=tol  # Tolerance for convergence. Stop if the change in inertia is less than tol.
        self.centroids=centroids  # List to store the centroids of clusters
        self.inertia=0.0  # Measure of the total distance of each point to its assigned centroid
        self.verbose=verbose  # Controls whether to print detailed debug statements
        self.run_till_max_iter=run_till_max_iter  # Run till max_iterations even if converged
        seed(rng_seed) # Seed for random number generation, for reproducibility

    # Calculate squared Euclidean distance from each data point to each centroid
    def distance_norm(self, data, centroids_distances):
        centroids_distances[:, :self.centroids.shape[0]] = \
            np.array([[np.linalg.norm(x - c)**2 for c in self.centroids] for x in data])

    # Lloyd's algorithm: Recompute centroids and assign points to the nearest cluster
    def _lloyds_iteration(self, data, centroids_distances):
        
        # Update distances based on current centroids
        self.distance_norm(data, centroids_distances)
        # Assign each point to the nearest centroid
        labels=centroids_distances.argmin(axis=1) 

        # Recalculate centroids as the mean of all points assigned to each centroid
        for idx in range(self.k):
            # Check if any points are assigned to this cluster
            self.centroids[idx]=data[labels == idx].mean(axis=0) if np.sum(labels == idx) > 0 else self.centroids[idx]
        # Compute new inertia as the sum of squared distances to the nearest centroid
        self.inertia=(centroids_distances.min(axis=1)).sum()
        if self.verbose:
            print(", inertia:", self.inertia)

    # Main method to fit the k-means model to the provided data
    def fit(self, data):

        # Initialize distance matrix
        centroids_distances=np.full((data.shape[0], self.k), np.inf)  

        previous_inertia=self.inertia

        for i in range(self.max_iterations):
            if self.verbose:
                print("Iteration:", i, end='')
            previous_inertia=self.inertia

            # Perform an iteration of Lloyd's algorithm
            self._lloyds_iteration(data, centroids_distances)

            if np.abs(previous_inertia - self.inertia) < self.tol and not self.run_till_max_iter:
                print("Converged at iteration:",i,
                "Inertia change less than tol:",self.tol,
                "\n Final inertia:",self.inertia)
                break
            if i == self.max_iterations:
                print("Converged at iteration:",i,"Max iterations reached",self.max_iterations) 
        return self.centroids