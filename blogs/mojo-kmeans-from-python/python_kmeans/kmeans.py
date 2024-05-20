import numpy as np
from numpy.random import seed

class Kmeans:
    def __init__(self, 
                 k=8, 
                 max_iterations=300, 
                 tol=1e-4, 
                 rng_seed=42, 
                 verbose=True, 
                 run_till_max_iter=False):
        self.k=k  # Number of clusters
        self.max_iterations=max_iterations # Maximum number of iterations to run the algorithm
        self.tol=tol  # Tolerance for convergence. Stop if the change in inertia is less than tol.
        self.centroids=[]  # List to store the centroids of clusters
        self.inertia=0.0  # Measure of the total distance of each point to its assigned centroid
        self.verbose=verbose  # Controls whether to print detailed debug statements
        self.run_till_max_iter=run_till_max_iter  # Run till max_iterations even if converged
        seed(rng_seed) # Seed for random number generation, for reproducibility

    # Calculate squared Euclidean distance from each data point to each centroid
    def distance_norm(self, data, centroids_distances):
        centroids_distances[:, :len(self.centroids)] = \
            np.array([[np.linalg.norm(x - c)**2 for c in self.centroids] for x in data])

    # Initialize centroids using the k-means++ algorithm to improve cluster quality
    def _kmeans_plus_plus(self, data):
        # Add the first centroid randomly chosen from the data
        self.centroids.append(data[np.random.randint(data.shape[0])])  

        # Create a full matrix for distance calculations; start with infinity
        centroids_distances=np.full((data.shape[0], self.k), np.inf)

        for idx_c in range(1, self.k):
            # Update distances for all points relative to the new set of centroids
            self.distance_norm(data, centroids_distances)
            # Find the minimum distance to any centroid for each point
            min_distances=centroids_distances.min(axis=1)

            # Probability of selecting next centroid is proportional to squared distance
            probs=min_distances / min_distances.sum()
            # Cumulative probabilities for selection
            cumulative_probs=probs.cumsum() 

            # Select the next centroid based on cumulative probabilities
            rand_prob=np.random.rand()
            for i in range(len(cumulative_probs)):
                if rand_prob < cumulative_probs[i]:
                    self.centroids.append(data[i,:])
                    break

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

        # Initialize centroids using k-means++
        self._kmeans_plus_plus(data)
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