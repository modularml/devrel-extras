from math.math import align_down
from .matrix import Matrix

fn list_to_matrix[dtype: DType](lst: List[Matrix[dtype]]) -> Matrix[dtype]:
    var new_mat = Matrix[dtype](len(lst),len(lst[0]))
    var tmpPtr = new_mat._matPtr
    for arr in lst:
        memcpy(tmpPtr, arr[]._matPtr, len(arr[]))
        tmpPtr += len(arr[])
    return new_mat


# Initialize centroids using the k-means++ algorithm to improve cluster quality
fn kmeans_plus_plus[dtype: DType = DType.float64, simd_width: Int = simdwidthof[dtype]()](data: Matrix[dtype], n_clusters: Int) raises -> List[Matrix[dtype]]:
    # Declare temporary variables
    var probs: Matrix[dtype]
    var cumulative_probs: Matrix[dtype]
    var centroids: List[Matrix[dtype]] = List[Matrix[dtype]](capacity=n_clusters)

    # Add the first centroid randomly chosen from the data
    centroids.append(data[int(random_si64(0, data.rows)),:])

    # Create a full matrix for distance calculations; start with infinity
    var centroids_distances=Matrix[dtype](data.rows, n_clusters, max_or_inf[dtype]())

    for idx_c in range(1, n_clusters):
        # Update distances for all points relative to the new set of centroids
        distance_norm[dtype, simd_width](centroids, data, centroids_distances)
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
                centroids.append(data[i,:])
                break
        
    return centroids


# Calculate squared Euclidean distance from each data point to each centroid
fn distance_norm[dtype: DType, simd_width: Int](centroids: List[Matrix[dtype]], data: Matrix[dtype], 
                inout centroids_distances: Matrix[dtype]):
    var simd_multiple=align_down(data.cols, simd_width)
    @parameter
    fn parallel_sqnorm(idx_centroid: Int):
        var centroid=centroids[idx_centroid]
        for idx_mat_row in range(data.rows):
            var sq_norm=SIMD[dtype, simd_width].splat(0)
            for idx_col in range(0, simd_multiple, simd_width):
                sq_norm += pow(
                    data._matPtr.load[width=simd_width](idx_mat_row*data.cols+idx_col) - 
                    centroid._matPtr.load[width=simd_width](idx_col),2)
            for idx_col in range(simd_multiple, data.cols):
                sq_norm[0] += pow(
                    data._matPtr.load[width=1](idx_mat_row*data.cols+idx_col) - 
                    centroid._matPtr.load[width=1](idx_col),2
                    )
            centroids_distances[idx_mat_row, idx_centroid]=sq_norm.reduce_add()
    parallelize[parallel_sqnorm](len(centroids), info.num_performance_cores())