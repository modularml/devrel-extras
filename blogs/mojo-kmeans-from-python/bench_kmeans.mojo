from mojo_kmeans import Matrix, Kmeans
from mojo_kmeans.utils import list_to_matrix, kmeans_plus_plus
from time import now
from python import Python
from benchmark import Unit, keep
import benchmark

def bench_mojo_kmeans(centroids: List[Matrix[DType.float64]], borrowed data: Matrix[DType.float64]) -> Float64:
    mojo_model = Kmeans(centroids, max_iterations=10, verbose=False, run_till_max_iter=True)
    t = now()
    _ = mojo_model.fit(data)
    t_mojo = Float64(now()-t)/1_000_000
    return t_mojo

def bench_python_kmeans(centroids: PythonObject, borrowed data: PythonObject) -> Float64:
    Python.add_to_path(".")
    py_kmeans = Python.import_module("python_kmeans")
    py_model = py_kmeans.Kmeans(centroids, max_iterations=10, verbose=False, run_till_max_iter=True)
    t = now()
    _ = py_model.fit(data)
    t_py = Float64(now()-t)/1_000_000
    return t_py

def bench_sklearn_kmeans(centroids: PythonObject, borrowed data: PythonObject) -> Float64:
    sklearn_cluster = Python.import_module("sklearn.cluster")
    sklearn_model = sklearn_cluster.KMeans(n_clusters=centroids.shape[0], 
                                            max_iter=10,
                                            verbose=False,
                                            tol=0,
                                            init=centroids)
    t = now()
    _ = sklearn_model.fit(data)
    return Float64(now()-t)/1_000_000

def get_dataset(n_clusters: PythonObject, n_samples: PythonObject, n_features: PythonObject) -> PythonObject:
    sklearn_datasets = Python.import_module("sklearn.datasets")
    data_numpy = sklearn_datasets.make_blobs(n_samples=n_samples, 
                                    cluster_std=8, 
                                    centers=n_clusters, 
                                    n_features=n_features, 
                                    return_centers=True,
                                    random_state=int(now()/1e10))
    return data_numpy

def main():
    np = Python.import_module("numpy")    
    Python.add_to_path(".")
    py_utils = Python.import_module("python_kmeans.utils")
    
    clusters_range = np.arange(5,185,15)
    samples_range = np.arange(2000,24000,2000)
    features_range = np.arange(200,4000,200)

    verbose = True
    benchImageDir = "benchdir"

    res1 = Matrix(int(len(clusters_range)),2)
    for idx in range(int(len(clusters_range))):
        data = get_dataset(clusters_range[idx],samples_range[0],features_range[0])[0]
        mojo_data = Matrix.from_numpy(data)
        centroids = kmeans_plus_plus[DType.float64, 4*simdwidthof[DType.float64]()](mojo_data, int(clusters_range[idx]))
        py_centroids = list_to_matrix(centroids).to_numpy()

        res1[idx,0] = bench_mojo_kmeans(centroids,mojo_data)
        res1[idx,1] = bench_sklearn_kmeans(py_centroids, data)
    x_label = "Numbers of Clusters"
    fig_label = "Samples: "+str(samples_range[0])+" Features: "+str(features_range[0])
    py_utils.plot_bench(res1.to_numpy(), 
                        clusters_range, 
                        x_label, 
                        fig_label,
                        benchImageDir+"/num_clusters.png")
    py_utils.plot_speedups(res1.to_numpy(),
                            clusters_range,
                            x_label,
                            fig_label,
                            benchImageDir+"/num_clusters_speedup.png")
    print("Numbers of Clusters speedups:")
    print(res1[:,1]/res1[:,0])

    res2 = Matrix(int(len(samples_range)),2)
    for idx in range(int(len(samples_range))):
        data = get_dataset(clusters_range[5],samples_range[idx],features_range[0])[0]
        mojo_data = Matrix.from_numpy(data)
        centroids = kmeans_plus_plus[DType.float64, 4*simdwidthof[DType.float64]()](mojo_data, int(clusters_range[5]))
        py_centroids = list_to_matrix(centroids).to_numpy()

        res2[idx,0] = bench_mojo_kmeans(centroids,mojo_data)
        res2[idx,1] = bench_sklearn_kmeans(py_centroids, data)
    x_label = "Number of Samples"
    fig_label = "Clusters: "+str(clusters_range[5])+" Features: "+str(features_range[0])
    py_utils.plot_bench(res2.to_numpy(), 
                        samples_range, 
                        x_label, 
                        fig_label,
                        benchImageDir+"/num_samples.png")
    py_utils.plot_speedups(res2.to_numpy(),
                            samples_range,
                            x_label,
                            fig_label,
                            benchImageDir+"/num_samples_speedup.png")
    print("Number of Samples speedups:")
    print(res2[:,1]/res2[:,0])

    res3 = Matrix(int(len(features_range)),2)
    for idx in range(int(len(features_range))):
        data = get_dataset(clusters_range[0],samples_range[1],features_range[idx])[0]
        mojo_data = Matrix.from_numpy(data)
        centroids = kmeans_plus_plus[DType.float64, 4*simdwidthof[DType.float64]()](mojo_data, int(clusters_range[0]))
        py_centroids = list_to_matrix(centroids).to_numpy()

        res3[idx,0] = bench_mojo_kmeans(centroids, mojo_data)
        res3[idx,1] = bench_sklearn_kmeans(py_centroids, data)
    x_label = "Number of Features"
    fig_label = "Samples: "+str(samples_range[1])+" Clusters: "+str(clusters_range[0])
    py_utils.plot_bench(res3.to_numpy(), 
                        features_range, 
                        x_label, 
                        fig_label, 
                        benchImageDir+"/num_features.png")
    py_utils.plot_speedups(res3.to_numpy(),
                            features_range,
                            x_label,
                            fig_label,
                            benchImageDir+"/num_features_speedup.png")
    print("Number of Features speedups:")
    print(res3[:,1]/res3[:,0])