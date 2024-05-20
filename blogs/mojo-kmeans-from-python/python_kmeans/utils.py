import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as decomp
import sklearn.impute as impute
import os

def plot_bench(mat, x_range, x_label, fig_label, filename):
    plt.figure(figsize=(10, 8))
    # Plot each column as a separate line with different markers
    plt.plot(x_range, mat[:, 0], marker='o', label=f'Mojo KMeans')
    plt.plot(x_range, mat[:, 1], marker='o', label=f'Python+NumPy KMeans')

    # Set labels and title with adjusted font size
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Time (ms)', fontsize=14)
    plt.title("Execution time: Mojo vs Python+NumPy KMeans")

    # Show legend
    plt.legend()
    plt.figtext(0.5, 0.01, fig_label, ha='center', fontsize=12)

    # Show plot
    plt.grid(True)

    path, fname = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(filename)

def plot_speedups(mat, x_range, x_label, fig_label, filename):
    # Plotting the bar chart on the right subplot
    speedups = mat[:, 1]/mat[:, 0]
    _, ax = plt.subplots(figsize=(12, 8))
    num_features = list(range(1, len(speedups) + 1))
    ax.bar(num_features, speedups, color='skyblue')
    for i in range(len(speedups)):
        text = f"{speedups[i]:.1f}x"
        plt.text(num_features[i], speedups[i], text, ha='center', va='bottom', fontsize=14)
    ax.set_title('Speedup: Mojo vs. Python+NumPy KMeans', fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Speedup', fontsize=14)
    plt.xticks(num_features, x_range)
    ax.grid(True)
    plt.figtext(0.5, 0.01, fig_label, ha='center', fontsize=12)
    path, fname = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(filename)

def plot_clusters(X, y, centroids_mojo, centroids_python, centroids_actual):
    X[np.isinf(X)] = np.nanmax(np.abs(X[np.isfinite(X)]))
    y[np.isinf(y)] = np.nanmax(np.abs(y[np.isfinite(y)]))

    pca = decomp.PCA(n_components=2, whiten=True, svd_solver="randomized")
    pca.fit(X)
    X_pca = pca.transform(X)
    X_pca[np.isinf(X_pca)] = np.nanmax(np.abs(X_pca[np.isfinite(X_pca)]))

    centroids_mojo_pca = pca.transform(centroids_mojo)
    centroids_python_pca = pca.transform(centroids_python)
    centroids_actual_pca = pca.transform(centroids_actual)

    # Get unique labels
    unique_labels = np.unique(y)

    # Define colors for each label
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(10, 8))

    # Plot each cluster
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        color = colors[i]
        cluster_points = X_pca[y == label]
        plt.scatter(np.take(cluster_points, 0, axis=1), np.take(cluster_points, 1, axis=1), color=color, label='Cluster'+str(label))

    plt.scatter(np.take(centroids_actual_pca, 0, axis=1), np.take(centroids_actual_pca, 1, axis=1), marker='o', color='red', facecolors='red', s=100, label='Actual centroids')
    plt.scatter(np.take(centroids_mojo_pca, 0, axis=1), np.take(centroids_mojo_pca, 1, axis=1), marker='^', color='orange', facecolors='orange', s=100, label='Mojo centroids')
    plt.scatter(np.take(centroids_python_pca, 0, axis=1), np.take(centroids_python_pca, 1, axis=1), marker='*', color='blue', facecolors='blue', s=100, label='Python centroids')

    plt.title('Scatter Plot of Data Points with Centroids')
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()