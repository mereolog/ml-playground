import numpy as np
from scipy.cluster.hierarchy import linkage
from typing import List, Union

def perform_hierarchical_clustering(
    data: List[List[Union[int, float]]],
    method: str = 'average',
    metric: str = 'euclidean'
) -> List[List[float]]:
    """
    Performs hierarchical clustering on the input data.

    Args:
        data: Input data as a list of lists (samples x features).
        method: The linkage criterion to use. See 
scipy.cluster.hierarchy.linkage docs.
        metric: The distance metric to use. See 
scipy.spatial.distance.pdist docs.

    Returns:
        The linkage matrix (N-1)x4 as a list of lists.
        Each row [i, j, distance, count] represents the merging of 
clusters i and j.

    Raises:
        ValueError: If the input data contains less than 2 samples.
    """
    data_np = np.array(data)

    if data_np.shape[0] < 2:
        raise ValueError("Input data must contain at least 2 samples for 
clustering.")

    # Perform hierarchical clustering
    linkage_matrix = linkage(data_np, method=method, metric=metric)

    # Convert NumPy matrix to list of lists for JSON serialization
    return linkage_matrix.tolist()

# No example usage with if name == 'main': as requested
