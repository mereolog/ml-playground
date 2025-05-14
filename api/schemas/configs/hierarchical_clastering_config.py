from schemas.configs.algorithms_configs import SupervisedAlgorithmsParams
from dataclasses import dataclass
from typing import Literal, Optional

import get_args


LinkageType = Literal["single", "complete", "average"]
MetricType = Literal["euclidean", "manhattan", "cosine", "chebyshev", "minkowski"] 

@dataclass
class HierarchicalClusteringParams(SupervisedAlgorithmsParams):
    """
    Parameters for the hierarchical clustering algorithm.
    
    Attributes:
        n_clusters: Number of clusters to form (default: 2).
        linkage: Method for linking clusters ('single', 'complete', 'average').
        distance_threshold: Maximum allowed distance for clusters; if None, n_clusters is used.
        metric: Distance metric used for calculations (e.g., 'euclidean', 'manhattan').
    """
    
    n_clusters: Optional[int] = 2
    linkage: LinkageType = "average"
    distance_threshold: float = 1.0
    metric: str = "euclidean"
    
    def __post_init__(self):
        if self.n_clusters is not None and self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1 or None if using distance_threshold.")
        if self.distance_threshold is not None and self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be greater than 0 if set.")
        if self.metric not in get_args(MetricType):
            raise ValueError(f"Invalid metric '{self.metric}'. Must be one of: {get_args(MetricType)}.")
        if self.linkage not in get_args(LinkageType):
            raise ValueError(f"Invalid linkage '{self.linkage}'. Must be one of: {get_args(LinkageType)}.")
