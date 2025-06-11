from typing import Literal, Optional
from pydantic import BaseModel, Field

from schemas.configs.algorithm_configs import SupervisedAlgorithmsParams

LinkageType = Literal["single", "complete", "average"]
MetricType = Literal["euclidean", "manhattan", "cosine", "chebyshev", "minkowski"]

class HierarchicalClusteringParams(SupervisedAlgorithmsParams, BaseModel):
    """
    Parameters for the hierarchical clustering algorithm.
    """

    n_clusters: Optional[int] = Field(
        default=2,
        ge=1,        
        description=("Number of clusters to form"),
    )
    linkage: LinkageType = Field(
        default="average",
        description="Method for linking clusters.",
    )
    distance_threshold: Optional[float] = Field(
        default=1.0,
        gt=0,      
        description=("Maximum allowed distance between clusters"),
    )
    metric: MetricType = Field(
        default="euclidean",
        description="Distance metric used for calculations.",
    )
