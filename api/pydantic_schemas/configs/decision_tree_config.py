from typing import Optional

from pydantic import Field
from pydantic_schemas.configs.algorithm_configs import SupervisedAlgorithmsParams


class DecisionTreeParams(SupervisedAlgorithmsParams):
    """
    Pydantic model for Decision Tree parameters.

    Attributes:
        max_tree_depth: Maximum depth of a tree (default: None). Must be a positive integer if set.
        min_leaf_samples: Minimal number of samples needed to create new leaf (default: 1). Must be > 0.
        min_split_samples: Minimal number of samples needed to make a split (default: 2). Must be > 0.
    """

    max_tree_depth: Optional[int] = Field(
        default=None,
        description="Maximum depth of a tree. If set, must be a positive integer.",
        gt=0,
    )
    min_leaf_samples: int = Field(
        default=1,
        description="Minimal number of samples needed to create new leaf.",
        gt=0,
    )
    min_split_samples: int = Field(
        default=2, description="Minimal number of samples needed to make a split.", gt=0
    )
