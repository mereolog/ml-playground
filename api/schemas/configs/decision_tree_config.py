from schemas.configs.algorithms_configs import SupervisedAlgorithmsParams
from dataclasses import dataclass
from typing import Optional

@dataclass
class DecisionTreeParams(SupervisedAlgorithmsParams):

    """
    Attributes:
        max_tree_depth: Maximum depth of a tree (default: None)
        min_leaf_samples: Minimal number of samples needed to create new leaf (default: 1)
        min_split_samples: Minimal number of samples needed to make a split (default: 2)


    """

    max_tree_depth: Optional[int] = None
    min_leaf_samples: int = 1
    min_split_samples: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.max_tree_depth <= 0:
            raise ValueError("Must be positive integer.")