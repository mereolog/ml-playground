from typing import List, Optional

from pydantic import Field
from schemas.configs.algorithm_configs import SupervisedAlgorithmsParams


class NaiveBayesParams(SupervisedAlgorithmsParams):
    """
    Pydantic model for Naive Bayes algorithm parameters.
    This model parameters config defines types and possible default arguments used by Naive Bayes model.

    Attributes:
        alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing). (default: 1.0)
        binarize: Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to be already binary. (default: 0.0)
        fit_prior: Whether to learn class prior probabilities or not. If false, a uniform prior will be used. (default: True)
        class_prior: Prior probabilities of the classes. If specified the priors are not adjusted according to the data. (default: None)
    """

    alpha: float = Field(
        default=1.0,
        description="Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).",
        ge=0.0,
    )
    binarize: Optional[float] = Field(
        default=0.0,
        description="Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to be already binary. Values greater than the threshold are set to 1 and values less than or equal to the threshold are set to 0.",
        ge=0.0,
    )
    fit_prior: bool = Field(
        default=True,
        description="Whether to learn class prior probabilities or not. If false, a uniform prior will be used.",
    )
    class_prior: Optional[List[float]] = Field(
        default=None,
        description="Prior probabilities of the classes. If specified the priors are not adjusted according to the data. The number of elements in class_prior must match the number of classes.",
    )
