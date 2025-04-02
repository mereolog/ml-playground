from dataclasses import dataclass, field 
from typing import Literal, Optional 
from algorithms_configs import SupervisedAlgorithmsParams

@dataclass
class NaiveBayesParams(SupervisedAlgorithmsParams):
    """This model parameters config defines types and possible default arguments used by Naive Bayes model

    Attributes:
        alpha: Smoothing parameter (default: 1.0)
        binarize: Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to be binary (default: 0.0)
        fit_prior: Whether to learn class prior probabilities or not (default: True)
        class_prior: Prior probabilities of the classes. If specified, the priors are not adjusted according to the data (default: None)
    """

    alpha: float = 1.0
    binarize: Optional[float] = 0.0
    fit_prior: bool = True
    class_prior: Optional[list] = None

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError("alpha must be greater than or equal to 0")
        if self.binarize is not None and self.binarize < 0:
            raise ValueError("binarize must be greater than or equal to 0 or None")
        

