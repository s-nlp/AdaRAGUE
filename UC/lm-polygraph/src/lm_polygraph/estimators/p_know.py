import numpy as np

from typing import Dict

from .estimator import Estimator


class PIKnow(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "P(IKnow)" as provided in the paper https://arxiv.org/abs/2207.05221.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question q and model generation a, the method uses the following prompt:
        Question: {q}
    and calculates the log-probability of 'I Know' token with minus sign.
    """

    def __init__(self):
        super().__init__(["p_iknow"], "sequence")

    def __str__(self):
        return "PIKnow"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates minus log-probability of True token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_iknow'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        piknown = stats["p_iknow"]
        return -np.array(piknown)
