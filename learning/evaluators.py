from abc import abstractmethod, abstractproperty, ABCMeta
import numpy as np
from learning.utils import computeIoU

class Evaluator(metaclass=ABCMeta):
    """Base class for evaluation functions."""

    @abstractproperty
    def worst_score(self):
        """
        The worst performance score.
        :return float.
        """
        pass
    @abstractproperty
    def mode(self):
        """
        the mode for performance score, either 'max' or 'min'
        e.g. 'max' for accuracy, AUC, precision and recall,
              and 'min' for error rate, FNR and FPR.
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Performance metric for a given prediction.
        This should be implemented.
        :param y_true: np.ndarray, shape: (N, 5 + num_classes).
        :param y_pred: np.ndarray, shape: (N, 5 + num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        Function to return whether current performance score is better than current best.
        This should be implemented.
        :param curr: float, current performance to be evaluated.
        :param best: float, current best performance.
        :return bool.
        """
        pass

class ErrorRateEvaluator(Evaluator):

    @property
    def worst_score(self):

        return 0.0

    @property
    def mode(self):

        return 'max'

    def score(self, y_true, y_pred):

        return computeIoU(y_pred, y_true)

    def is_better(self, curr, best, **kwargs):

        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps