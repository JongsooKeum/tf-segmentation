from abc import abstractmethod, abstractproperty, ABCMeta
import numpy as np
from sklearn.metrics import accuracy_score

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

class AccuracyEvaluator(Evaluator):
    """ Evaluator with Recall metric."""
    @property
    def worst_score(self):
        """The worst performance score."""
        return 0.0

    @property
    def mode(self):
        """The mode for performance score."""
        return 'max'

    def score(self, y_true, y_pred):
        """Compute Pixel Accuracy for a given predicted mask"""
        acc = []
        for t, p in zip(y_true, y_pred):
            # ignore unknwon region
            ignore = np.where(t[...,0].reshape(-1) == -1)
            acc.append(accuracy_score(np.delete(t.argmax(axis=-1).reshape(-1), ignore[0]),
                                      np.delete(p.argmax(axis=-1).reshape(-1), ignore[0])))
        return sum(acc)/len(acc)

    def is_better(self, curr, best, **kwargs):
        """
        Return whether current performance scores is better than current best,
        with consideration of the relative threshold to the given performance score.
        :param kwargs: dict, extra arguments.
            - score_threshold: float, relative threshold for measuring the new optimum,
                               to only focus on significant changes.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps