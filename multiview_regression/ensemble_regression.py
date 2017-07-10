from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.base import RegressorMixin, BaseEstimator, clone
from sklearn.externals import six
import numpy as np


def _parallel_fit_estimator(estimator, X, y):
    """Private function used to fit an estimator within a job."""
    estimator.fit(X, y)
    return estimator


class MeanRegression(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, n_jobs=4):
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_estimator)(clone(clf), X, y)
            for _, clf in self.estimators)
        return self

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        check_is_fitted(self, 'estimators_')
        return self._predict(X)

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        predictions = [clf.predict(X).reshape(len(X))
                       for clf in self.estimators_]
        return np.average(predictions, axis=0)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(MeanRegression, self).get_params(deep=False)
        else:
            out = super(MeanRegression, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            for name, step in six.iteritems(self.named_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out