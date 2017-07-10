from sklearn.base import TransformerMixin
import numpy as np


class Metric:
    def value(self, word):
        raise NotImplemented


class FreqMetric(Metric):
    def __init__(self, freq):
        self.freq = freq

    def value(self, word):
        if word in self.freq:
            return self.freq[word]
        else:
            return 0


class GradeDic(Metric):
    def __init__(self, freq):
        self.freq = freq

    def value(self, word):
        if word in self.freq:
            return self.freq[word]
        else:
            return 5


class WordLength(Metric):
    def value(self, word):
        return len(word)


class FreqSubtlex(Metric):
    def __init__(self, freq):
        self.freq = freq

    def value(self, word):
        if word in self.freq:
            return [self.freq[word]['wf'], self.freq[word]['cd']]
        else:
            return [0, 0]


class LinguisticTransform(TransformerMixin):
    def __init__(self, metrics):
        self.metrics = metrics

    def transform(self, X, y=None, **fit_params):
        out_m = []

        for example in X:
            values = []
            for metric in self.metrics:
                result = metric.value(example[0])
                if isinstance(result, list):
                    for v in result:
                        values.append(v)
                else:
                    values.append(metric.value(example[0]))
            out_m.append(values)
        return np.asarray(out_m).reshape(len(out_m), -1)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self