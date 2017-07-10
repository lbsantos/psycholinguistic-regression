from sklearn.base import TransformerMixin
import numpy as np


class W2VTransformer(TransformerMixin):
    def __init__(self, model, oov=False):
        self.model = model
        self.oov = oov

    def transform(self, X, y=None, **fit_params):
        out_m = []

        for example in X:
            if example[0] in self.model:
                out_m.append(self.model[example[0]])
            elif self.oov:
                out_m.append(np.random.rand(1, self.model.vector_size)[0])
            else:
                print('Word', example[0], 'not in vocabulary')
                raise KeyError
        return np.array(out_m)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self