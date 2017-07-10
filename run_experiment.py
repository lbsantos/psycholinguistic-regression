from multiview_regression.linguistic_transform import (
    FreqMetric, WordLength, FreqSubtlex, LinguisticTransform)
from multiview_regression.w2v_transformer import W2VTransformer
from multiview_regression.ensemble_regression import MeanRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
from sklearn import linear_model
import numpy as np
import pandas
import gensim
import json
import os

freq_time = time.time()
freq_nilc_embeddings = json.load(
    open('./featues/freq_nilc_embeddings.json'))

freq_subtlex = json.load(
    open('./featues/SUBTLEX-BRPOR.no-web.valid-char.json'))

freq_cbf = json.load(
    open('./featues/tbuf.json'))

freq_cbe = json.load(
    open('./featues/tbue.json'))

freq_imdb = json.load(
    open('./featues/SubIMBD_tokens_lematizados.json'))


word2vec_time = time.time()
model_embedding = gensim.models.Word2Vec.load_word2vec_format(
    './featues/models_nathan/skip_s600_w5_m5',
    binary=True,
    unicode_errors='ignore'
)

model_embedding_glove = gensim.models.Word2Vec.load_word2vec_format(
    './featues/models_nathan/vectors_300.txt',
    binary=False,
    unicode_errors='ignore'
)



def mse_spearman_pearson(ground_truth, predictions):
    ground_truth = ground_truth.reshape(len(ground_truth))
    mse = np.mean((ground_truth - predictions) ** 2)
    spearman = spearmanr(ground_truth, predictions)[0]
    pearson = pearsonr(ground_truth, predictions)[0]
    return mse, spearman, pearson


def my_cross_val_score(estimator, X, y, cv):
    cv_iter = list(cv.split(X, y))
    scores = []
    for train, test in cv_iter:
        estimator.fit(X=X[train], y=y[train])
        preds = estimator.predict(X=X[test]).reshape(len(test))
        scores.append(
            mse_spearman_pearson(y[test], preds))
    return np.array(scores)


def my_cross_val_score2(estimator, X, y, cv):
    cv_iter = list(cv.split(X, y))
    scores = []
    scores_clf1 = []
    scores_clf2 = []
    individual_results = {}
    for train, test in cv_iter:
        estimator.fit(X=X[train], y=y[train])
        scores.append(
            mse_spearman_pearson(y[test],
                                 estimator.predict(X=X[test])))

        for estimator, (name, _) in zip(estimator.estimators_,
                                        estimator.estimators):
            individual_results[name] = estimator.predict(X_new)

    return np.array(scores), np.array(scores_clf1), np.array(scores_clf2)


clf_linguistic = make_pipeline(LinguisticTransform(
    [FreqMetric(freq_nilc_embeddings),
     FreqSubtlex(freq_subtlex),
     FreqMetric(freq_cbf),
     FreqMetric(freq_cbe),
     FreqMetric(freq_imdb),
     WordLength()]),
    linear_model.ElasticNet())

clf_w2v = make_pipeline(W2VTransformer(model_embedding),
                     linear_model.Ridge())

clf_glove = make_pipeline(W2VTransformer(model_embedding_glove),
                     linear_model.ElasticNet())

clf_ensemble = MeanRegression([('linguistic', clf_linguistic),
                       ('w2v', clf_w2v),
                       ('w2v_glove', clf_glove)])
scores = list()
scores_std = list()
n_folds = 5

path_root = './data/'
files = ['word_lists_concretenes.csv',
         'word_lists_familiarity.csv',
         'word_lists_imagery.csv',
         'word_lists_oao.csv']

cv = KFold(5)
for file in files:
    data_path = os.path.join(path_root, file)
    data = pandas.read_csv(data_path)
    data = np.asarray(data)
    data_X = data[:, 0].reshape((len(data), -1))
    data_y = data[:, 1].reshape((len(data), -1))

    scores = my_cross_val_score(clf_linguistic, data_X, data_y, cv)
    print("Linguistic")
    print(file)
    print(scores.mean(axis=0))

    scores = my_cross_val_score(clf_w2v, data_X, data_y, cv)
    print("Embeddings Skip")
    print(file)
    print(scores.mean(axis=0))

    scores = my_cross_val_score(clf_glove, data_X, data_y, cv)
    print("Embeddings GloVe")
    print(file)
    print(scores.mean(axis=0))

    scores = my_cross_val_score(clf_ensemble, data_X, data_y, cv)
    print("Ensemble")
    print(file)
    print(scores.mean(axis=0))