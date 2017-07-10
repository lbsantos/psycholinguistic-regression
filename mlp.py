from multiview_regression import W2VTransformer
from keras.layers import Input, Dense, Dropout, merge
from scipy.stats import spearmanr, pearsonr
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from keras.models import Sequential, Model
import numpy as np
import pandas
import gensim
import json
import keras
import os

seed = 42
np.random.seed(seed)


def build_model(neurons):
    model = Sequential()
    model.add(Dense(neurons,
                    input_shape=(neurons,),
                    activation='relu',
                    init='glorot_normal'))
    model.add(Dropout(0.5))

    model.add(Dense(neurons,
                    input_shape=(neurons,),
                    activation='relu',
                    init='glorot_normal'))
    model.add(Dropout(0.5))

    model.add(Dense(1, init='glorot_normal'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


def mse_spearman_pearson(ground_truth, predictions):
    ground_truth = ground_truth.reshape(len(ground_truth))
    predictions = predictions.reshape(len(ground_truth))
    mse = np.mean((ground_truth - predictions) ** 2)
    spearman = spearmanr(ground_truth, predictions)[0]
    pearson = pearsonr(ground_truth, predictions)[0]
    return mse, spearman, pearson


def my_cross_val_score(X, y, cv):
    cv_iter = list(cv.split(X, y))
    scores = []
    for train, test in cv_iter:
        estimator = build_model(300)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=2)
        estimator.fit(X[train],
                      y[train],
                      validation_split=0.1,
                      batch_size=2,
                      nb_epoch=100,
                      callbacks=[early_stopping])
        preds = estimator.predict(X[test]).reshape(len(test))
        ground_truth = y[test].reshape(len(test))
        scores.append(
            mse_spearman_pearson(ground_truth, preds))
    return np.array(scores)


model_embedding = gensim.models.Word2Vec.load_word2vec_format(
    './models_nathan/tokenized/skip_s300_w5_m5',
    binary=True,
    unicode_errors='ignore'
)

w2v_transform = W2VTransformer(model_embedding)

path_root = './data/'
files = ['word_lists_concretenes.csv',
         'word_lists_familiarity.csv',
         'word_lists_imagery.csv',
         'word_lists_oao.csv',
         'word_lists_aoa_final_1717.csv']

cv = KFold(5)
for file in files:
    data_path = os.path.join(path_root, file)
    data = pandas.read_csv(data_path)
    data = np.asarray(data)
    data_X = data[:, 0].reshape((len(data), -1))
    data_y = data[:, 1].reshape((len(data), -1))
    data_new2 = w2v_transform.fit_transform(data_X)
    print(data_new2.shape)
    scores = my_cross_val_score(data_new2,
                                data_y,
                                cv)
    print("ReLu + Linear")
    print(file)
    print(scores.mean(axis=0))
    print("\n======\n")