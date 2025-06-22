
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.layers.activation import LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
import numpy as np
from keras.callbacks import EarlyStopping, Callback


class PrintLossCallback(Callback):  
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def train_dual_svm(X_train, Y_train, X_test, Y_test):

    model = OneVsRestClassifier(SVC(kernel='rbf', probability=True, class_weight='balanced'))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, Y_train)

    y_prob = model.predict_proba(X_test_scaled)

    return y_prob
