
from keras.layers import Concatenate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, log_loss
from constants import *
from tensorflow.keras import utils
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import joblib


class LOGISTIC_REGRESSION():
    def __init__(self,
            train_title,
            train_text,
            train_rating,
            val_title,
            val_text,
            val_rating,
    ):
        self.embedding_dim = train_title.shape[2]
        self.train_input = Concatenate(axis=-1)([train_title, train_text])
        self.train_input = tf.reshape(self.train_input, [self.train_input.shape[0], -1])
        self.val_input = Concatenate(axis=-1)([val_title, val_text])
        self.val_input = tf.reshape(self.val_input, [self.val_input.shape[0], -1])
        self.train_rating = train_rating
        self.train_rating = [np.argmax(vec) for vec in self.train_rating]
        self.val_rating = val_rating
        self.val_rating = [np.argmax(vec) for vec in self.val_rating]
        self.opt_gamma = 0

    def trainModel(self):
        step = 1 / self.embedding_dim

        max_acc = 0
        history = {
            'accuracy': [],
            'c': []
        }
        
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(self.train_input, self.train_rating)
        pred = model.predict(self.val_input)
        acc = accuracy_score(np.array(self.val_rating), pred)

        joblib.dump(model, PATH + MODEL + LOGISTIC_REGRESSION_MODEL)
        # plt.figure()
        # plt.plot(history['accuracy'], label='Valid Accuracy')
        # plt.title('LOGISTIC_REGRESSION Model')
        # plt.ylabel('Value')
        # plt.xlabel('C')
        # plt.xticks(history['c'])
        # plt.legend()
        # plt.savefig(PATH + CHART + LOGISTIC_REGRESSION_CHART)
        # plt.close()
    
    def testModel(self, x_test, y_test):
        model = joblib.load(PATH + MODEL + LOGISTIC_REGRESSION_MODEL)
        input = Concatenate(axis=-1)(x_test)
        input = tf.reshape(input, (input.shape[0], -1))
        labels = [np.argmax(vec) for vec in y_test]
        y_pred = model.predict(input)
        report = classification_report(labels, y_pred)
        acc = accuracy_score(labels, y_pred)
        print(report)

        with open(PATH + REPORT + LOGISTIC_REGRESSION_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + LOGISTIC_REGRESSION_REPORT}..................")
