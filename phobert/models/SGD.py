from keras.layers import Concatenate
from sklearn.linear_model import SGDClassifier
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


class SGD():
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
            'iteration': []
        }
        
        iterations = 100
        max_acc = 0
        model = SGDClassifier(loss='hinge')
        labels = np.unique(self.train_rating)
        for i in range(iterations):
            model.partial_fit(self.train_input, self.train_rating, classes=labels)
            pred = model.predict(self.val_input)
            acc = accuracy_score(np.array(self.val_rating), pred)
            history['accuracy'].append(acc)
            history['iteration'].append(i + 1)
            print(f'interation {i+1}: val_acc: {acc}')

            if max_acc < acc:
                with open(PATH + MODEL + SGD_MODEL, 'wb') as file:
                    pickle.dump(model, file)
                    max_acc = acc

        print(f'Best Valid Accuracy: {max_acc}')

        plt.figure()
        plt.plot(history['accuracy'], label='Valid Accuracy')
        plt.title('SGD Model')
        plt.ylabel('Value')
        plt.xlabel('Iteration')
        plt.legend()
        plt.savefig(PATH + CHART + SGD_CHART)
        plt.close()
    
    def testModel(self, x_test, y_test):
        with open(PATH + MODEL + SGD_MODEL, 'rb') as f:
            model = pickle.load(f)
        input = Concatenate(axis=-1)(x_test)
        input = tf.reshape(input, (input.shape[0], -1))
        labels = [np.argmax(vec) for vec in y_test]
        y_pred = model.predict(input)
        report = classification_report(labels, y_pred)
        acc = accuracy_score(labels, y_pred)
        print(report)

        with open(PATH + REPORT + SGD_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + SGD_REPORT}..................")
