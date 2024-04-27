from keras.layers import Concatenate
from sklearn.ensemble import RandomForestClassifier
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


class DECISION_FOREST():
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
        self.opt_num_tree = 0

    def trainModel(self):
        step = 1 / self.embedding_dim

        max_acc = 0
        history = {
            'accuracy': [],
            'tree_number': []
        }
        max_trees = 1000
        for num in range(50, max_trees + 1, 50):
            model = RandomForestClassifier(n_estimators=num, random_state=42)
            model.fit(self.train_input, self.train_rating)
            pred = model.predict(self.val_input)
            acc = accuracy_score(np.array(self.val_rating), pred)

            print(f'num_tree: {num}, val_accuracy: {acc}')
            if acc > max_acc:
                max_acc = acc
                self.opt_num_tree = num
                joblib.dump(model, PATH + MODEL + DECISION_FOREST_MODEL)

        plt.figure()
        plt.plot(history['accuracy'], label='Valid Accuracy')
        plt.title('DECISION_FOREST Model')
        plt.ylabel('Value')
        plt.xlabel('Num_Tree')
        plt.xticks(history['tree_number'])
        plt.legend()
        plt.savefig(PATH + CHART + DECISION_FOREST_CHART)
        plt.close()
    
    def testModel(self, x_test, y_test):
        model = joblib.load(PATH + MODEL + DECISION_FOREST_MODEL)
        input = Concatenate(axis=-1)(x_test)
        input = tf.reshape(input, (input.shape[0], -1))
        labels = [np.argmax(vec) for vec in y_test]
        y_pred = model.predict(input)
        report = classification_report(labels, y_pred)
        acc = accuracy_score(labels, y_pred)
        report += f'Optimized Tree Number: {self.opt_num_tree}'
        print(report)

        with open(PATH + REPORT + DECISION_FOREST_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + DECISION_FOREST_REPORT}..................")
